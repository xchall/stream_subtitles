//
//  EspeakBridge.c
//  stream_subtitles
//
//  Mirrors `piper::phonemize_eSpeak` from piper-phonemize 1.1.0: one
//  `espeak_TextToPhonemesWithTerminator` call per clause, `(lang)` switch flags
//  dropped, clause punctuation appended and sentences separated by '\n'.
//

#include "EspeakBridge.h"

#include <espeak-ng/speak_lib.h>

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Values from espeak-ng's internal translate.h. They are part of the library's
// documented terminator encoding: the low 20 bits identify the clause.
#define BR_CLAUSE_PERIOD       (40 | 0x00080000)
#define BR_CLAUSE_QUESTION     (41 | 0x00002000 | 0x00080000)
#define BR_CLAUSE_EXCLAMATION  (42 | 0x00003000 | 0x00080000)
#define BR_CLAUSE_COMMA        (20 | 0x00001000 | 0x00040000)
#define BR_CLAUSE_COLON        (21 | 0x00001000 | 0x00040000)
#define BR_CLAUSE_SEMICOLON    (22 | 0x00001000 | 0x00040000)
#define BR_CLAUSE_TYPE_SENTENCE 0x00080000
#define BR_CLAUSE_MASK         0x000FFFFF

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_sample_rate = 0;
static char g_voice[64] = {0};

// MARK: - Growable string

typedef struct {
    char *bytes;
    size_t length;
    size_t capacity;
} Buffer;

static bool buffer_reserve(Buffer *buffer, size_t extra) {
    size_t needed = buffer->length + extra + 1; // + NUL
    if (needed <= buffer->capacity) {
        return true;
    }
    size_t capacity = buffer->capacity ? buffer->capacity : 256;
    while (capacity < needed) {
        capacity *= 2;
    }
    char *grown = realloc(buffer->bytes, capacity);
    if (grown == NULL) {
        return false;
    }
    buffer->bytes = grown;
    buffer->capacity = capacity;
    return true;
}

static bool buffer_append(Buffer *buffer, const char *bytes, size_t count) {
    if (!buffer_reserve(buffer, count)) {
        return false;
    }
    memcpy(buffer->bytes + buffer->length, bytes, count);
    buffer->length += count;
    buffer->bytes[buffer->length] = '\0';
    return true;
}

static bool buffer_append_string(Buffer *buffer, const char *string) {
    return buffer_append(buffer, string, strlen(string));
}

// MARK: - Public API

int esb_init(const char *dataDirectory) {
    if (dataDirectory == NULL || dataDirectory[0] == '\0') {
        return -1;
    }

    pthread_mutex_lock(&g_lock);
    if (g_sample_rate <= 0) {
        int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataDirectory, 0);
        if (rate <= 0) {
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        g_sample_rate = rate;
    }
    int rate = g_sample_rate;
    pthread_mutex_unlock(&g_lock);
    return rate;
}

int esb_set_voice(const char *voice) {
    if (voice == NULL) {
        return -1;
    }

    pthread_mutex_lock(&g_lock);
    int result = -1;
    if (g_sample_rate > 0) {
        if (strncmp(g_voice, voice, sizeof(g_voice)) == 0) {
            result = 0;
        } else if (espeak_SetVoiceByName(voice) == 0) {
            strncpy(g_voice, voice, sizeof(g_voice) - 1);
            g_voice[sizeof(g_voice) - 1] = '\0';
            result = 0;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

char *esb_phonemize(const char *text, const char *voice) {
    if (text == NULL || voice == NULL) {
        return NULL;
    }

    pthread_mutex_lock(&g_lock);
    if (g_sample_rate <= 0) {
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }

    if (strncmp(g_voice, voice, sizeof(g_voice)) != 0) {
        if (espeak_SetVoiceByName(voice) != 0) {
            pthread_mutex_unlock(&g_lock);
            return NULL;
        }
        strncpy(g_voice, voice, sizeof(g_voice) - 1);
        g_voice[sizeof(g_voice) - 1] = '\0';
    }

    Buffer out = {0};
    if (!buffer_reserve(&out, 0)) {
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }
    out.bytes[0] = '\0';

    const void *cursor = text;
    bool ok = true;
    while (cursor != NULL && ok) {
        int terminator = 0;
        const char *phonemes = espeak_TextToPhonemesWithTerminator(
            &cursor, espeakCHARS_AUTO, espeakPHONEMES_IPA, &terminator);
        if (phonemes == NULL) {
            break;
        }

        // Drop "(en)" style language switches: the current voice cannot
        // pronounce them and piper removes them too.
        bool in_flag = false;
        const char *chunk = phonemes;
        while (*chunk != '\0' && ok) {
            if (in_flag) {
                if (*chunk == ')') {
                    in_flag = false;
                }
                chunk++;
                continue;
            }
            if (*chunk == '(') {
                in_flag = true;
                chunk++;
                continue;
            }
            // Append the longest run that is not a flag.
            const char *start = chunk;
            while (*chunk != '\0' && *chunk != '(') {
                chunk++;
            }
            ok = buffer_append(&out, start, (size_t)(chunk - start));
        }
        if (!ok) {
            break;
        }

        int punctuation = terminator & BR_CLAUSE_MASK;
        const char *mark = NULL;
        switch (punctuation) {
            case BR_CLAUSE_PERIOD: mark = "."; break;
            case BR_CLAUSE_QUESTION: mark = "?"; break;
            case BR_CLAUSE_EXCLAMATION: mark = "!"; break;
            case BR_CLAUSE_COMMA: mark = ","; break;
            case BR_CLAUSE_COLON: mark = ":"; break;
            case BR_CLAUSE_SEMICOLON: mark = ";"; break;
            default: break;
        }
        if (mark != NULL) {
            ok = buffer_append_string(&out, mark);
            if (ok && mark[0] != '.' && mark[0] != '?' && mark[0] != '!') {
                ok = buffer_append_string(&out, " ");
            }
        }

        if ((terminator & BR_CLAUSE_TYPE_SENTENCE) == BR_CLAUSE_TYPE_SENTENCE) {
            ok = buffer_append_string(&out, "\n");
        }
    }

    // Drop the sentence break that always follows the last clause.
    while (out.length > 0 && out.bytes[out.length - 1] == '\n') {
        out.length--;
        out.bytes[out.length] = '\0';
    }

    pthread_mutex_unlock(&g_lock);

    if (!ok) {
        free(out.bytes);
        return NULL;
    }
    return out.bytes;
}

void esb_free(char *text) {
    free(text);
}
