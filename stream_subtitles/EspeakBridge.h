//
//  EspeakBridge.h
//  stream_subtitles
//
//  Minimal C API over espeak-ng. Piper voices consume espeak-ng phonemes, so
//  this is the same front-end `piper.exe` uses on the desktop, compiled for
//  iOS. Swift only sees these four functions; the espeak headers stay out of
//  the bridging header.
//

#ifndef ESPEAK_BRIDGE_H
#define ESPEAK_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/// Initialises espeak-ng once. `dataDirectory` must be the directory that
/// *contains* `espeak-ng-data` (i.e. the app bundle root).
/// Returns the synthesis sample rate (> 0) on success, or a negative value.
int esb_init(const char *dataDirectory);

/// Selects an espeak voice by name ("en-us", "ru", ...). Cached, so repeated
/// calls with the same name are cheap. Returns 0 on success.
int esb_set_voice(const char *voice);

/// Phonemises `text` with the given voice and returns IPA as a malloc'ed
/// UTF-8 string, or NULL on failure. The caller must free it with `esb_free`.
///
/// The result follows piper's layout: words are separated by a space, clauses
/// end with their punctuation (".", "?", "!", ",", ":", ";") and sentences are
/// separated by '\n'. Nothing else is normalised here — the Swift side applies
/// NFD and maps the symbols to voice ids.
char *esb_phonemize(const char *text, const char *voice);

/// Frees a string returned by `esb_phonemize`.
void esb_free(char *text);

#ifdef __cplusplus
}
#endif

#endif /* ESPEAK_BRIDGE_H */
