import Foundation

/// Turns plain text into the IPA token sequence a Piper voice expects.
///
/// Piper voices are trained on espeak-ng phonemes, so the ONNX graph takes IPA
/// symbols (mapped to ids by `phoneme_id_map`), not letters. `EspeakPhonemizer`
/// is that front-end: the real espeak-ng, compiled into the app as
/// `Vendor/espeak/espeakng.xcframework`. Nothing here guesses pronunciation.
protocol Phonemizer {
    /// Returns IPA tokens including the sentence markers `^` (BOS) and `$` (EOS)
    /// and ` ` between words, exactly as `piper-phonemize` does.
    func phonemes(for text: String) -> [String]
}

enum Phonemizers {
    /// `voice` is the espeak voice name stored in the Piper voice config
    /// (`en-us`, `ru`, ...). Returns nil when espeak-ng or that voice is
    /// unavailable.
    static func make(for language: AppLanguage, voice: String? = nil) -> Phonemizer? {
        switch language {
        case .english, .russian:
            guard let voice, let espeak = EspeakPhonemizer.make(voice: voice) else {
                NSLog("espeak-ng: voice %@ unavailable", voice ?? language.rawValue)
                return nil
            }
            return espeak
        case .french, .spanish:
            return nil
        }
    }
}

/// Shared text cleanup for the TTS input. espeak-ng expands digits and reads
/// punctuation itself, so only stray punctuation glued to a word is removed.
enum SpeechText {
    /// Drops punctuation/whitespace at both ends of a word, keeping the inside.
    static func sanitize(_ word: String) -> String {
        let allowed = CharacterSet.letters.union(.decimalDigits)
        var scalars = Array(word.lowercased().unicodeScalars)
        while let first = scalars.first, !allowed.contains(first) { scalars.removeFirst() }
        while let last = scalars.last, !allowed.contains(last) { scalars.removeLast() }
        return String(String.UnicodeScalarView(scalars))
    }
}
