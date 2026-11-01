import Foundation

/// espeak-ng front-end for the bundled Piper voices.
///
/// Desktop Piper never phonemises text itself: `piper.exe` and `piper-phonemize`
/// call espeak-ng and feed the resulting IPA symbols to the ONNX graph. The
/// voices are trained on exactly those symbols, so this is the only way to get
/// the same pronunciation on iOS. The rules-based phonemizers stay around as a
/// fallback for the (unlikely) case when the espeak data is missing.
struct EspeakPhonemizer: Phonemizer {
    /// espeak voice name from the voice's `.onnx.json` (`en-us`, `ru`, ...).
    let voice: String

    /// Returns nil when espeak-ng could not be initialised or the voice is
    /// unknown, so the caller can fall back to the rule-based phonemizer.
    static func make(voice: String) -> EspeakPhonemizer? {
        guard EspeakRuntime.isReady, esb_set_voice(voice) == 0 else { return nil }
        return EspeakPhonemizer(voice: voice)
    }

    func phonemes(for text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard let raw = esb_phonemize(trimmed, voice) else { return [] }
        defer { esb_free(raw) }

        // piper normalises every clause to NFD before looking symbols up in
        // `phoneme_id_map`, so precomposed characters become base letter +
        // combining mark and match the ids the voice was trained on.
        let ipa = String(cString: raw).decomposedStringWithCanonicalMapping

        var tokens: [String] = ["^"]
        for scalar in ipa.unicodeScalars {
            if scalar == " " || scalar == "\n" {
                if tokens.last != " " { tokens.append(" ") }
            } else {
                tokens.append(String(scalar))
            }
        }
        while tokens.last == " " { tokens.removeLast() }
        tokens.append("$")
        return tokens
    }
}

/// One-time espeak-ng setup: locates `espeak-ng-data` in the bundle and
/// initialises the library.
enum EspeakRuntime {
    static let sampleRate: Int = initialize()
    static var isReady: Bool { sampleRate > 0 }

    private static func initialize() -> Int {
        guard let directory = dataDirectory else {
            NSLog("espeak-ng: espeak-ng-data not found in the app bundle")
            return -1
        }
        let rate = Int(esb_init(directory))
        if rate <= 0 {
            NSLog("espeak-ng: espeak_Initialize failed (%d) for %@", rate, directory)
        } else {
            NSLog("espeak-ng %d Hz, data at %@", rate, directory)
        }
        return rate
    }

    /// espeak-ng expects the directory that *contains* `espeak-ng-data`.
    private static var dataDirectory: String? {
        var candidates: [URL] = []
        if let resources = Bundle.main.resourceURL { candidates.append(resources) }
        candidates.append(Bundle.main.bundleURL)
        candidates.append(
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        )

        let fileManager = FileManager.default
        for candidate in candidates {
            let probe = candidate.appendingPathComponent("espeak-ng-data/en_dict")
            if fileManager.fileExists(atPath: probe.path) {
                return candidate.path
            }
        }
        return nil
    }
}
