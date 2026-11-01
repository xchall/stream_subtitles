import Foundation

/// Sidecar configuration that ships next to every Piper voice
/// (`<voice>.onnx.json`). It carries the phoneme → id table, the sample rate and
/// the VITS defaults; the weights themselves live in the `.onnx` file.
struct PiperVoiceConfig: Decodable {
    struct Audio: Decodable {
        let sampleRate: Int
        let quality: String?

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case quality
        }
    }

    struct Inference: Decodable {
        let noiseScale: Double?
        let lengthScale: Double?
        let noiseW: Double?

        enum CodingKeys: String, CodingKey {
            case noiseScale = "noise_scale"
            case lengthScale = "length_scale"
            case noiseW = "noise_w"
        }
    }

    struct LanguageInfo: Decodable {
        let code: String?
        let family: String?
        let nameEnglish: String?

        enum CodingKeys: String, CodingKey {
            case code
            case family
            case nameEnglish = "name_english"
        }
    }

    /// espeak-ng voice the training phonemes were generated with.
    struct Espeak: Decodable {
        let voice: String
    }

    let audio: Audio
    let inference: Inference?
    let espeak: Espeak?
    let phonemeIDMap: [String: [Int]]
    let numSymbols: Int?
    let language: LanguageInfo?

    enum CodingKeys: String, CodingKey {
        case audio
        case inference
        case espeak
        case language
        case phonemeIDMap = "phoneme_id_map"
        case numSymbols = "num_symbols"
    }

    var sampleRate: Double { Double(audio.sampleRate) }
    var noiseScale: Float { Float(inference?.noiseScale ?? 0.667) }
    var noiseW: Float { Float(inference?.noiseW ?? 0.8) }
    var defaultLengthScale: Float { Float(inference?.lengthScale ?? 1.0) }
}

/// Names and locations of the Piper voices bundled into the app.
///
/// The Xcode target uses a synchronized folder, so everything under
/// `stream_subtitles/Models/tts` is copied flat into the app bundle root.
/// Both layouts (flat and nested) are probed, which keeps the app working when
/// the folder is added as a real folder reference instead.
enum PiperVoice {
    /// Voice id = file name without the `.onnx` suffix.
    enum ID: String, CaseIterable {
        case english = "en_US-ryan-high"
        case russian = "ru_RU-ruslan-medium"

        static func forLanguage(_ language: AppLanguage) -> ID? {
            switch language {
            case .english: return .english
            case .russian: return .russian
            case .french, .spanish: return nil
            }
        }
    }

    struct Files {
        let model: URL
        let config: URL
    }

    static func files(for language: AppLanguage) throws -> Files {
        guard let id = ID.forLanguage(language) else {
            throw NSError(domain: "PiperVoice", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Нет голоса Piper для языка \(language.title)"
            ])
        }
        return try files(for: id)
    }

    static func files(for id: ID) throws -> Files {
        guard let model = locateModel(id) else {
            throw NSError(domain: "PiperVoice", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "В бандле не найден голос \(id.rawValue).onnx"
            ])
        }
        guard let config = locateConfig(id, near: model) else {
            throw NSError(domain: "PiperVoice", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "В бандле не найден \(id.rawValue).onnx.json"
            ])
        }
        return Files(model: model, config: config)
    }

    // MARK: - Bundle lookup

    private static let subdirectories = ["", "Models/tts", "Models", "tts"]

    private static func locateModel(_ id: ID) -> URL? {
        let bundle = Bundle.main
        for directory in subdirectories {
            let url = directory.isEmpty
                ? bundle.url(forResource: id.rawValue, withExtension: "onnx")
                : bundle.url(forResource: id.rawValue, withExtension: "onnx", subdirectory: directory)
            if let url, FileManager.default.fileExists(atPath: url.path) { return url }
        }
        // Development fallback: the model next to the built executable.
        let local = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent()
            .appendingPathComponent("\(id.rawValue).onnx")
        return FileManager.default.fileExists(atPath: local.path) ? local : nil
    }

    private static func locateConfig(_ id: ID, near model: URL) -> URL? {
        // The file is literally called `<voice>.onnx.json`, so the resource name
        // keeps the `.onnx` part and the extension is `json`.
        let name = "\(id.rawValue).onnx"
        let sibling = model.deletingLastPathComponent().appendingPathComponent("\(name).json")
        if FileManager.default.fileExists(atPath: sibling.path) { return sibling }

        let bundle = Bundle.main
        for directory in subdirectories {
            let url = directory.isEmpty
                ? bundle.url(forResource: name, withExtension: "json")
                : bundle.url(forResource: name, withExtension: "json", subdirectory: directory)
            if let url, FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}
