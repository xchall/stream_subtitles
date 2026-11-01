import Foundation
import OnnxRuntimeBindings

/// Piper (VITS) text-to-speech backed by one of the bundled ONNX voices.
///
/// Graph contract (identical for `en_US-ryan-high` and `ru_RU-ruslan-medium`):
///
///     input          int64   (1, T)   phoneme ids from `phoneme_id_map`
///     input_lengths  int64   (1,)     T
///     scales         float32 (3,)     [noise_scale, length_scale, noise_w]
///     output         float32 (1,1,1,S) 22 050 Hz mono audio
///
/// One instance owns one ORT session and must not be used concurrently: the
/// speaker runs every call on its own serial queue.
final class PiperTTS {
    let config: PiperVoiceConfig
    let sampleRate: Double

    private let env: ORTEnv
    private let session: ORTSession
    private let phonemizer: Phonemizer

    init(modelURL: URL, configURL: URL, language: AppLanguage) throws {
        let data = try Data(contentsOf: configURL)
        config = try JSONDecoder().decode(PiperVoiceConfig.self, from: data)
        sampleRate = config.sampleRate

        guard let phonemizer = Phonemizers.make(for: language, voice: config.espeak?.voice) else {
            throw NSError(domain: "PiperTTS", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Нет фонемизатора для языка \(language.title)"
            ])
        }
        self.phonemizer = phonemizer

        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(2)
        session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
    }

    static func make(for language: AppLanguage) throws -> PiperTTS {
        let files = try PiperVoice.files(for: language)
        return try PiperTTS(modelURL: files.model, configURL: files.config, language: language)
    }

    /// Debug helper: the IPA tokens that would be fed to the voice.
    func phonemes(for text: String) -> [String] {
        phonemizer.phonemes(for: text)
    }

    /// Synthesises `text` into mono float samples at `sampleRate`.
    func synthesize(text: String, speed: Double = 1.0) throws -> [Float] {
        let tokens = phonemizer.phonemes(for: text)
        let ids = phonemeIDs(for: tokens)
        NSLog("PiperTTS %d ids from %d tokens: %@", ids.count, tokens.count,
              tokens.joined(separator: " "))
        guard ids.count > 1 else { return [] }

        let input = try Self.int64Tensor(ids.map(Int64.init), shape: [1, ids.count])
        let lengths = try Self.int64Tensor([Int64(ids.count)], shape: [1])
        let lengthScale = config.defaultLengthScale / Float(max(speed, 0.1))
        let scales = try Self.floatTensor(
            [config.noiseScale, lengthScale, config.noiseW],
            shape: [3]
        )

        let outputs = try session.run(
            withInputs: [
                "input": input,
                "input_lengths": lengths,
                "scales": scales
            ],
            outputNames: ["output"],
            runOptions: nil as ORTRunOptions?
        )

        guard let value = outputs["output"] else {
            throw NSError(domain: "PiperTTS", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "ONNX-граф не вернул output"
            ])
        }
        return Self.limited(try Self.floats(from: value))
    }

    // MARK: - Phoneme ids

    /// Piper's layout (`piper::phonemes_to_ids`):
    ///
    ///     ^  _  p  _  p  _  …  p  _  $
    ///
    /// A pad id (`_`) follows the BOS symbol and every phoneme. The voices were
    /// trained with those pads; without them the duration predictor collapses
    /// and the result is rushed, unintelligible audio.
    private func phonemeIDs(for tokens: [String]) -> [Int] {
        var ids: [Int] = []
        ids.reserveCapacity(tokens.count * 2 + 2)

        let pad = config.phonemeIDMap["_"] ?? []
        ids.append(contentsOf: config.phonemeIDMap["^"] ?? [])
        ids.append(contentsOf: pad)

        for token in tokens where token != "^" && token != "$" {
            appendIDs(for: token, to: &ids)
            ids.append(contentsOf: pad)
        }

        ids.append(contentsOf: config.phonemeIDMap["$"] ?? [])
        return ids
    }

    private func appendIDs(for token: String, to ids: inout [Int]) {
        if let mapped = config.phonemeIDMap[token] {
            ids.append(contentsOf: mapped)
            return
        }
        // The map is keyed by single IPA symbols, so split a stray digraph.
        for scalar in token.unicodeScalars {
            if let mapped = config.phonemeIDMap[String(scalar)] {
                ids.append(contentsOf: mapped)
            }
        }
    }

    // MARK: - Tensors

    private static func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buffer in
            NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Int64>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .int64,
                            shape: shape.map { NSNumber(value: $0) })
    }

    private static func floatTensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buffer in
            NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Float>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .float,
                            shape: shape.map { NSNumber(value: $0) })
    }

    private static func floats(from value: ORTValue) throws -> [Float] {
        let data = try value.tensorData()
        return (data as Data).withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    /// VITS occasionally overshoots ±1.0; scale down instead of hard clipping.
    private static func limited(_ samples: [Float]) -> [Float] {
        var peak: Float = 0
        for sample in samples { peak = max(peak, abs(sample)) }
        guard peak > 0.99 else { return samples }
        let gain = 0.99 / peak
        return samples.map { $0 * gain }
    }
}
