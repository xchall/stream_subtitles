import Foundation
import OnnxRuntimeBindings

/// Streaming (no VAD) RNN-T recognizer built on the exported icefall LSTM RNN-T ONNX graphs.
///
/// Exactly like the PyTorch reference implementation:
///  * every 40 ms the encoder consumes a 9-frame (90 ms) window of fbank features,
///  * the LSTM state is carried from step to step,
///  * the encoder output (1 frame) is greedily decoded with at most 3 symbols per frame,
///  * pauses are decoded too: the model simply emits nothing while nobody talks.
final class StreamingRNNT: StreamingRecognizer {
    static let windowFrames = 9
    static let strideFrames = 4
    static let maxSymbolsPerFrame = 3
    static let contextSize = 2
    static let blankId = 0

    private let env: ORTEnv
    private let encoder: ORTSession
    private let decoder: ORTSession
    private let joiner: ORTSession
    private let tokens: [String]

    private let fbank = FbankStream()
    private var pendingFrames: [[Float]] = []

    private var state0: ORTValue
    private var state1: ORTValue
    private var decoderOut: ORTValue?
    private var decoderContext: [Int] = []

    private(set) var hypothesis: [Int] = []
    private(set) var text: String = ""

    /// Called on the recognizer's own queue whenever new text is available.
    var onText: ((String) -> Void)?

    // MARK: - Setup

    init(encoderURL: URL, decoderURL: URL, joinerURL: URL, tokensURL: URL,
         intraOpThreads: Int = 2) throws {
        env = try ORTEnv(loggingLevel: .warning)

        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(Int32(intraOpThreads))
        //try options.setInterOpNumThreads(1)

        encoder = try ORTSession(env: env, modelPath: encoderURL.path, sessionOptions: options)
        decoder = try ORTSession(env: env, modelPath: decoderURL.path, sessionOptions: options)
        joiner = try ORTSession(env: env, modelPath: joinerURL.path, sessionOptions: options)

        tokens = Self.loadTokens(from: tokensURL)

        state0 = try Self.zeroTensor(shape: [12, 1, 512])
        state1 = try Self.zeroTensor(shape: [12, 1, 1024])
        hypothesis = [Self.blankId, Self.blankId]
        decoderContext = hypothesis.suffix(Self.contextSize).map { $0 }
        decoderOut = try runDecoder(context: decoderContext)
    }

    private static func loadTokens(from url: URL) -> [String] {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        // Each line is "<token> <id>"; tokens may themselves contain spaces.
        return content.split(separator: "\n", omittingEmptySubsequences: false).map { line in
            guard let space = line.lastIndex(of: " ") else { return String(line) }
            return String(line[line.startIndex..<space])
        }
    }

    // MARK: - Audio in

    /// Feeds 16 kHz mono float samples. Returns nothing; text is delivered through `onText`.
    func accept(_ samples: [Float]) {
        pendingFrames.append(contentsOf: fbank.push(samples))
        while pendingFrames.count >= Self.windowFrames {
            let window = Array(pendingFrames.prefix(Self.windowFrames))
            pendingFrames.removeFirst(Self.strideFrames)
            runWindow(window)
        }
    }

    /// Pushes trailing silence so the last words of the stream are flushed.
    func flush(tailSeconds: Double = 0.5) {
        let count = Int(tailSeconds * Double(FbankComputer.sampleRate))
        accept([Float](repeating: 0, count: count))
    }

    func reset() {
        fbank.reset()
        pendingFrames.removeAll(keepingCapacity: true)
        hypothesis = [Self.blankId, Self.blankId]
        decoderContext = hypothesis
        decoderOut = try? runDecoder(context: decoderContext)
        state0 = (try? Self.zeroTensor(shape: [12, 1, 512])) ?? state0
        state1 = (try? Self.zeroTensor(shape: [12, 1, 1024])) ?? state1
        text = ""
        onText?("")
    }

    // MARK: - One encoder step

    private func runWindow(_ frames: [[Float]]) {
        do {
            var input = [Float]()
            input.reserveCapacity(Self.windowFrames * FbankComputer.numMelBins)
            for frame in frames { input.append(contentsOf: frame) }

            let x = try Self.floatTensor(input, shape: [1, Self.windowFrames, FbankComputer.numMelBins])
            let outputs = try encoder.run(
                withInputs: ["x": x, "state0": state0, "state1": state1],
                outputNames: ["encoder_out", "new_state0", "new_state1"],
                runOptions: nil as ORTRunOptions?
            )

            guard let encoderOut = outputs["encoder_out"],
                  let newState0 = outputs["new_state0"],
                  let newState1 = outputs["new_state1"] else { return }

            state0 = newState0
            state1 = newState1

            let encoded = try Self.floats(from: encoderOut)
            guard encoded.count >= 512 else { return }
            let encFrame = Array(encoded.suffix(512))

            var emitted = false
            for _ in 0..<Self.maxSymbolsPerFrame {
                let encTensor = try Self.floatTensor(encFrame, shape: [1, 512])
                let dec = try currentDecoderOutput()
                let logits = try joiner.run(
                    withInputs: ["encoder_out": encTensor, "decoder_out": dec],
                    outputNames: ["logits"],
                    runOptions: nil as ORTRunOptions?
                )
                guard let logitsValue = logits["logits"] else { break }
                let token = try Self.argmax(logitsValue)
                if token == Self.blankId { break }

                hypothesis.append(token)
                appendTokenText(token)

                decoderContext = Array(hypothesis.suffix(Self.contextSize))
                decoderOut = try runDecoder(context: decoderContext)
                emitted = true
            }
            if emitted { onText?(text) }
        } catch {
            NSLog("StreamingRNNT step failed: \(error)")
        }
    }

    private func appendTokenText(_ id: Int) {
        guard id > 0, id < tokens.count else { return }
        var piece = tokens[id].replacingOccurrences(of: "\u{2581}", with: " ")
        if text.isEmpty {
            while piece.hasPrefix(" ") { piece.removeFirst() }
        }
        text.append(contentsOf: piece)
    }

    /// Decoder output for the current prediction context (recomputed only when it changes).
    private func currentDecoderOutput() throws -> ORTValue {
        if let decoderOut, decoderContext == Array(hypothesis.suffix(Self.contextSize)) {
            return decoderOut
        }
        let out = try runDecoder(context: decoderContext)
        decoderOut = out
        return out
    }

    private func runDecoder(context: [Int]) throws -> ORTValue {
        var ctx = context
        while ctx.count < Self.contextSize { ctx.insert(Self.blankId, at: 0) }
        let data = ctx.map { Int64($0) }
        let tensor = try ORTValue(
            tensorData: NSMutableData(bytes: data, length: data.count * MemoryLayout<Int64>.stride),
            elementType: .int64,
            shape: [1, NSNumber(value: Self.contextSize)]
        )
        let out = try decoder.run(
            withInputs: ["y": tensor],
            outputNames: ["decoder_out"],
            runOptions: nil as ORTRunOptions?
        )
        guard let value = out["decoder_out"] else {
            throw NSError(domain: "StreamingRNNT", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "decoder_out missing"])
        }
        return value
    }

    // MARK: - Tensor helpers

    private static func zeroTensor(shape: [Int]) throws -> ORTValue {
        var count = 1
        for d in shape { count *= d }
        let data = NSMutableData(length: count * MemoryLayout<Float>.stride)!
        return try ORTValue(tensorData: data, elementType: .float,
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

    private static func argmax(_ value: ORTValue) throws -> Int {
        let values = try floats(from: value)
        guard !values.isEmpty else { return blankId }
        var best = -Float.greatestFiniteMagnitude
        var bestIndex = 0
        for (i, v) in values.enumerated() where v > best {
            best = v
            bestIndex = i
        }
        return bestIndex
    }
}
