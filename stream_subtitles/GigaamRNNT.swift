import Foundation
import OnnxRuntimeBindings

/// Russian streaming RNN-T recognizer built on the exported GigaAM v3 RNN-T
/// ONNX graphs (`v3_rnnt_encoder.onnx`, `v3_rnnt_decoder.onnx`,
/// `v3_rnnt_joint.onnx`).
///
/// The exported encoder has no recurrent state: it consumes a whole log-mel
/// chunk `[1, 64, T]` and returns `[1, 768, T/4]`. To stay streaming we feed it
/// overlapping windows: every 1.28 s of new audio is encoded together with
/// 0.64 s of already-encoded left context, whose outputs are dropped. The last
/// output frame of each window is withheld and re-encoded with the next one so
/// the subsampling convolutions never see a hard cut.
///
/// Decoding is plain greedy RNN-T, exactly like `gigaam.onnx_utils._decode_rnnt_batch`:
///  * at most 3 symbols per encoder frame,
///  * the prediction network state advances only when a symbol is emitted,
///  * blank id is 33, the vocabulary is 33 characters (space + 32 letters).
final class GigaamRNNT: StreamingRecognizer {
    static let blankId = 33
    static let maxSymbolsPerFrame = 3

    /// New audio encoded per step (128 mel frames = 1.28 s).
    static let chunkFrames = 128
    /// Already-decoded audio prepended as left context (64 mel frames = 0.64 s).
    static let contextFrames = 64

    static let encoderHidden = 768
    static let decoderHidden = 320
    static let numClasses = 34

    /// GigaAM vocabulary: id 0 is space, 1...32 are lowercase Russian letters.
    static let vocabulary: [String] = " абвгдежзийклмнопрстуфхцчшщъыьэюя".map(String.init)

    private let env: ORTEnv
    private let encoder: ORTSession
    private let decoder: ORTSession
    private let joint: ORTSession

    private let fbank = GigaamFbankStream()

    /// Log-mel frames not yet dropped, `melStart` is the absolute index of the first.
    private var melFrames: [[Float]] = []
    private var melStart = 0
    /// Absolute index of the next encoder output frame to decode.
    private var nextOutputFrame = 0
    /// Absolute index of the next log-mel frame covered by a decoded output.
    private var nextMelToEncode = 0

    private var predH: ORTValue
    private var predC: ORTValue
    private var predOut: ORTValue

    private(set) var text: String = ""
    var onText: ((String) -> Void)?

    // MARK: - Setup

    init(encoderURL: URL, decoderURL: URL, jointURL: URL,
         intraOpThreads: Int = 4) throws {
        env = try ORTEnv(loggingLevel: .warning)

        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(Int32(intraOpThreads))

        encoder = try ORTSession(env: env, modelPath: encoderURL.path, sessionOptions: options)
        decoder = try ORTSession(env: env, modelPath: decoderURL.path, sessionOptions: options)
        joint = try ORTSession(env: env, modelPath: jointURL.path, sessionOptions: options)

        let h = try Self.zeroTensor(shape: [1, 1, Self.decoderHidden])
        let c = try Self.zeroTensor(shape: [1, 1, Self.decoderHidden])
        let started = try Self.runDecoder(decoder, label: Self.blankId, h: h, c: c)
        predH = h
        predC = c
        predOut = started.out
    }

    // MARK: - Audio in

    func accept(_ samples: [Float]) {
        melFrames.append(contentsOf: fbank.push(samples))

        while let window = makeWindow() {
            runWindow(window.frames, windowStartMel: window.start, final: false)
        }
    }

    /// Pushes trailing silence and encodes whatever is left.
    func flush(tailSeconds: Double = 0.5) {
        let count = Int(tailSeconds * Double(GigaamFbankComputer.sampleRate))
        accept([Float](repeating: 0, count: count))
        runRemaining()
    }

    func reset() {
        fbank.reset()
        melFrames.removeAll(keepingCapacity: true)
        melStart = 0
        nextOutputFrame = 0
        nextMelToEncode = 0

        if let h = try? Self.zeroTensor(shape: [1, 1, Self.decoderHidden]),
           let c = try? Self.zeroTensor(shape: [1, 1, Self.decoderHidden]) {
            predH = h
            predC = c
            if let started = try? Self.runDecoder(decoder, label: Self.blankId, h: h, c: c) {
                predOut = started.out
            }
        }

        text = ""
        onText?("")
    }

    // MARK: - Windowing

    private struct Window {
        let frames: [[Float]]
        let start: Int        // absolute mel frame index of frames[0]
    }

    /// A window is ready once `chunkFrames` new mel frames exist past the last
    /// decoded point. Windows always start/end on multiples of the subsampling
    /// factor (4), so output frame `i` maps to global mel frame `start + 4 * i`.
    private func makeWindow() -> Window? {
        let runEnd = nextMelToEncode + Self.chunkFrames
        guard runEnd <= melStart + melFrames.count else { return nil }

        let start = Self.alignDown(max(0, nextMelToEncode - Self.contextFrames))
        let from = start - melStart
        let to = runEnd - melStart
        guard from >= 0, to <= melFrames.count, to > from else { return nil }
        return Window(frames: Array(melFrames[from..<to]), start: start)
    }

    /// Encodes the mel frames that are still waiting, padding with the last
    /// frame so the length stays a multiple of 4.
    private func runRemaining() {
        guard !melFrames.isEmpty else { return }

        let start = Self.alignDown(max(0, nextMelToEncode - Self.contextFrames))
        let from = start - melStart
        guard from >= 0, from < melFrames.count else { return }

        var frames = Array(melFrames[from...])
        let targetEnd = Self.alignUp(melStart + melFrames.count, 4)
        while start + frames.count < targetEnd, let last = frames.last {
            frames.append(last)
        }
        runWindow(frames, windowStartMel: start, final: true)
    }

    // MARK: - One encoder step

    private func runWindow(_ window: [[Float]], windowStartMel: Int, final: Bool) {
        let t = window.count
        guard t >= 4, t % 4 == 0 else { return }
        let tOut = t / 4

        do {
            // The encoder wants (N, 64, T): feature-major, not frame-major.
            var input = [Float]()
            input.reserveCapacity(GigaamFbankComputer.numMelBins * t)
            for b in 0..<GigaamFbankComputer.numMelBins {
                for frame in window {
                    input.append(frame[b])
                }
            }
            let audio = try Self.floatTensor(input, shape: [1, GigaamFbankComputer.numMelBins, t])
            let length = try Self.int64Tensor([Int64(t)], shape: [1])

            let outputs = try encoder.run(
                withInputs: ["audio_signal": audio, "length": length],
                outputNames: ["encoded", "encoded_len"],
                runOptions: nil as ORTRunOptions?
            )
            guard let encodedValue = outputs["encoded"] else { return }
            let encoded = try Self.floats(from: encodedValue)

            let windowStartOut = windowStartMel / 4
            let firstLocal = max(0, nextOutputFrame - windowStartOut)
            let lastLocal = final ? tOut : tOut - 1
            guard lastLocal > firstLocal else { return }

            for local in firstLocal..<lastLocal {
                decodeFrame(encoded, tOut: tOut, index: local)
            }

            nextOutputFrame = windowStartOut + lastLocal
            nextMelToEncode = nextOutputFrame * 4
            pruneBuffer()
        } catch {
            NSLog("GigaamRNNT step failed: \(error)")
        }
    }

    /// Greedy RNN-T decode of a single encoder frame.
    private func decodeFrame(_ encoded: [Float], tOut: Int, index: Int) {
        do {
            var frame = [Float](repeating: 0, count: Self.encoderHidden)
            for c in 0..<Self.encoderHidden {
                frame[c] = encoded[c * tOut + index]
            }
            let encTensor = try Self.floatTensor(frame, shape: [1, Self.encoderHidden, 1])
            var emitted = false

            for _ in 0..<Self.maxSymbolsPerFrame {
                let dec = try Self.floats(from: predOut)
                let decTensor = try Self.floatTensor(dec, shape: [1, Self.decoderHidden, 1])
                let outputs = try joint.run(
                    withInputs: ["enc": encTensor, "dec": decTensor],
                    outputNames: ["joint"],
                    runOptions: nil as ORTRunOptions?
                )
                guard let logits = outputs["joint"] else { break }
                let token = try Self.argmax(logits)
                if token == Self.blankId { break }

                appendToken(token)

                let next = try Self.runDecoder(decoder, label: token, h: predH, c: predC)
                predOut = next.out
                predH = next.h
                predC = next.c
                emitted = true
            }

            if emitted { onText?(text) }
        } catch {
            NSLog("GigaamRNNT decode failed: \(error)")
        }
    }

    private func appendToken(_ id: Int) {
        guard id >= 0, id < Self.vocabulary.count else { return }
        var piece = Self.vocabulary[id]
        if text.isEmpty {
            while piece.hasPrefix(" ") { piece.removeFirst() }
        }
        text.append(contentsOf: piece)
    }

    private func pruneBuffer() {
        let keepFrom = Self.alignDown(max(0, nextMelToEncode - Self.contextFrames))
        guard keepFrom > melStart else { return }
        let drop = min(keepFrom - melStart, melFrames.count)
        if drop > 0 {
            melFrames.removeFirst(drop)
            melStart += drop
        }
    }

    private static func alignDown(_ value: Int) -> Int {
        value - value % 4
    }

    private static func alignUp(_ value: Int, _ multiple: Int) -> Int {
        let rem = value % multiple
        return rem == 0 ? value : value + (multiple - rem)
    }

    // MARK: - Decoder

    private struct DecoderOutput {
        let out: ORTValue
        let h: ORTValue
        let c: ORTValue
    }

    private static func runDecoder(_ session: ORTSession, label: Int,
                                   h: ORTValue, c: ORTValue) throws -> DecoderOutput {
        let x = try int64Tensor([Int64(label)], shape: [1, 1])
        let outputs = try session.run(
            withInputs: ["x": x, "hi": h, "ci": c],
            outputNames: ["dec", "ho", "co"],
            runOptions: nil as ORTRunOptions?
        )
        guard let out = outputs["dec"], let newH = outputs["ho"], let newC = outputs["co"] else {
            throw NSError(domain: "GigaamRNNT", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "decoder outputs missing"])
        }
        return DecoderOutput(out: out, h: newH, c: newC)
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

    private static func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buffer in
            NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Int64>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .int64,
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
