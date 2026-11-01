import AVFoundation
import Foundation

/// Captures the microphone and delivers 16 kHz mono float32 samples.
///
/// It uses the shared `AudioEngineHub` instead of its own `AVAudioEngine`: iOS
/// has one system voice-processing I/O unit, and a second engine made playback
/// fail with `kAudioUnitErr_InvalidPropertyValue` (-10851) while the microphone
/// was running. One engine, one `.playAndRecord` session, both directions.
final class AudioInput {
    private let hub = AudioEngineHub.shared
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                             sampleRate: 16000,
                                             channels: 1,
                                             interleaved: false)!
    private var converter: AVAudioConverter?
    private var running = false
    private(set) var level: Float = 0

    var onSamples: (([Float]) -> Void)?

    // MARK: - Permission

    static func requestPermission(completion: @escaping (Bool) -> Void) {
        AVAudioApplication.requestRecordPermission { granted in
            DispatchQueue.main.async { completion(granted) }
        }
    }

    // MARK: - Start / stop

    func start() throws {
        guard !running else { return }

        // `.record` would forbid playback on the same session, so the shared
        // configuration (`.playAndRecord` + `.voiceChat`) is used for both.
        try AudioSessionConfig.activate()

        let input = hub.engine.inputNode
        let hardwareFormat = input.inputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 else {
            throw NSError(domain: "AudioInput", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Микрофон недоступен"])
        }
        guard let converter = AVAudioConverter(from: hardwareFormat, to: targetFormat) else {
            throw NSError(domain: "AudioInput", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Не удалось создать конвертер"])
        }
        self.converter = converter

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: hardwareFormat) { [weak self] buffer, _ in
            autoreleasepool {
                self?.handle(buffer: buffer)
            }
        }

        try hub.startInputIfNeeded()
        running = true
    }

    func stop() {
        guard running else { return }
        running = false
        hub.engine.inputNode.removeTap(onBus: 0)
        converter = nil
        // The shared engine is left running: the voice may still be speaking
        // the tail of the transcript.
    }

    // MARK: - Conversion

    private func handle(buffer: AVAudioPCMBuffer) {
        guard let converter else { return }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return
        }

        var error: NSError?
        var supplied = false
        converter.convert(to: converted, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }

        if let error {
            NSLog("AudioInput conversion error: \(error)")
            return
        }
        guard converted.frameLength > 0, let channel = converted.floatChannelData?[0] else { return }

        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
        level = Self.rms(samples)
        onSamples?(samples)
    }

    private static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return sqrt(sum / Float(samples.count))
    }
}
