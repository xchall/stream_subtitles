import AVFoundation
import Foundation

/// One `AVAudioEngine` shared by the microphone and the speech player.
///
/// The session runs in `.playAndRecord` + `.voiceChat`, which is the mode the
/// system voice-processing I/O unit needs for hardware echo cancellation: what
/// the app plays is subtracted from the microphone signal. That unit is a
/// single, device-wide resource — running a second engine for playback made the
/// output fail with `auou/vpio/appl render err: -1` and produced chopped,
/// sped-up syllables. So both directions live in the same graph.
final class AudioEngineHub {
    static let shared = AudioEngineHub()

    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()

    /// Called on the main thread when the last scheduled buffer has been played.
    var onIdle: (() -> Void)?

    private var connectedSampleRate: Double?
    private var pendingBuffers = 0
    private let stateLock = NSLock()

    private init() {
        engine.attach(player)
        // Both Piper voices are 22.05 kHz mono. Connect once, up front: calling
        // `play()` on a node whose output was just changed made AVFoundation
        // throw "player started when in a disconnected state".
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 22050,
                                   channels: 1,
                                   interleaved: false)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        connectedSampleRate = 22050
    }

    // MARK: - Output

    func play(_ samples: [Float], sampleRate: Double) {
        guard !samples.isEmpty, sampleRate > 0 else { return }
        let rate = connectedSampleRate ?? 22050
        let pcm = sampleRate == rate ? samples : Self.resample(samples, from: sampleRate, to: rate)

        if !engine.isRunning {
            do {
                try startEngine()
            } catch {
                NSLog("AudioEngineHub start failed: \(error)")
                return
            }
        }
        if !player.isPlaying { player.play() }

        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: rate,
                                         channels: 1,
                                         interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(pcm.count)),
              let channel = buffer.floatChannelData?[0] else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(pcm.count)
        pcm.withUnsafeBufferPointer { source in
            channel.update(from: source.baseAddress!, count: pcm.count)
        }
        stateLock.lock()
        pendingBuffers += 1
        stateLock.unlock()
        player.scheduleBuffer(buffer, at: nil, options: []) { [weak self] in
            DispatchQueue.main.async { self?.bufferPlayed() }
        }
    }

    func stopPlayback() {
        player.stop()
        stateLock.lock()
        pendingBuffers = 0
        stateLock.unlock()
        onIdle?()
    }

    private func bufferPlayed() {
        stateLock.lock()
        pendingBuffers = max(0, pendingBuffers - 1)
        let idle = pendingBuffers == 0
        stateLock.unlock()
        if idle { onIdle?() }
    }

    /// Starts the shared graph, making sure the audio session is active first.
    ///
    /// Without this the engine used to be started while the session was still
    /// in its default category (or `.record`), and Core Audio rejected it with
    /// `kAudioUnitErr_InvalidPropertyValue` (-10851).
    private func startEngine() throws {
        do {
            try AudioSessionConfig.activate()
        } catch {
            NSLog("AudioSession activate failed: \(error)")
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            // No usable input (mic permission denied, simulator quirk): the
            // `.voiceChat` I/O unit still wants one, so fall back to pure
            // playback for the voice.
            try? AudioSessionConfig.activateForPlayback()
            engine.prepare()
            try engine.start()
        }
    }

    private static func resample(_ samples: [Float], from source: Double, to target: Double) -> [Float] {
        let ratio = source / target
        let count = Int(Double(samples.count) / ratio)
        guard count > 0 else { return [] }
        var result = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let position = Double(index) * ratio
            let low = min(Int(position), samples.count - 1)
            let high = min(low + 1, samples.count - 1)
            let fraction = Float(position - Double(low))
            result[index] = samples[low] * (1 - fraction) + samples[high] * fraction
        }
        return result
    }

    // MARK: - Input

    func startInputIfNeeded() throws {
        guard !engine.isRunning else { return }
        engine.prepare()
        try engine.start()
    }
}
