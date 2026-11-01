import AVFoundation
import Foundation

/// Shared audio-session setup.
///
/// `.playAndRecord` + `.voiceChat` is what turns on the system echo canceller:
/// the hardware subtracts everything the app is playing from the microphone
/// signal, so the TTS voice does not get transcribed back into the subtitles.
/// The same session serves both the microphone (`AudioInput`) and the player.
enum AudioSessionConfig {
    static func activate() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord,
                                mode: .voiceChat,
                                options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true, options: [])
    }

    /// Fallback used only when the voice-processing input unit refuses to start.
    static func activateForPlayback() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [.duckOthers])
        try session.setActive(true, options: [])
    }
}

/// Plays synthesised mono speech. Buffers are queued on the player node of the
/// shared engine, so consecutive words follow each other without gaps.
final class SpeechPlayer {
    /// Called on the main thread when the queued audio has finished playing.
    var onIdle: (() -> Void)? {
        didSet {
            AudioEngineHub.shared.onIdle = { [weak self] in self?.onIdle?() }
        }
    }

    func play(_ samples: [Float], sampleRate: Double) {
        AudioEngineHub.shared.play(samples, sampleRate: sampleRate)
    }

    /// Drops everything that is still queued and stops the output.
    func stop() {
        AudioEngineHub.shared.stopPlayback()
    }
}
