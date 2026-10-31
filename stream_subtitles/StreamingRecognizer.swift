import Foundation

/// Common surface of every streaming recognizer the app can load.
///
/// The engine only talks to this protocol, so English (icefall RNN-T) and
/// Russian (GigaAM v3 RNN-T) share exactly the same microphone / UI logic.
protocol StreamingRecognizer: AnyObject {
    /// Called on the recognizer's own queue whenever the text changes.
    var onText: ((String) -> Void)? { get set }
    /// The full transcript produced since the last reset.
    var text: String { get }

    /// Feeds 16 kHz mono float samples.
    func accept(_ samples: [Float])
    /// Pushes trailing silence so the tail of the stream is decoded.
    func flush(tailSeconds: Double)
    /// Drops all audio and decoder state.
    func reset()
}
