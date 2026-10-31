import Foundation

/// Owns the recognizer, the microphone and the UI-facing state.
///
/// The ONNX sessions live on a private serial queue: audio chunks are pushed from
/// the microphone callback and the model runs there, so nothing blocks the UI and
/// the sessions are only ever touched from one thread.
final class StreamingEngine: ObservableObject {
    enum State: Equatable {
        case idle
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var transcript: String = ""
    @Published private(set) var isListening: Bool = false
    @Published private(set) var language: AppLanguage = .english
    @Published private(set) var permissionDenied: Bool = false

    private let queue = DispatchQueue(label: "com.dobriy.streamsubtitles.inference", qos: .userInitiated)
    private let audio = AudioInput()
    private var recognizer: StreamingRNNT?
    private var loaded = false

    // MARK: - Model

    func load() {
        guard case .idle = state else { return }
        state = .loading

        queue.async { [weak self] in
            guard let self else { return }
            do {
                let urls = try ModelLocator.urls(for: .english)
                let recognizer = try StreamingRNNT(encoderURL: urls.encoder,
                                                   decoderURL: urls.decoder,
                                                   joinerURL: urls.joiner,
                                                   tokensURL: urls.tokens)
                recognizer.onText = { [weak self] text in
                    DispatchQueue.main.async { self?.transcript = text }
                }
                self.recognizer = recognizer
                self.loaded = true
                DispatchQueue.main.async { self.state = .ready }
            } catch {
                DispatchQueue.main.async { self.state = .failed(error.localizedDescription) }
            }
        }
    }

    // MARK: - Language

    func select(_ language: AppLanguage) {
        guard language != self.language else { return }
        if isListening { stopListening() }
        self.language = language
    }

    // MARK: - Microphone

    var canListen: Bool {
        state == .ready && !permissionDenied && language.hasModel
    }

    func toggleListening() {
        if isListening {
            stopListening()
        } else {
            startListening()
        }
    }

    private func startListening() {
        guard state == .ready, language.hasModel else { return }

        AudioInput.requestPermission { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.permissionDenied = true
                return
            }
            do {
                self.audio.onSamples = { [weak self] samples in
                    guard let self else { return }
                    self.queue.async { self.recognizer?.accept(samples) }
                }
                try self.audio.start()
                self.isListening = true
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    private func stopListening() {
        audio.stop()
        audio.onSamples = nil
        isListening = false
        // Flush the last ~0.5 s of audio that is still sitting in the window buffer.
        queue.async { self.recognizer?.flush() }
    }
}

/// Finds the ONNX graphs and the token table inside the app bundle.
enum ModelLocator {
    struct URLs {
        let encoder: URL
        let decoder: URL
        let joiner: URL
        let tokens: URL
    }

    static func urls(for language: AppLanguage) throws -> URLs {
        guard language.hasModel else {
            throw NSError(domain: "ModelLocator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Нет модели для этого языка"])
        }
        return URLs(encoder: try find("encoder", "onnx"),
                    decoder: try find("decoder", "onnx"),
                    joiner: try find("joiner", "onnx"),
                    tokens: try find("tokens", "txt"))
    }

    private static func find(_ name: String, _ ext: String) throws -> URL {
        let bundle = Bundle.main
        let candidates: [URL?] = [
            bundle.url(forResource: name, withExtension: ext),
            bundle.url(forResource: name, withExtension: ext, subdirectory: "Models"),
            bundle.url(forResource: name, withExtension: ext, subdirectory: "Models/en"),
            bundle.url(forResource: name, withExtension: ext, subdirectory: "en")
        ]
        for case let url? in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        throw NSError(domain: "ModelLocator", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "В бандле не найден файл \(name).\(ext)"
        ])
    }
}
