import Foundation

/// Owns the recognizer, the microphone and the UI-facing state.
///
/// The ONNX sessions live on a private serial queue: audio chunks are pushed from
/// the microphone callback and the model runs there, so nothing blocks the UI and
/// the sessions are only ever touched from one thread.
///
/// Only one language is kept in memory at a time: switching languages unloads the
/// previous recognizer (its ONNX sessions are released on the inference queue)
/// and loads the model for the new one.
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
    private var recognizer: StreamingRecognizer?

    /// Incremented on every model load request; read on the main thread.
    private var loadGeneration = 0
    /// Mirrors `loadGeneration` on the inference queue, so a slow model load that
    /// was superseded by a newer language selection can be discarded on arrival.
    private var activeGeneration = 0

    // MARK: - Model

    func load() {
        guard case .idle = state else { return }
        loadModel(for: language)
    }

    private func loadModel(for language: AppLanguage) {
        loadGeneration += 1
        let token = loadGeneration

        guard language.hasModel else {
            state = .ready
            queue.async { [weak self] in
                self?.unloadRecognizer()
            }
            return
        }

        state = .loading
        queue.async { [weak self] in
            guard let self else { return }
            self.activeGeneration = token
            self.unloadRecognizer()

            do {
                // The big ONNX sessions are Objective-C objects; the pool makes
                // sure the old graphs are really gone before the new one loads.
                let recognizer = try autoreleasepool { try RecognizerFactory.make(for: language) }
                guard self.activeGeneration == token else { return }

                recognizer.onText = { [weak self] text in
                    DispatchQueue.main.async {
                        guard let self, self.loadGeneration == token else { return }
                        self.transcript = text
                    }
                }
                self.recognizer = recognizer

                DispatchQueue.main.async {
                    guard self.loadGeneration == token else { return }
                    self.state = .ready
                }
            } catch {
                DispatchQueue.main.async {
                    guard self.loadGeneration == token else { return }
                    self.state = .failed(error.localizedDescription)
                }
            }
        }
    }

    /// Drops the current model; the ONNX Runtime sessions free their weights.
    private func unloadRecognizer() {
        recognizer = nil
    }

    // MARK: - Language

    func select(_ language: AppLanguage) {
        guard language != self.language else { return }
        if isListening { stopListening() }

        self.language = language
        transcript = ""
        loadModel(for: language)
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
        queue.async { self.recognizer?.flush(tailSeconds: 0.5) }
    }
}

/// Builds the recognizer that matches the language's exported model family.
enum RecognizerFactory {
    static func make(for language: AppLanguage) throws -> StreamingRecognizer {
        let files = try ModelLocator.files(for: language)
        switch files.family {
        case .icefall:
            guard let tokens = files.tokens else {
                throw NSError(domain: "RecognizerFactory", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Нет файла tokens.txt"])
            }
            return try StreamingRNNT(encoderURL: files.encoder,
                                     decoderURL: files.decoder,
                                     joinerURL: files.joiner,
                                     tokensURL: tokens)
        case .gigaam:
            return try GigaamRNNT(encoderURL: files.encoder,
                                  decoderURL: files.decoder,
                                  jointURL: files.joiner)
        }
    }
}

/// Finds the ONNX graphs and the token table inside the app bundle.
enum ModelLocator {
    enum Family {
        case icefall
        case gigaam
    }

    struct Files {
        let family: Family
        let encoder: URL
        let decoder: URL
        let joiner: URL
        let tokens: URL?
    }

    static func files(for language: AppLanguage) throws -> Files {
        guard language.hasModel else {
            throw NSError(domain: "ModelLocator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Нет модели для этого языка"])
        }

        switch language {
        case .english:
            return Files(family: .icefall,
                         encoder: try find("encoder", "onnx", for: language),
                         decoder: try find("decoder", "onnx", for: language),
                         joiner: try find("joiner", "onnx", for: language),
                         tokens: try find("tokens", "txt", for: language))
        case .russian:
            return Files(family: .gigaam,
                         encoder: try find("v3_rnnt_encoder", "onnx", for: language),
                         decoder: try find("v3_rnnt_decoder", "onnx", for: language),
                         joiner: try find("v3_rnnt_joint", "onnx", for: language),
                         tokens: nil)
        case .french, .spanish:
            throw NSError(domain: "ModelLocator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Нет модели для этого языка"])
        }
    }

    private static func find(_ name: String, _ ext: String, for language: AppLanguage) throws -> URL {
        let bundle = Bundle.main
        let folder = language.modelFolder ?? language.rawValue
        let candidates: [URL?] = [
            bundle.url(forResource: name, withExtension: ext),
            bundle.url(forResource: name, withExtension: ext, subdirectory: "Models"),
            bundle.url(forResource: name, withExtension: ext, subdirectory: "Models/\(folder)"),
            bundle.url(forResource: name, withExtension: ext, subdirectory: folder)
        ]
        for case let url? in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        throw NSError(domain: "ModelLocator", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "В бандле не найден файл \(name).\(ext)"
        ])
    }
}
