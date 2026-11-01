import Foundation

/// Turns the live transcript into spoken phrases.
///
/// The streaming ASR is never blocked by the voice. This class uses two
/// independent serial queues:
///
///  * `accumulatorQueue` receives the growing transcript, finds word boundaries,
///    buffers words and runs the phrase timers. Its blocks are tiny (string
///    slicing), so the recogniser always gets its next audio chunk on time.
///  * `synthesisQueue` owns the Piper voice and runs the expensive ONNX
///    inference, one phrase at a time. It is `.utility` while ASR inference
///    runs at `.userInitiated`, so under CPU pressure iOS schedules the
///    recogniser first and the voice uses the remaining cores.
///
/// Phrase rule:
///  * the transcript is append-only, so a space followed by another character
///    finalises the word before that space;
///  * when no new text arrives for `pause`, everything accumulated is
///    synthesised as one phrase, including the last (still unconfirmed) word;
///  * if the speaker never pauses, a forced flush happens after
///    `maxPhraseDuration`. It only speaks words with a confirmed boundary: the
///    word that is still being recognised stays in the buffer, so it is never
///    cut in half.
///
/// Synthesis jobs are FIFO, so phrases are spoken in transcript order even when
/// the voice is slower than the recogniser.
final class StreamingSpeaker {
    private enum Timing {
        /// Silence (no new transcript text) that closes the current phrase.
        static let pause: TimeInterval = 1.2
        /// Hard limit for one batch while the speaker keeps talking.
        static let maxPhraseDuration: TimeInterval = 10.0
    }

    /// Cheap text handling and timers. Never runs ONNX.
    private let accumulatorQueue = DispatchQueue(
        label: "com.dobriy.streamsubtitles.tts.accumulator",
        qos: .userInitiated
    )

    /// Expensive Piper inference. Lower QoS than the ASR queue on purpose.
    private let synthesisQueue = DispatchQueue(
        label: "com.dobriy.streamsubtitles.tts.synthesis",
        qos: .utility
    )

    private let player = SpeechPlayer()

    /// Piper voice; touched only on `synthesisQueue`.
    private var tts: PiperTTS?
    private var language: AppLanguage?

    /// Shared flags, guarded by `stateLock`.
    private let stateLock = NSLock()
    private var enabled = true
    /// Bumped whenever queued speech must be discarded (mute, stop, voice
    /// change). Every queued job carries the token it was created with.
    private var generation = 0

    /// Transcript cursor and phrase buffer; touched only on `accumulatorQueue`.
    private var lastText = ""
    private var committedUpTo = 0
    private var pendingWords: [String] = []
    private var phraseStartedAt: Date?
    private var pauseWorkItem: DispatchWorkItem?
    private var capWorkItem: DispatchWorkItem?

    /// Called on the main thread when the speaking state changes.
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Called on the main thread with a human-readable load/synthesis error.
    var onError: ((String) -> Void)?

    init() {
        // The player reports when the queued audio has actually finished, so the
        // toolbar icon stops animating at the right moment.
        player.onIdle = { [weak self] in
            self?.onSpeakingChanged?(false)
        }
    }

    // MARK: - Voice lifecycle

    /// Loads (or reloads) the voice that matches the selected language.
    func prepare(for language: AppLanguage) {
        synthesisQueue.async { [weak self] in
            guard let self else { return }
            guard self.language != language || self.tts == nil else { return }

            let token = self.bumpGeneration()
            self.accumulatorQueue.async { [weak self] in
                self?.resetToEnd()
            }
            self.tts = nil
            self.language = language
            self.stopPlayback()

            guard language.hasModel else { return }
            do {
                let started = Date()
                let tts = try PiperTTS.make(for: language)
                guard self.isCurrent(token) else { return }
                self.tts = tts
                NSLog("Piper voice %@ loaded in %.2fs",
                      String(describing: language), Date().timeIntervalSince(started))
            } catch {
                guard self.isCurrent(token) else { return }
                NSLog("Piper voice load failed: \(error)")
                DispatchQueue.main.async { self.onError?(error.localizedDescription) }
            }
        }
    }

    func setEnabled(_ value: Bool) {
        stateLock.lock()
        let changed = enabled != value
        enabled = value
        if changed { generation += 1 }
        stateLock.unlock()
        guard changed else { return }

        // Keep the transcript cursor where it is: after unmuting we speak only
        // new text instead of replaying everything that was transcribed while
        // the voice was off.
        accumulatorQueue.async { [weak self] in
            self?.resetToEnd()
        }
        if !value { stopPlayback() }
    }

    /// Cancels queued speech immediately (stop, language change, toggle off).
    func stop() {
        bumpGeneration()
        accumulatorQueue.async { [weak self] in
            self?.resetToEnd()
        }
        synthesisQueue.async { [weak self] in
            self?.tts = nil
            self?.language = nil
        }
        stopPlayback()
    }

    // MARK: - Transcript in

    /// Feeds the whole (monotonically growing) transcript.
    func accept(_ text: String) {
        accumulatorQueue.async { [weak self] in
            self?.ingest(text)
        }
    }

    /// Speaks whatever is buffered right now (microphone stopped, or the app is
    /// about to switch language).
    func finish() {
        accumulatorQueue.async { [weak self] in
            self?.flushPhrase(includeTail: true, reason: "finish")
        }
    }

    // MARK: - Accumulation

    private func ingest(_ text: String) {
        guard isEnabled else {
            // Stay in sync while muted so unmuting does not read old text.
            lastText = text
            committedUpTo = Self.trimmedEnd(in: text)
            pendingWords.removeAll()
            cancelTimers()
            phraseStartedAt = nil
            return
        }

        guard text != lastText else { return }

        if !text.hasPrefix(lastText) {
            // The recogniser restarted or replaced its hypothesis.
            lastText = ""
            committedUpTo = 0
            pendingWords.removeAll()
            phraseStartedAt = nil
        }
        lastText = text

        // A word becomes final when a space is followed by another character.
        let boundary = Self.commitEnd(in: text)
        if boundary > committedUpTo {
            let start = text.index(text.startIndex, offsetBy: committedUpTo)
            let end = text.index(text.startIndex, offsetBy: boundary)
            pendingWords.append(contentsOf: Self.words(in: String(text[start..<end])))
            committedUpTo = boundary
        }

        let tailEnd = Self.trimmedEnd(in: text)
        guard !pendingWords.isEmpty || tailEnd > committedUpTo else {
            cancelTimers()
            phraseStartedAt = nil
            return
        }

        if phraseStartedAt == nil {
            phraseStartedAt = Date()
            scheduleCap(token: currentGeneration)
        }
        schedulePause(token: currentGeneration)
    }

    /// Last character offset at which a word is confirmed: a whitespace with at
    /// least one non-whitespace character somewhere after it.
    private static func commitEnd(in text: String) -> Int {
        var index = text.endIndex
        while index > text.startIndex {
            let previous = text.index(before: index)
            if text[previous].isWhitespace {
                let next = text.index(after: previous)
                if next < text.endIndex,
                   text[next...].contains(where: { !$0.isWhitespace }) {
                    return text.distance(from: text.startIndex, to: previous)
                }
            }
            index = previous
        }
        return 0
    }

    /// Character offset after dropping trailing whitespace.
    private static func trimmedEnd(in text: String) -> Int {
        var end = text.endIndex
        while end > text.startIndex {
            let previous = text.index(before: end)
            guard text[previous].isWhitespace else { break }
            end = previous
        }
        return text.distance(from: text.startIndex, to: end)
    }

    private static func words(in chunk: String) -> [String] {
        chunk.split(whereSeparator: { $0.isWhitespace }).compactMap { word in
            let clean = SpeechText.sanitize(String(word))
            return clean.isEmpty ? nil : clean
        }
    }

    // MARK: - Timers

    /// Silence timer: closes the phrase when no new text arrives for `pause`.
    private func schedulePause(token: Int) {
        pauseWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self, self.isCurrent(token) else { return }
            self.pauseWorkItem = nil
            self.flushPhrase(includeTail: true, reason: "pause")
        }
        pauseWorkItem = item
        accumulatorQueue.asyncAfter(deadline: .now() + Timing.pause, execute: item)
    }

    /// Hard cap: flushes confirmed words even if the speaker never pauses.
    private func scheduleCap(token: Int) {
        capWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self, self.isCurrent(token) else { return }
            self.capWorkItem = nil
            self.flushPhrase(includeTail: false, reason: "10s cap")
        }
        capWorkItem = item
        accumulatorQueue.asyncAfter(deadline: .now() + Timing.maxPhraseDuration, execute: item)
    }

    private func cancelTimers() {
        pauseWorkItem?.cancel()
        pauseWorkItem = nil
        capWorkItem?.cancel()
        capWorkItem = nil
    }

    // MARK: - Phrase flush

    /// Sends the current buffer to the synthesis queue.
    ///
    /// - Parameter includeTail: `true` after silence / on stop, when the last
    ///   word is safe to speak. `false` for the 10 s cap, where the last word
    ///   may still be in the middle of recognition and must stay buffered.
    private func flushPhrase(includeTail: Bool, reason: String) {
        cancelTimers()
        guard isEnabled else {
            resetToEnd()
            return
        }

        var words = pendingWords
        pendingWords.removeAll()

        if includeTail {
            let end = Self.trimmedEnd(in: lastText)
            if end > committedUpTo {
                let start = lastText.index(lastText.startIndex, offsetBy: committedUpTo)
                let stop = lastText.index(lastText.startIndex, offsetBy: end)
                words.append(contentsOf: Self.words(in: String(lastText[start..<stop])))
                committedUpTo = end
            }
        }

        // Anything left is an unfinished word (only possible after a cap
        // flush). Keep the buffer alive for the next batch.
        if Self.trimmedEnd(in: lastText) > committedUpTo {
            phraseStartedAt = Date()
            scheduleCap(token: currentGeneration)
            schedulePause(token: currentGeneration)
        } else {
            phraseStartedAt = nil
        }

        guard !words.isEmpty else { return }

        let phrase = words.joined(separator: " ")
        let token = currentGeneration
        NSLog("TTS phrase (%@, %d words): «%@»", reason, words.count, phrase)
        enqueue(phrase, token: token)
    }

    // MARK: - Synthesis queue

    private func enqueue(_ phrase: String, token: Int) {
        synthesisQueue.async { [weak self] in
            self?.synthesize(phrase, token: token)
        }
    }

    /// Runs on `synthesisQueue`, one phrase at a time.
    private func synthesize(_ phrase: String, token: Int) {
        guard isCurrent(token), isEnabled, let tts else { return }

        DispatchQueue.main.async { [weak self] in
            self?.onSpeakingChanged?(true)
        }

        do {
            let started = Date()
            let samples = try tts.synthesize(text: phrase)
            guard isCurrent(token) else { return }

            let seconds = Double(samples.count) / tts.sampleRate
            NSLog("TTS «%@» → %.2fs audio in %.2fs", phrase, seconds,
                  Date().timeIntervalSince(started))
            guard !samples.isEmpty else {
                DispatchQueue.main.async { [weak self] in
                    self?.onSpeakingChanged?(false)
                }
                return
            }

            let padded = samples + [Float](repeating: 0, count: Int(tts.sampleRate * 0.02))
            DispatchQueue.main.async { [weak self] in
                guard let self, self.isCurrent(token) else { return }
                self.player.play(padded, sampleRate: tts.sampleRate)
            }
        } catch {
            NSLog("TTS failed for «%@»: %@", phrase, String(describing: error))
            DispatchQueue.main.async { [weak self] in
                self?.onSpeakingChanged?(false)
                self?.onError?(error.localizedDescription)
            }
        }
    }

    private func stopPlayback() {
        DispatchQueue.main.async { [weak self] in
            self?.player.stop()
            self?.onSpeakingChanged?(false)
        }
    }

    // MARK: - Buffer reset

    /// Drops the phrase in flight but marks everything already transcribed as
    /// consumed, so the next ingest starts from the current end of the text.
    private func resetToEnd() {
        cancelTimers()
        pendingWords.removeAll()
        committedUpTo = Self.trimmedEnd(in: lastText)
        phraseStartedAt = nil
    }

    // MARK: - Shared state

    private var isEnabled: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return enabled
    }

    private var currentGeneration: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return generation
    }

    @discardableResult
    private func bumpGeneration() -> Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        generation += 1
        return generation
    }

    private func isCurrent(_ token: Int) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return generation == token
    }
}
