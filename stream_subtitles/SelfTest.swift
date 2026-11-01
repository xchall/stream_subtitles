#if DEBUG
import Foundation

/// Tiny debug-only harness: launch the app with `--selftest <path-to-16k-wav>`
/// and it decodes the file with the very same streaming loop the microphone uses.
///
///     xcrun simctl launch <device> com.dobriy.streamsubtitles --selftest /path/in/container.wav
///     xcrun simctl launch <device> com.dobriy.streamsubtitles --selftest /path.wav --lang ru
///
/// The transcript is written to `Documents/selftest.txt` and printed to the console.
enum SelfTest {
    static var isRequested: Bool {
        ProcessInfo.processInfo.arguments.contains("--selftest")
    }

    /// `--tts "text"` synthesises one phrase, `--tts-verify` transcribes it back
    /// with the ASR model, `--tts-stream` feeds a live transcript word by word.
    static var isTTSRequested: Bool {
        let args = ProcessInfo.processInfo.arguments
        return args.contains("--tts") || args.contains("--tts-stream")
    }

    static func runIfRequested() async {
        if isRequested { await runRecognizerTest() }
        if isTTSRequested { await runTTS() }
    }

    private static func runRecognizerTest() async {
        guard isRequested else { return }

        let args = ProcessInfo.processInfo.arguments
        let index = args.firstIndex(of: "--selftest")!
        var path = index + 1 < args.count ? args[index + 1] : ""
        if path.isEmpty || path.hasPrefix("--") {
            path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("selftest.wav").path
        }

        var language = AppLanguage.english
        if let langIndex = args.firstIndex(of: "--lang"), langIndex + 1 < args.count,
           let selected = AppLanguage(rawValue: args[langIndex + 1]) {
            language = selected
        }

        await Task.detached(priority: .userInitiated) {
            do {
                let started = Date()
                let recognizer = try RecognizerFactory.make(for: language)
                NSLog("SELFTEST [%@] model loaded in %.2fs",
                      language.rawValue, Date().timeIntervalSince(started))

                let samples = try WavFile.load(path: URL(fileURLWithPath: path))
                let audioSeconds = Double(samples.count) / 16000.0

                let inferenceStarted = Date()
                let block = 320  // 20 ms
                var offset = 0
                while offset < samples.count {
                    let end = min(offset + block, samples.count)
                    recognizer.accept(Array(samples[offset..<end]))
                    offset = end
                }
                recognizer.flush(tailSeconds: 0.5)
                let elapsed = Date().timeIntervalSince(inferenceStarted)

                let text = recognizer.text
                let stats = String(format: "audio %.2fs, inference %.2fs, RTF %.2f",
                                   audioSeconds, elapsed, elapsed / max(audioSeconds, 0.001))
                let line = "SELFTEST RESULT [\(language.rawValue)]: \(text)\nSELFTEST STATS: \(stats)\n"
                NSLog("%@", line)
                let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("selftest.txt")
                try? line.write(to: out, atomically: true, encoding: .utf8)
                NSLog("SELFTEST done in %.2fs", Date().timeIntervalSince(started))
            } catch {
                NSLog("SELFTEST FAILED: \(error)")
            }
        }.value
    }

    // MARK: - TTS harness

    private static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private static func runTTS() async {
        let args = ProcessInfo.processInfo.arguments

        var language = AppLanguage.english
        if let index = args.firstIndex(of: "--lang"), index + 1 < args.count,
           let selected = AppLanguage(rawValue: args[index + 1]) {
            language = selected
        }

        var text = "hello world one two three four five"
        let textFlag = args.contains("--tts-stream") ? "--tts-stream" : "--tts"
        if let index = args.firstIndex(of: textFlag), index + 1 < args.count,
           !args[index + 1].hasPrefix("--") {
            text = args[index + 1]
        }

        let streamed = args.contains("--tts-stream")
        let verify = args.contains("--tts-verify")

        await Task.detached(priority: .userInitiated) {
            do {
                if streamed {
                    try await streamedTTS(text: text, language: language)
                    return
                }

                let started = Date()
                let tts = try PiperTTS.make(for: language)
                let samples = try tts.synthesize(text: text)
                let elapsed = Date().timeIntervalSince(started)
                let seconds = Double(samples.count) / tts.sampleRate

                NSLog("TTSRESULT [%@] %@ → %.2fs audio in %.2fs (RTF %.2f)",
                      language.rawValue, text, seconds, elapsed,
                      elapsed / max(seconds, 0.001))

                let url = documents.appendingPathComponent("tts-\(language.rawValue).wav")
                try WavFile.write(samples, sampleRate: tts.sampleRate, to: url)
                NSLog("TTSWAV %@", url.path)

                if verify {
                    try await verifyTTS(samples, sampleRate: tts.sampleRate, language: language)
                }
            } catch {
                NSLog("TTS FAILED: \(error)")
            }
        }.value
    }

    /// Simulates the live transcript: grows word by word, exactly like the
    /// recogniser does, so the batching rule can be observed in the logs.
    private static func streamedTTS(text: String, language: AppLanguage) async throws {
        let speaker = StreamingSpeaker()
        speaker.prepare(for: language)
        try await Task.sleep(nanoseconds: 2_000_000_000)

        var transcript = ""
        for word in text.split(whereSeparator: { $0.isWhitespace }) {
            transcript += transcript.isEmpty ? String(word) : " " + word
            speaker.accept(transcript)
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        // Let the silence timer close the phrase, the way a real pause does;
        // `finish()` then only has to handle the (already empty) tail.
        try await Task.sleep(nanoseconds: 2_500_000_000)
        speaker.finish()
        try await Task.sleep(nanoseconds: 10_000_000_000)
        speaker.stop()
    }

    private static func verifyTTS(_ samples: [Float],
                                  sampleRate: Double,
                                  language: AppLanguage) async throws {
        let recognizer = try RecognizerFactory.make(for: language)
        let audio = resample(samples, from: sampleRate, to: 16000)

        let block = 320  // 20 ms
        var offset = 0
        while offset < audio.count {
            let end = min(offset + block, audio.count)
            recognizer.accept(Array(audio[offset..<end]))
            offset = end
        }
        recognizer.flush(tailSeconds: 0.5)
        NSLog("TTSVERIFY [%@]: %@", language.rawValue, recognizer.text)
    }

    private static func resample(_ samples: [Float], from source: Double, to target: Double) -> [Float] {
        guard source != target, !samples.isEmpty else { return samples }
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
}

/// Minimal RIFF/WAVE reader: 16-bit PCM or 32-bit float, mono, 16 kHz.
enum WavFile {
    static func load(path: URL) throws -> [Float] {
        let data = try Data(contentsOf: path)
        guard data.count > 44, data[0..<4] == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) else {
            throw NSError(domain: "WavFile", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "not a RIFF/WAVE file"])
        }

        var format = 1
        var channels = 1
        var sampleRate = 16000
        var bits = 16
        var payload: Data?

        var offset = 12
        while offset + 8 <= data.count {
            let id = data[offset..<offset + 4]
            let size = Int(data.readUInt32(at: offset + 4))
            let body = offset + 8
            guard body + size <= data.count else { break }

            if id == Data("fmt ".utf8) {
                format = Int(data.readUInt16(at: body))
                channels = Int(data.readUInt16(at: body + 2))
                sampleRate = Int(data.readUInt32(at: body + 4))
                bits = Int(data.readUInt16(at: body + 14))
            } else if id == Data("data".utf8) {
                payload = data.subdata(in: body..<(body + size))
            }
            offset = body + size + (size % 2)
        }

        guard let payload else {
            throw NSError(domain: "WavFile", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "no data chunk"])
        }
        guard sampleRate == 16000, channels == 1 else {
            throw NSError(domain: "WavFile", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "expected 16 kHz mono, got \(sampleRate) Hz / \(channels) ch"
            ])
        }

        if format == 3 && bits == 32 {
            return payload.withUnsafeBytes { raw in
                raw.bindMemory(to: Float.self).map { $0 }
            }
        }
        guard bits == 16 else {
            throw NSError(domain: "WavFile", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "unsupported bit depth \(bits)"])
        }
        return payload.withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { Float($0) / 32768.0 }
        }
    }

    /// Writes mono 16-bit PCM, which is what `afplay`/`afinfo` and the debug
    /// harness expect.
    static func write(_ samples: [Float], sampleRate: Double, to url: URL) throws {
        let rate = UInt32(sampleRate.rounded())
        let byteCount = samples.count * 2

        var data = Data(capacity: 44 + byteCount)
        func append(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: Data("RIFF".utf8))
        append(UInt32(36 + byteCount))
        data.append(contentsOf: Data("WAVE".utf8))
        data.append(contentsOf: Data("fmt ".utf8))
        append(16)               // fmt chunk size
        append16(1)              // PCM
        append16(1)              // mono
        append(rate)
        append(rate * 2)         // byte rate
        append16(2)              // block align
        append16(16)             // bits per sample
        data.append(contentsOf: Data("data".utf8))
        append(UInt32(byteCount))

        for sample in samples {
            let clamped = max(-1, min(1, sample))
            append16(UInt16(bitPattern: Int16(clamped * 32767)))
        }
        try data.write(to: url)
    }
}

private extension Data {
    func readUInt16(at index: Int) -> UInt16 {
        UInt16(self[index]) | (UInt16(self[index + 1]) << 8)
    }

    func readUInt32(at index: Int) -> UInt32 {
        UInt32(self[index]) | (UInt32(self[index + 1]) << 8)
        | (UInt32(self[index + 2]) << 16) | (UInt32(self[index + 3]) << 24)
    }
}
#endif
