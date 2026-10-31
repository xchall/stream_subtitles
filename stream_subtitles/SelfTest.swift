#if DEBUG
import Foundation

/// Tiny debug-only harness: launch the app with `--selftest <path-to-16k-wav>` and it
/// decodes the file with the very same streaming loop the microphone uses.
///
///     xcrun simctl launch <device> com.dobriy.streamsubtitles --selftest /path/in/container.wav
///
/// The transcript is written to `Documents/selftest.txt` and printed to the console.
enum SelfTest {
    static var isRequested: Bool {
        ProcessInfo.processInfo.arguments.contains("--selftest")
    }

    static func runIfRequested() async {
        guard isRequested else { return }

        let args = ProcessInfo.processInfo.arguments
        let index = args.firstIndex(of: "--selftest")!
        var path = index + 1 < args.count ? args[index + 1] : ""
        if path.isEmpty || path.hasPrefix("--") {
            path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("selftest.wav").path
        }

        await Task.detached(priority: .userInitiated) {
            do {
                let started = Date()
                let urls = try ModelLocator.urls(for: .english)
                let recognizer = try StreamingRNNT(encoderURL: urls.encoder,
                                                   decoderURL: urls.decoder,
                                                   joinerURL: urls.joiner,
                                                   tokensURL: urls.tokens)
                NSLog("SELFTEST model loaded in \(Date().timeIntervalSince(started))s")

                let samples = try WavFile.load(path: URL(fileURLWithPath: path))
                let block = 320  // 20 ms
                var offset = 0
                while offset < samples.count {
                    let end = min(offset + block, samples.count)
                    recognizer.accept(Array(samples[offset..<end]))
                    offset = end
                }
                recognizer.flush()

                let text = recognizer.text
                let line = "SELFTEST RESULT: \(text)\n"
                NSLog("%@", line)
                let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("selftest.txt")
                try? line.write(to: out, atomically: true, encoding: .utf8)
                NSLog("SELFTEST done in \(Date().timeIntervalSince(started))s")
            } catch {
                NSLog("SELFTEST FAILED: \(error)")
            }
        }.value
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
