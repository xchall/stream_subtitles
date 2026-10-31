import Foundation

/// GigaAM's log-mel features.
///
/// GigaAM v3 was trained with `gigaam.preprocess.FeatureExtractor`, which is
/// `torchaudio.transforms.MelSpectrogram(...)` followed by `log(clamp(x, 1e-9, 1e9))`:
///
///     sample_rate  16000      n_fft         320
///     win_length   320 (20 ms) hop_length   160 (10 ms)
///     n_mels       64         center        false   (no padding at all)
///     window       periodic Hann           mel_scale   htk
///     norm         none       power         2.0     (|STFT|^2)
///
/// Note that this is *not* the Kaldi fbank used by the English model: 64 bins,
/// no preemphasis, no DC removal, no dither, HTK mel scale, and `n_fft = 320`
/// (which is not a power of two, so the spectrum is evaluated directly).
final class GigaamFbankComputer {
    static let sampleRate = 16000
    static let winLength = 320
    static let hopLength = 160
    static let fftSize = 320
    static let numMelBins = 64
    static let numFftBins = fftSize / 2 + 1          // 161
    static let logFloor = 1e-9
    static let logCeil = 1e9

    private let window: [Double]
    private let melBank: [[Double]]                   // [64][161]
    private let cosTable: [Double]                    // [161 * 320]
    private let sinTable: [Double]                    // [161 * 320]

    init() {
        var w = [Double](repeating: 0, count: Self.fftSize)
        for i in 0..<Self.winLength {
            // torch.hann_window(periodic=True)
            w[i] = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(Self.winLength))
        }
        window = w
        melBank = Self.makeMelBank()

        var cosTable = [Double](repeating: 0, count: Self.numFftBins * Self.fftSize)
        var sinTable = [Double](repeating: 0, count: Self.numFftBins * Self.fftSize)
        for k in 0..<Self.numFftBins {
            let base = k * Self.fftSize
            for n in 0..<Self.fftSize {
                let angle = -2.0 * Double.pi * Double(k) * Double(n) / Double(Self.fftSize)
                cosTable[base + n] = cos(angle)
                sinTable[base + n] = sin(angle)
            }
        }
        self.cosTable = cosTable
        self.sinTable = sinTable
    }

    /// One 64-bin log-mel frame from exactly `winLength` raw samples.
    func frame(_ samples: [Double]) -> [Float] {
        var power = [Double](repeating: 0, count: Self.numFftBins)
        for k in 0..<Self.numFftBins {
            let base = k * Self.fftSize
            var real = 0.0
            var imag = 0.0
            for n in 0..<Self.fftSize {
                let v = samples[n] * window[n]
                real += v * cosTable[base + n]
                imag += v * sinTable[base + n]
            }
            power[k] = real * real + imag * imag
        }

        var out = [Float](repeating: 0, count: Self.numMelBins)
        for b in 0..<Self.numMelBins {
            var acc = 0.0
            let row = melBank[b]
            for k in 0..<Self.numFftBins {
                acc += power[k] * row[k]
            }
            out[b] = Float(log(min(max(acc, Self.logFloor), Self.logCeil)))
        }
        return out
    }

    // MARK: - Mel filterbank (torchaudio, mel_scale = htk, norm = none)

    private static func makeMelBank() -> [[Double]] {
        // torch.linspace(0, sample_rate / 2, n_fft / 2 + 1)
        let allFreqs = (0..<numFftBins).map {
            Double(sampleRate) / 2.0 * Double($0) / Double(numFftBins - 1)
        }

        let melMin = hzToMel(0.0)
        let melMax = hzToMel(Double(sampleRate) / 2.0)
        let points = (0..<(numMelBins + 2)).map { i -> Double in
            let mel = melMin + (melMax - melMin) * Double(i) / Double(numMelBins + 1)
            return melToHz(mel)
        }

        var bank = [[Double]](repeating: [Double](repeating: 0, count: numFftBins),
                              count: numMelBins)
        for b in 0..<numMelBins {
            let left = points[b]
            let center = points[b + 1]
            let right = points[b + 2]
            for k in 0..<numFftBins {
                let f = allFreqs[k]
                let up = (f - left) / (center - left)
                let down = (right - f) / (right - center)
                bank[b][k] = max(0.0, min(up, down))
            }
        }
        return bank
    }

    private static func hzToMel(_ hz: Double) -> Double {
        2595.0 * log10(1.0 + hz / 700.0)
    }

    private static func melToHz(_ mel: Double) -> Double {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
}

/// Turns a continuous 16 kHz stream into GigaAM mel frames.
///
/// With `center = false` frame `n` covers exactly `[n * 160, n * 160 + 320)`,
/// so there is no reflection padding and nothing is emitted until 20 ms of
/// audio are available.
final class GigaamFbankStream {
    private let computer = GigaamFbankComputer()
    private var samples: [Float] = []
    private var base = 0          // absolute index of samples[0]
    private var nextFrame = 0

    func reset() {
        samples.removeAll(keepingCapacity: true)
        base = 0
        nextFrame = 0
    }

    /// Appends audio and returns every frame that is now complete.
    func push(_ newSamples: [Float]) -> [[Float]] {
        samples.append(contentsOf: newSamples)

        var frames: [[Float]] = []
        while nextFrame * GigaamFbankComputer.hopLength + GigaamFbankComputer.winLength
                <= base + samples.count {
            let start = nextFrame * GigaamFbankComputer.hopLength - base
            var window = [Double](repeating: 0, count: GigaamFbankComputer.winLength)
            for i in 0..<GigaamFbankComputer.winLength {
                window[i] = Double(samples[start + i])
            }
            frames.append(computer.frame(window))
            nextFrame += 1
        }

        // Frames are left-aligned, so nothing before the next frame start is needed.
        let keepFrom = nextFrame * GigaamFbankComputer.hopLength
        if keepFrom > base {
            let drop = keepFrom - base
            if drop >= samples.count {
                samples.removeAll(keepingCapacity: true)
            } else {
                samples.removeFirst(drop)
            }
            base = keepFrom
        }
        return frames
    }
}
