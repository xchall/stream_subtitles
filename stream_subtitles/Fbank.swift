import Foundation

/// Kaldi-compatible 80-bin log-mel filterbank.
///
/// The ONNX encoder was trained on icefall / lhotse features, which are produced by
/// `torchaudio.compliance.kaldi.fbank` with the following settings:
///
///     sample_rate    16000      frame_length  25 ms (400 samples)
///     frame_shift    10 ms      n_fft         512 (rounded to power of two)
///     window         povey      preemphasis   0.97
///     low_freq       20 Hz      high_freq     -400 Hz (=> 7600 Hz)
///     num_mel_bins   80         use_power     true
///     snip_edges     false      dither        0
///     remove_dc_offset           true
///
/// Everything here mirrors that implementation exactly (including the per-frame
/// preemphasis that replicates the first sample and the reflection padding for
/// `snip_edges = false`), so the features fed to the model match the training data.
final class FbankComputer {
    static let sampleRate = 16000
    static let frameLength = 400
    static let frameShift = 160
    static let fftSize = 512
    static let numMelBins = 80
    static let numFftBins = fftSize / 2          // 256
    static let leftPadding = frameLength / 2 - frameShift / 2   // 120

    /// std::numeric_limits<float>::epsilon(), the floor used by Kaldi for log-fbank.
    static let logEpsilon: Float = 1.1920929e-07

    private let window: [Double]
    private let melBank: [[Double]]   // [80][256]

    init() {
        var w = [Double](repeating: 0, count: Self.frameLength)
        for i in 0..<Self.frameLength {
            let hann = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(Self.frameLength - 1))
            w[i] = pow(hann, 0.85)
        }
        window = w
        melBank = Self.makeMelBank()
    }

    // MARK: - Single frame

    /// Computes one 80-bin log-mel frame.
    /// - Parameter samples: a closure that returns the raw 16 kHz sample at an absolute index.
    ///   Negative indices are handled by the caller (reflection); this method only asks for
    ///   indices inside the frame.
    func frame(startSample: Int, sample: (Int) -> Double) -> [Float] {
        let mel = linearMel(startSample: startSample, sample: sample)
        var out = [Float](repeating: 0, count: Self.numMelBins)
        for b in 0..<Self.numMelBins {
            out[b] = Float(log(max(mel[b], Double(Self.logEpsilon))))
        }
        return out
    }

    /// Linear (pre-log) mel energies of one frame. Exposed for tests.
    func linearMel(startSample: Int, sample: (Int) -> Double) -> [Double] {
        var s = [Double](repeating: 0, count: Self.frameLength)
        var mean = 0.0
        for i in 0..<Self.frameLength {
            s[i] = sample(startSample + i)
            mean += s[i]
        }
        mean /= Double(Self.frameLength)

        // remove DC offset, then preemphasis (first sample is replicated)
        var prev = s[0] - mean
        for i in 0..<Self.frameLength {
            let centered = s[i] - mean
            s[i] = (centered - 0.97 * prev) * window[i]
            prev = centered
        }

        var re = [Double](repeating: 0, count: Self.fftSize)
        var im = [Double](repeating: 0, count: Self.fftSize)
        for i in 0..<Self.frameLength { re[i] = s[i] }
        Self.fft(&re, &im)

        var out = [Double](repeating: 0, count: Self.numMelBins)
        for b in 0..<Self.numMelBins {
            var acc = 0.0
            let row = melBank[b]
            for k in 0..<Self.numFftBins {
                acc += (re[k] * re[k] + im[k] * im[k]) * row[k]
            }
            out[b] = acc
        }
        return out
    }

    // MARK: - Mel filterbank

    private static func makeMelBank() -> [[Double]] {
        let nyquist = 0.5 * Double(sampleRate)
        var highFreq = -400.0
        if highFreq <= 0 { highFreq += nyquist }

        func mel(_ f: Double) -> Double { 1127.0 * log(1.0 + f / 700.0) }

        let melLow = mel(20.0)
        let melHigh = mel(highFreq)
        let delta = (melHigh - melLow) / Double(numMelBins + 1)
        let binWidth = Double(sampleRate) / Double(fftSize)

        var bins = [[Double]](repeating: [Double](repeating: 0, count: numFftBins),
                              count: numMelBins)
        for b in 0..<numMelBins {
            let left = melLow + Double(b) * delta
            let center = melLow + Double(b + 1) * delta
            let right = melLow + Double(b + 2) * delta
            for k in 0..<numFftBins {
                let m = mel(binWidth * Double(k))
                let up = (m - left) / (center - left)
                let down = (right - m) / (right - center)
                bins[b][k] = max(0.0, min(up, down))
            }
        }
        return bins
    }

    // MARK: - FFT (radix-2, in place)

    private static func fft(_ re: inout [Double], _ im: inout [Double]) {
        let n = re.count
        var j = 0
        for i in 1..<n {
            var bit = n >> 1
            while j & bit != 0 {
                j ^= bit
                bit >>= 1
            }
            j |= bit
            if i < j {
                re.swapAt(i, j)
                im.swapAt(i, j)
            }
        }

        var length = 2
        while length <= n {
            let angle = -2.0 * Double.pi / Double(length)
            let wRe = cos(angle)
            let wIm = sin(angle)
            var i = 0
            while i < n {
                var curRe = 1.0
                var curIm = 0.0
                let half = length / 2
                for k in 0..<half {
                    let aRe = re[i + k]
                    let aIm = im[i + k]
                    let bRe = re[i + k + half] * curRe - im[i + k + half] * curIm
                    let bIm = re[i + k + half] * curIm + im[i + k + half] * curRe
                    re[i + k] = aRe + bRe
                    im[i + k] = aIm + bIm
                    re[i + k + half] = aRe - bRe
                    im[i + k + half] = aIm - bIm
                    let nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += length
            }
            length <<= 1
        }
    }
}

/// Turns a continuous 16 kHz stream into overlapping fbank frames.
///
/// With `snip_edges = false` frame `n` covers absolute samples
/// `[n * 160 - 120, n * 160 + 280)`, where negative indices are mirrored
/// (`x[-k] = x[k]`), exactly like torchaudio's reflection padding.
final class FbankStream {
    private let computer = FbankComputer()
    private var samples: [Float] = []
    private var base = 0          // absolute index of samples[0]
    private var nextFrame = 0

    var pendingSamples: Int { base + samples.count }

    func reset() {
        samples.removeAll(keepingCapacity: true)
        base = 0
        nextFrame = 0
    }

    /// Appends audio and returns every frame that is now complete.
    func push(_ newSamples: [Float]) -> [[Float]] {
        samples.append(contentsOf: newSamples)

        var frames: [[Float]] = []
        // Frame n needs samples up to n * 160 + 279 inclusive.
        while nextFrame * FbankComputer.frameShift + 280 <= base + samples.count {
            let start = nextFrame * FbankComputer.frameShift - FbankComputer.leftPadding
            frames.append(computer.frame(startSample: start, sample: { abs in
                if abs < 0 {
                    // Reflection: padded[-k] = x[k - 1] ... i.e. sample(-1) == x[0], sample(-120) == x[119]
                    let idx = -abs - 1
                    return idx < self.samples.count ? Double(self.samples[idx]) : 0
                }
                let idx = abs - self.base
                guard idx >= 0, idx < self.samples.count else { return 0 }
                return Double(self.samples[idx])
            }))
            nextFrame += 1
        }

        // Drop audio that no future frame can reference.
        let keepFrom = max(0, nextFrame * FbankComputer.frameShift - FbankComputer.leftPadding)
        if keepFrom > base {
            samples.removeFirst(keepFrom - base)
            base = keepFrom
        }
        return frames
    }
}
