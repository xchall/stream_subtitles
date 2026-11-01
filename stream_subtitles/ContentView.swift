import SwiftUI

struct ContentView: View {
    @StateObject private var engine = StreamingEngine()
    @State private var pulse = false

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 0) {
                    transcript
                    controls
                }
            }
            .navigationTitle("Subtitles")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    voiceButton
                    languageMenu
                }
            }
        }
        .tint(.white)
        .preferredColorScheme(.dark)
        .onAppear { if !isSelfTestRun { engine.load() } }
        .task {
            #if DEBUG
            await SelfTest.runIfRequested()
            #endif
        }
        .onChange(of: engine.isListening) { _, listening in
            pulse = listening
        }
    }

    private var isSelfTestRun: Bool {
        #if DEBUG
        return SelfTest.isRequested || SelfTest.isTTSRequested
        #else
        return false
        #endif
    }

    // MARK: - Voice toggle (top right)

    private var voiceButton: some View {
        Button(action: engine.toggleSpeech) {
            Image(systemName: voiceSymbol)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(engine.isSpeechEnabled ? .white : .white.opacity(0.4))
                .symbolEffect(.variableColor.iterative, isActive: engine.isSpeaking)
                .frame(width: 38, height: 30)
                .background(.white.opacity(0.12), in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(engine.isSpeechEnabled ? "Выключить озвучку" : "Включить озвучку")
    }

    private var voiceSymbol: String {
        guard engine.isSpeechEnabled else { return "speaker.slash.fill" }
        return engine.isSpeaking ? "speaker.wave.3.fill" : "speaker.wave.2.fill"
    }

    // MARK: - Infinite transcript

    @ViewBuilder
    private var transcript: some View {
        if engine.transcript.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                placeholder
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 22)
            .padding(.top, 18)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else {
            ScrollView {
                Text(engine.transcript)
                    .font(.system(size: 24, weight: .regular))
                    .foregroundStyle(.white.opacity(0.95))
                    .lineSpacing(7)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 22)
                    .padding(.top, 18)
                    .padding(.bottom, 24)
            }
            .defaultScrollAnchor(.bottom)
            .scrollIndicators(.hidden)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch engine.state {
            case .idle, .loading:
                HStack(spacing: 10) {
                    ProgressView().tint(.white)
                    Text("Загрузка модели…")
                        .font(.system(size: 20))
                        .foregroundStyle(.white.opacity(0.55))
                }
            case .failed(let message):
                Text(message)
                    .font(.system(size: 18))
                    .foregroundStyle(.red.opacity(0.9))
            case .ready:
                Text(engine.language.hasModel
                     ? "Нажмите на микрофон — текст появится здесь."
                     : "Модель для языка \(engine.language.title) пока недоступна.")
                    .font(.system(size: 20))
                    .foregroundStyle(.white.opacity(0.45))
            }
        }
        .padding(.top, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Language picker (top right)

    private var languageMenu: some View {
        Menu {
            ForEach(AppLanguage.allCases) { language in
                Button {
                    engine.select(language)
                } label: {
                    if language == engine.language {
                        Label("\(language.flag)  \(language.title)", systemImage: "checkmark")
                    } else {
                        Text("\(language.flag)  \(language.title)")
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "globe")
                    .font(.system(size: 15, weight: .medium))
                Text(engine.language.shortCode)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 13)
            .padding(.vertical, 7)
            .background(.white.opacity(0.12), in: Capsule())
        }
    }

    // MARK: - Microphone button

    private var controls: some View {
        VStack(spacing: 14) {
            ZStack {
                if engine.isListening {
                    Circle()
                        .fill(Color.red.opacity(0.22))
                        .frame(width: 84, height: 84)
                        .scaleEffect(pulse ? 1.45 : 1.0)
                        .opacity(pulse ? 0.0 : 0.9)
                        .animation(.easeOut(duration: 1.2).repeatForever(autoreverses: false),
                                   value: pulse)
                }

                Button(action: engine.toggleListening) {
                    ZStack {
                        Circle()
                            .fill(engine.isListening
                                  ? AnyShapeStyle(LinearGradient(colors: [.red, .orange],
                                                                 startPoint: .topLeading,
                                                                 endPoint: .bottomTrailing))
                                  : AnyShapeStyle(Color.white.opacity(0.12)))
                            .frame(width: 84, height: 84)
                        Image(systemName: engine.isListening ? "mic.fill" : "mic.slash.fill")
                            .font(.system(size: 30, weight: .semibold))
                            .foregroundStyle(engine.isListening ? .white : .white.opacity(0.65))
                    }
                }
                .buttonStyle(.plain)
                .disabled(!engine.canListen)
                .opacity(engine.canListen ? 1 : 0.4)
            }
            .frame(height: 96)

            Text(statusText)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(statusColor)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
                .animation(.easeInOut(duration: 0.2), value: statusText)
        }
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity)
    }

    private var statusText: String {
        if engine.permissionDenied {
            return "Разрешите доступ к микрофону в Настройках"
        }
        switch engine.state {
        case .idle, .loading:
            return "Подготовка…"
        case .failed:
            return "Ошибка загрузки модели"
        case .ready:
            if !engine.language.hasModel {
                return "Модель для языка \(engine.language.title) пока недоступна"
            }
            return engine.isListening ? "Слушаю…" : "Нажмите, чтобы включить микрофон"
        }
    }

    private var statusColor: Color {
        if case .failed = engine.state { return .red.opacity(0.9) }
        if engine.isListening { return .red.opacity(0.9) }
        return .white.opacity(0.45)
    }
}

#Preview {
    ContentView()
}
