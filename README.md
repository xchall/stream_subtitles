# stream_subtitles

iOS-приложение (SwiftUI, минимум iOS 26) для распознавания речи в реальном времени:
бесконечное окно с текстом, выбор языка в правом верхнем углу (RU / EN / FR / ES) и
кнопка включения/выключения микрофона. Пока работает только английский — для него
в репозитории лежит модель.

Модель запускается на iPhone через **ONNX Runtime** (без Core ML).

## Требования

* Xcode 26.5+ (iOS 26 SDK)
* iOS 26.0+
* Пакет `onnxruntime-swift-package-manager` 1.24.2+ подтягивается через Swift Package Manager

## Запуск

1. Открыть `stream_subtitles.xcodeproj`.
2. Выбрать свою команду (Signing & Capabilities → Team) и устройство.
3. Run. При первом запуске приложение загрузит ~335 МБ моделей (1–2 секунды),
   затем попросит доступ к микрофону.

## Как это работает

Полностью повторяет референсную PyTorch-реализацию из `stt/stream.py`:

* микрофон → AVAudioEngine → конвертация в 16 кГц mono float32;
* kaldi-совместимый fbank (80 бин, 25 мс / 10 мс, окно Povey, preemphasis 0.97,
  `snip_edges = false`) — `Fbank.swift`;
* окно 9 кадров (90 мс), шаг 4 кадра (40 мс), состояние LSTM (12 слоёв) переносится
  между шагами — `StreamingRNNT.swift`;
* жадное декодирование: до 3 символов на кадр, контекст декодера — 2 последних токена;
* VAD нет: обрабатывается каждый кадр, в паузах модель просто ничего не выдаёт.

Текст дописывается в окно вывода по мере появления токенов, так что задержка —
порядка 100–150 мс после произнесения.

### Файлы

| Файл | Назначение |
|------|------------|
| `stream_subtitles/ContentView.swift` | интерфейс: окно вывода, выбор языка, кнопка микрофона |
| `stream_subtitles/StreamingEngine.swift` | состояние, загрузка модели, очередь инференса |
| `stream_subtitles/AudioInput.swift` | захват микрофона и ресемплинг в 16 кГц |
| `stream_subtitles/Fbank.swift` | kaldi-совместимые признаки |
| `stream_subtitles/StreamingRNNT.swift` | потоковый RNN-T декодер на ONNX Runtime |
| `stream_subtitles/AppLanguage.swift` | список языков (EN / RU / FR / ES) |
| `stream_subtitles/SelfTest.swift` | debug-режим проверки на wav-файле |

## Модель

`stream_subtitles/Models/`:

* `encoder.onnx` — 83.4 M параметров, 12-слойный LSTM (512 / 1024), вход `[1, 9, 80]`,
  состояния `[12, 1, 512]` и `[12, 1, 1024]`;
* `decoder.onnx` — предиктор, вход `[1, 2]` int64;
* `joiner.onnx` — выход 500 логитов (BPE-500);
* `tokens.txt` — таблица токенов.

Это экспорт `icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03`
(LibriSpeech, английский).

## Проверка без микрофона (debug)

Можно прогнать wav-файл через тот же самый потоковый декодер:

```bash
xcrun simctl launch --console-pty booted com.dobriy.streamsubtitles \
  --selftest /path/in/container.wav
```

Результат пишется в `Documents/selftest.txt` приложения. Ожидаемый текст для
тестового файла из LibriSpeech:

```
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
```
