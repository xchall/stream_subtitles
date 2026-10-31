import Foundation

enum AppLanguage: String, CaseIterable, Identifiable {
    case english = "en"
    case russian = "ru"
    case french = "fr"
    case spanish = "es"

    var id: String { rawValue }

    /// Short label shown in the picker (top-right corner).
    var shortCode: String { rawValue.uppercased() }

    var title: String {
        switch self {
        case .english: return "English"
        case .russian: return "Русский"
        case .french: return "Français"
        case .spanish: return "Español"
        }
    }

    var flag: String {
        switch self {
        case .english: return "🇬🇧"
        case .russian: return "🇷🇺"
        case .french: return "🇫🇷"
        case .spanish: return "🇪🇸"
        }
    }

    /// Only English ships with a model for now.
    var hasModel: Bool { self == .english }

    /// Directory inside the app bundle that holds the ONNX graphs for this language.
    var modelFolder: String? { hasModel ? rawValue : nil }
}
