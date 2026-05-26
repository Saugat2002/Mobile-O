import Foundation

/// Paper / iOS deployment targets (arXiv 2602.20161 §4.5).
enum ModelPackRequirements {
    /// MLX LLM weight quantization (paper: 8-bit on GPU).
    static let llmBits = 8
    /// Core ML precision for vision, DiT, VAE, MCP (paper: float32).
    static let coreMLPrecisionLabel = "FP32"
    /// Paper claims total footprint below this during mobile deployment.
    static let targetFootprintMB: Double = 2048
    /// Bump when on-disk pack layout or required LLM bits changes (triggers LLM re-fetch prompt).
    static let packRevision = 2
    private static let packRevisionKey = "installedModelPackRevision"

    static var installedPackRevision: Int {
        get { UserDefaults.standard.integer(forKey: packRevisionKey) }
        set { UserDefaults.standard.set(newValue, forKey: packRevisionKey) }
    }

    static func markPackInstalled() {
        installedPackRevision = packRevision
    }

    static var needsPackRevisionUpgrade: Bool {
        installedPackRevision > 0 && installedPackRevision < packRevision
    }
}

enum LLMPackReader {
    struct Quantization: Decodable {
        let bits: Int?
        let group_size: Int?
    }

    struct Config: Decodable {
        let quantization: Quantization?
    }

    static func bits(at llmDirectory: URL) -> Int? {
        let url = llmDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(Config.self, from: data)
        else { return nil }
        return config.quantization?.bits
    }

    static func isPaperAligned(llmDirectory: URL) -> Bool {
        bits(at: llmDirectory) == ModelPackRequirements.llmBits
    }
}
