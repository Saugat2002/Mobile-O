import UIKit

/// Resolves a fixed image for captioning / understanding benchmarks.
enum BenchmarkImage {
    private static let assetNames = ["EditExampleKid", "EditExamplePineapple"]

    /// Image used for image-captioning benchmarks (asset catalog or synthetic fallback).
    static func captionTestImage() -> UIImage {
        for name in assetNames {
            if let image = UIImage(named: name) {
                return image
            }
        }
        return syntheticTestImage()
    }

    static var captionImageSource: String {
        for name in assetNames {
            if UIImage(named: name) != nil {
                return "asset:\(name)"
            }
        }
        return "synthetic:512x512"
    }

    /// Simple RGB bitmap when catalog images are missing from the build.
    private static func syntheticTestImage() -> UIImage {
        let size = 512
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size))
        return renderer.image { context in
            UIColor.systemTeal.setFill()
            context.fill(CGRect(x: 0, y: 0, width: size, height: size))
            UIColor.white.setFill()
            context.fill(CGRect(x: 64, y: 64, width: size - 128, height: size - 128))
        }
    }
}
