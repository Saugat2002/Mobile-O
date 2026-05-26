import Foundation
import UIKit

/// One benchmark task's saved input/output (first measured run).
struct BenchmarkTaskSample: Codable, Equatable, Sendable {
    let task: String
    let inputPrompt: String?
    let outputText: String?
    /// File name under `Benchmarks/<runId>/` (e.g. `generated.png`).
    let inputImageName: String?
    let outputImageName: String?
}

struct BenchmarkRunArtifacts: Codable, Equatable, Sendable {
    let samples: [BenchmarkTaskSample]
}

enum BenchmarkArtifactStore {
    static func directory(runId: String) -> URL {
        BenchmarkStorage.rootDirectory.appendingPathComponent(runId, isDirectory: true)
    }

    static func imageURL(runId: String, fileName: String) -> URL {
        directory(runId: runId).appendingPathComponent(fileName)
    }

    static func save(
        runId: String,
        generationPrompt: String,
        generatedImage: UIImage?,
        captionImage: UIImage,
        captionPrompt: String,
        captionOutput: String,
        chatPrompt: String,
        chatOutput: String
    ) throws -> BenchmarkRunArtifacts {
        try BenchmarkStorage.ensureDirectories()
        let dir = directory(runId: runId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var genImageName: String?
        if let generatedImage, let data = generatedImage.pngData() {
            genImageName = "generated.png"
            try data.write(to: dir.appendingPathComponent(genImageName!), options: .atomic)
        }

        let captionInputName = "caption_input.png"
        if let data = captionImage.pngData() {
            try data.write(to: dir.appendingPathComponent(captionInputName), options: .atomic)
        }

        let samples: [BenchmarkTaskSample] = [
            BenchmarkTaskSample(
                task: "image_generation",
                inputPrompt: generationPrompt,
                outputText: nil,
                inputImageName: nil,
                outputImageName: genImageName
            ),
            BenchmarkTaskSample(
                task: "image_captioning",
                inputPrompt: captionPrompt,
                outputText: captionOutput,
                inputImageName: captionInputName,
                outputImageName: nil
            ),
            BenchmarkTaskSample(
                task: "text_chat",
                inputPrompt: chatPrompt,
                outputText: chatOutput,
                inputImageName: nil,
                outputImageName: nil
            )
        ]

        return BenchmarkRunArtifacts(samples: samples)
    }

    static func saveImageGenerationOnly(
        runId: String,
        generationPrompt: String,
        generatedImage: UIImage?
    ) throws -> BenchmarkRunArtifacts {
        try BenchmarkStorage.ensureDirectories()
        let dir = directory(runId: runId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var genImageName: String?
        if let generatedImage, let data = generatedImage.pngData() {
            genImageName = "generated.png"
            try data.write(to: dir.appendingPathComponent(genImageName!), options: .atomic)
        }

        let samples = [
            BenchmarkTaskSample(
                task: "image_generation",
                inputPrompt: generationPrompt,
                outputText: nil,
                inputImageName: nil,
                outputImageName: genImageName
            )
        ]
        return BenchmarkRunArtifacts(samples: samples)
    }

    static func deleteArtifacts(runId: String) {
        let dir = directory(runId: runId)
        try? FileManager.default.removeItem(at: dir)
    }
}
