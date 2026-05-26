import Foundation

/// Which tasks a benchmark run executes.
enum BenchmarkSuiteMode: String, Codable, Sendable {
    case fullSuite = "full_suite"
    case imageGenerationOnly = "image_generation_only"
}

/// Generation and run-count settings used for a single benchmark suite (independent of main Settings).
struct BenchmarkRunConfiguration: Codable, Equatable, Sendable {
    var warmupRuns: Int = 0
    var measuredRuns: Int = 3
    var numSteps: Double = 15
    var enableCFG: Bool = true
    var guidanceScale: Double = 1.3
    var schedulerRawValue: String = MobileOGenerator.SchedulerType.customScheduler.rawValue
    /// Optional label shown in history (e.g. "Paper-like").
    var presetLabel: String?
    /// If true, unloads DiT/VAE after the image-generation phase of a benchmark suite.
    var unloadDiffusionAfterImageGen: Bool = false
    /// Full 3-task suite vs image generation only (for memory isolation).
    var suiteMode: BenchmarkSuiteMode = .fullSuite

    var scheduler: MobileOGenerator.SchedulerType {
        get { MobileOGenerator.SchedulerType(rawValue: schedulerRawValue) ?? .customScheduler }
        set { schedulerRawValue = newValue.rawValue }
    }

    var configSummary: String {
        let cfg = enableCFG ? "CFG \(String(format: "%.1f", guidanceScale))" : "CFG off"
        let label = presetLabel.map { "\($0) · " } ?? ""
        let mode = suiteMode == .imageGenerationOnly ? " · image gen only" : ""
        return "\(label)\(Int(numSteps)) steps · \(cfg) · \(scheduler.rawValue)\(mode)"
    }

    var isImageGenerationOnly: Bool { suiteMode == .imageGenerationOnly }

    static let paperLike = BenchmarkRunConfiguration(
        warmupRuns: 0,
        measuredRuns: 3,
        numSteps: 20,
        enableCFG: true,
        guidanceScale: 1.5,
        schedulerRawValue: MobileOGenerator.SchedulerType.customScheduler.rawValue,
        presetLabel: "Paper-like"
    )

    static let fast = BenchmarkRunConfiguration(
        warmupRuns: 0,
        measuredRuns: 3,
        numSteps: 8,
        enableCFG: true,
        guidanceScale: 1.3,
        schedulerRawValue: MobileOGenerator.SchedulerType.customScheduler.rawValue,
        presetLabel: "Fast"
    )

    /// Fewer steps, no CFG — lower peak RAM and faster (quality trade-off).
    static let lowMemory = BenchmarkRunConfiguration(
        warmupRuns: 0,
        measuredRuns: 3,
        numSteps: 8,
        enableCFG: false,
        guidanceScale: 1.0,
        schedulerRawValue: MobileOGenerator.SchedulerType.customScheduler.rawValue,
        presetLabel: "Low memory",
        unloadDiffusionAfterImageGen: true
    )

    /// Single image-gen run with memory probes — isolates DiT/VAE + LLM peak vs paper &lt;2 GB claim.
    static let imageGenMemoryProbe = BenchmarkRunConfiguration(
        warmupRuns: 0,
        measuredRuns: 1,
        numSteps: 4,
        enableCFG: false,
        guidanceScale: 1.0,
        schedulerRawValue: MobileOGenerator.SchedulerType.customScheduler.rawValue,
        presetLabel: "Image gen memory probe",
        unloadDiffusionAfterImageGen: true,
        suiteMode: .imageGenerationOnly
    )

    func toConfigDictionary(
        generationPrompt: String,
        captionPrompt: String,
        chatPrompt: String,
        captionImageSource: String
    ) -> [String: String] {
        [
            "warmup_runs": "\(warmupRuns)",
            "measured_runs": "\(measuredRuns)",
            "num_steps": "\(Int(numSteps))",
            "enable_cfg": "\(enableCFG)",
            "guidance_scale": "\(guidanceScale)",
            "scheduler": scheduler.rawValue,
            "preset_label": presetLabel ?? "",
            "suite_mode": suiteMode.rawValue,
            "generation_prompt": generationPrompt,
            "caption_prompt": captionPrompt,
            "chat_prompt": chatPrompt,
            "caption_image_source": captionImageSource
        ]
    }
}

/// Per-task aggregated metrics (measured runs only, excluding warmup).
struct BenchmarkTaskSummary: Codable, Equatable, Sendable {
    let task: String
    let sampleCount: Int
    let wallClockMean: Double?
    let pipelineMean: Double?
    let diffusionMean: Double?
    let vaeMean: Double?
    let visionEncoderMean: Double?
    let ttftMean: Double?
    let totalMean: Double?
    let peakMemoryMean: Double?
    let modelLoadFootprintMB: Double?
    let tokensPerSecondMean: Double?
}

/// Compact summary saved with each run for in-app display.
struct BenchmarkRunSummary: Codable, Equatable, Sendable {
    let runId: String
    let createdAt: Date
    let deviceModel: String
    let osVersion: String
    let config: BenchmarkRunConfiguration
    let tasks: [BenchmarkTaskSummary]
    /// Saved prompts, model outputs, and image files from the first measured run.
    let artifacts: BenchmarkRunArtifacts?
}

/// Index entry for history list (embedded in `benchmarks_index.json`).
struct BenchmarkRunIndexEntry: Codable, Equatable, Identifiable, Sendable {
    var id: String { runId }
    let runId: String
    let createdAt: Date
    let deviceModel: String
    let osVersion: String
    let config: BenchmarkRunConfiguration
    let summary: BenchmarkRunSummary
}
