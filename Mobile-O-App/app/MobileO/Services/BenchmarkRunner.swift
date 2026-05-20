import Foundation
import UIKit

/// Runs automated Mobile-O benchmarks and exports CSV + JSON under Documents/Benchmarks/.
@Observable
@MainActor
final class BenchmarkRunner {
    enum State: Equatable {
        case idle
        case running(String)
        case completed
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var progress: Double = 0
    private(set) var statusLine: String = ""
    private(set) var lastCSVURL: URL?
    private(set) var lastJSONURL: URL?
    private(set) var rowCount: Int = 0

    /// Warm-up runs (discarded from primary stats but still logged).
    var warmupRuns: Int = 1
    /// Measured runs per task.
    var measuredRuns: Int = 3

    private func appendMetric(
        to rows: inout [BenchmarkMetricRow],
        runId: String,
        device: String,
        os: String,
        task: String,
        phase: String,
        runIndex: Int,
        isWarmup: Bool,
        metric: String,
        value: Double,
        unit: String,
        notes: String = ""
    ) {
        rows.append(BenchmarkMetricRow(
            runId: runId,
            deviceModel: device,
            osVersion: os,
            task: task,
            phase: phase,
            runIndex: runIndex,
            isWarmup: isWarmup,
            metric: metric,
            value: value,
            unit: unit,
            notes: notes
        ))
    }

    private func appendMemory(
        to rows: inout [BenchmarkMetricRow],
        runId: String,
        device: String,
        os: String,
        task: String,
        phase: String,
        runIndex: Int,
        isWarmup: Bool,
        before: MemorySnapshot,
        peakMB: Double,
        after: MemorySnapshot
    ) {
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: task, phase: phase,
                     runIndex: runIndex, isWarmup: isWarmup, metric: "memory_resident_before_mb",
                     value: before.residentMB, unit: "MB")
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: task, phase: phase,
                     runIndex: runIndex, isWarmup: isWarmup, metric: "memory_footprint_before_mb",
                     value: before.footprintMB, unit: "MB")
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: task, phase: phase,
                     runIndex: runIndex, isWarmup: isWarmup, metric: "memory_footprint_peak_mb",
                     value: peakMB, unit: "MB")
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: task, phase: phase,
                     runIndex: runIndex, isWarmup: isWarmup, metric: "memory_footprint_after_mb",
                     value: after.footprintMB, unit: "MB")
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: task, phase: phase,
                     runIndex: runIndex, isWarmup: isWarmup, metric: "memory_resident_after_mb",
                     value: after.residentMB, unit: "MB")
    }

    func runFullSuite(
        generationModel: MobileOModel,
        understandingModel: ImageUnderstandingModel,
        textChatModel: TextChatModel,
        settings: SettingsViewModel
    ) async {
        if case .running = state { return }
        state = .running("Starting")
        progress = 0
        lastCSVURL = nil
        lastJSONURL = nil

        let runId = BenchmarkExporter.makeRunId()
        let (device, os) = BenchmarkExporter.deviceMetadata()
        var rows: [BenchmarkMetricRow] = []

        let genPrompt = "a red rose with morning dew drops"
        let captionPrompt = "What is in this image? Describe in one sentence."
        let chatPrompt = "Explain what a neural network is in two sentences."
        let captionImage = BenchmarkImage.captionTestImage()

        let totalSteps = max(1, warmupRuns + measuredRuns)
        let taskCount = 3
        var stepIndex = 0

        func bumpProgress(_ label: String) {
            stepIndex += 1
            progress = Double(stepIndex) / Double(totalSteps * taskCount + 1)
            statusLine = label
            state = .running(label)
        }

        let config: [String: String] = [
            "warmup_runs": "\(warmupRuns)",
            "measured_runs": "\(measuredRuns)",
            "num_steps": "\(Int(settings.numSteps))",
            "enable_cfg": "\(settings.enableCFG)",
            "guidance_scale": "\(settings.guidanceScale)",
            "scheduler": generationModel.schedulerType.rawValue,
            "generation_prompt": genPrompt,
            "caption_prompt": captionPrompt,
            "chat_prompt": chatPrompt,
            "caption_image_source": BenchmarkImage.captionImageSource
        ]

        appendMetric(to: &rows, runId: runId, device: device, os: os, task: "meta", phase: "config",
                     runIndex: 0, isWarmup: false, metric: "warmup_runs", value: Double(warmupRuns), unit: "count")
        appendMetric(to: &rows, runId: runId, device: device, os: os, task: "meta", phase: "config",
                     runIndex: 0, isWarmup: false, metric: "measured_runs", value: Double(measuredRuns), unit: "count")

        // MARK: Image generation
        let genParams = MobileOGenerator.GenerationParameters(
            prompt: genPrompt,
            numSteps: Int(settings.numSteps),
            guidanceScale: settings.enableCFG ? Float(settings.guidanceScale) : 0,
            enableCFG: settings.enableCFG,
            seed: 42,
            progressCallback: nil
        )

        for run in 0..<(warmupRuns + measuredRuns) {
            let isWarmup = run < warmupRuns
            bumpProgress(isWarmup ? "Warm-up image generation" : "Image generation run \(run - warmupRuns + 1)/\(measuredRuns)")

            let before = MemorySnapshot.capture()
            let monitor = MemoryPeakMonitor()
            monitor.start()

            do {
                let (timing, wallClock) = try await generationModel.benchmarkGeneration(genParams)
                let peakMB = monitor.peakFootprintMB()
                let after = MemorySnapshot.capture()

                appendMemory(to: &rows, runId: runId, device: device, os: os, task: "image_generation",
                             phase: "total", runIndex: run, isWarmup: isWarmup, before: before, peakMB: peakMB, after: after)
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "total",
                             runIndex: run, isWarmup: isWarmup, metric: "wall_clock_s", value: wallClock, unit: "s")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "text_encoding",
                             runIndex: run, isWarmup: isWarmup, metric: "tokenization_s", value: timing.tokenizationTime, unit: "s",
                             notes: "estimated split")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "text_encoding",
                             runIndex: run, isWarmup: isWarmup, metric: "llm_encode_s", value: timing.llmTime, unit: "s",
                             notes: "estimated split")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "connector",
                             runIndex: run, isWarmup: isWarmup, metric: "connector_s", value: timing.connectorTime, unit: "s")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "diffusion",
                             runIndex: run, isWarmup: isWarmup, metric: "diffusion_s", value: timing.diffusionTime, unit: "s",
                             notes: "DiT steps")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "vae_decode",
                             runIndex: run, isWarmup: isWarmup, metric: "vae_s", value: timing.vaeTime, unit: "s")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "total",
                             runIndex: run, isWarmup: isWarmup, metric: "pipeline_total_s", value: timing.totalTime, unit: "s")
                appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_generation", phase: "total",
                             runIndex: run, isWarmup: isWarmup, metric: "inference_steps", value: Double(genParams.numSteps), unit: "count")
            } catch {
                _ = monitor.stop()
                state = .failed("Image generation failed: \(error.localizedDescription)")
                return
            }
        }

        // MARK: Image captioning / understanding
        for run in 0..<(warmupRuns + measuredRuns) {
            let isWarmup = run < warmupRuns
            bumpProgress(isWarmup ? "Warm-up captioning" : "Captioning run \(run - warmupRuns + 1)/\(measuredRuns)")

            let before = MemorySnapshot.capture()
            let monitor = MemoryPeakMonitor()
            monitor.start()

            _ = await understandingModel.understand(image: captionImage, prompt: captionPrompt)
            let peakMB = monitor.peakFootprintMB()
            let after = MemorySnapshot.capture()

            appendMemory(to: &rows, runId: runId, device: device, os: os, task: "image_captioning",
                         phase: "total", runIndex: run, isWarmup: isWarmup, before: before, peakMB: peakMB, after: after)
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_captioning", phase: "vision_encoder",
                         runIndex: run, isWarmup: isWarmup, metric: "vision_encoder_s",
                         value: understandingModel.visionEncoderTime, unit: "s")
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_captioning", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "time_to_first_token_s",
                         value: understandingModel.timeToFirstToken, unit: "s")
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_captioning", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "total_s", value: understandingModel.totalTime, unit: "s")
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_captioning", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "tokens_generated",
                         value: Double(understandingModel.tokensGenerated), unit: "count")
            let captionTPS = understandingModel.totalTime > 0
                ? Double(understandingModel.tokensGenerated) / understandingModel.totalTime : 0
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "image_captioning", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "tokens_per_second", value: captionTPS, unit: "tok/s")
        }

        // MARK: Text chat
        let messages: [[String: String]] = [
            ["role": "user", "content": chatPrompt]
        ]

        for run in 0..<(warmupRuns + measuredRuns) {
            let isWarmup = run < warmupRuns
            bumpProgress(isWarmup ? "Warm-up text chat" : "Text chat run \(run - warmupRuns + 1)/\(measuredRuns)")

            let before = MemorySnapshot.capture()
            let monitor = MemoryPeakMonitor()
            monitor.start()

            _ = await textChatModel.chat(messages: messages)
            let peakMB = monitor.peakFootprintMB()
            let after = MemorySnapshot.capture()

            appendMemory(to: &rows, runId: runId, device: device, os: os, task: "text_chat",
                         phase: "total", runIndex: run, isWarmup: isWarmup, before: before, peakMB: peakMB, after: after)
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "text_chat", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "time_to_first_token_s",
                         value: textChatModel.timeToFirstToken, unit: "s")
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "text_chat", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "total_s", value: textChatModel.totalTime, unit: "s")
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "text_chat", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "tokens_generated",
                         value: Double(textChatModel.tokensGenerated), unit: "count")
            let chatTPS = textChatModel.totalTime > 0
                ? Double(textChatModel.tokensGenerated) / textChatModel.totalTime : 0
            appendMetric(to: &rows, runId: runId, device: device, os: os, task: "text_chat", phase: "llm_decode",
                         runIndex: run, isWarmup: isWarmup, metric: "tokens_per_second", value: chatTPS, unit: "tok/s")
        }

        do {
            statusLine = "Writing results…"
            lastCSVURL = try BenchmarkExporter.writeCSV(rows: rows, runId: runId)
            lastJSONURL = try BenchmarkExporter.writeJSON(rows: rows, runId: runId, config: config)
            rowCount = rows.count
            progress = 1
            state = .completed
            statusLine = "Saved \(rowCount) metrics"
        } catch {
            state = .failed("Export failed: \(error.localizedDescription)")
        }
    }
}
