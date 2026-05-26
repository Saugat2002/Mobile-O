import Foundation

enum BenchmarkResultsParser {
    /// Aggregates metric rows into per-task means (non-warmup runs only).
    static func summarize(rows: [BenchmarkMetricRow], measuredRuns: Int) -> [BenchmarkTaskSummary] {
        let tasks = ["image_generation", "image_captioning", "text_chat"]
        return tasks.compactMap { task in
            let hasMetrics = rows.contains { $0.task == task && !$0.isWarmup }
            guard hasMetrics else { return nil }
            return summarizeTask(rows: rows, task: task, measuredRuns: measuredRuns)
        }
    }

    private static func summarizeTask(
        rows: [BenchmarkMetricRow],
        task: String,
        measuredRuns: Int
    ) -> BenchmarkTaskSummary {
        let measured = rows.filter { $0.task == task && !$0.isWarmup }
        let indices = Set(measured.map(\.runIndex)).sorted()
        let usedIndices = Array(indices.suffix(measuredRuns))
        let filtered = measured.filter { usedIndices.contains($0.runIndex) }

        func mean(_ metric: String, phase: String? = nil) -> Double? {
            let values = filtered.filter { row in
                row.metric == metric && (phase == nil || row.phase == phase)
            }.map(\.value)
            guard !values.isEmpty else { return nil }
            return values.reduce(0, +) / Double(values.count)
        }

        let modelLoad = rows.first(where: {
            $0.task == task && $0.phase == "model_load" && $0.metric == "memory_footprint_mb"
        })?.value

        return BenchmarkTaskSummary(
            task: task,
            sampleCount: usedIndices.count,
            wallClockMean: mean("wall_clock_s", phase: "total"),
            pipelineMean: mean("pipeline_total_s", phase: "total"),
            diffusionMean: mean("diffusion_s", phase: "diffusion"),
            vaeMean: mean("vae_s", phase: "vae_decode"),
            visionEncoderMean: mean("vision_encoder_s", phase: "vision_encoder"),
            ttftMean: mean("time_to_first_token_s", phase: "llm_decode"),
            totalMean: mean("total_s", phase: "llm_decode"),
            peakMemoryMean: mean("memory_footprint_peak_mb", phase: "total"),
            modelLoadFootprintMB: modelLoad,
            tokensPerSecondMean: mean("tokens_per_second", phase: "llm_decode")
        )
    }

    static func makeRunSummary(
        runId: String,
        createdAt: Date,
        deviceModel: String,
        osVersion: String,
        config: BenchmarkRunConfiguration,
        rows: [BenchmarkMetricRow],
        artifacts: BenchmarkRunArtifacts? = nil
    ) -> BenchmarkRunSummary {
        BenchmarkRunSummary(
            runId: runId,
            createdAt: createdAt,
            deviceModel: deviceModel,
            osVersion: osVersion,
            config: config,
            tasks: summarize(rows: rows, measuredRuns: config.measuredRuns),
            artifacts: artifacts
        )
    }
}
