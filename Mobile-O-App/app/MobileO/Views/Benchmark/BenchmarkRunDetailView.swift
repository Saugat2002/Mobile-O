import SwiftUI

/// Full results for one saved benchmark run.
struct BenchmarkRunDetailView: View {
    let entry: BenchmarkRunIndexEntry

    @State private var showShareCSV = false
    @State private var showShareJSON = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                configSection
                if let artifacts = entry.summary.artifacts {
                    artifactsSection(artifacts)
                }
                ForEach(entry.summary.tasks, id: \.task) { task in
                    taskCard(task)
                }
                exportSection
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(entry.runId)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showShareCSV) {
            if FileManager.default.fileExists(atPath: BenchmarkStorage.csvURL(runId: entry.runId).path) {
                ShareSheet(items: [BenchmarkStorage.csvURL(runId: entry.runId)])
            }
        }
        .sheet(isPresented: $showShareJSON) {
            if FileManager.default.fileExists(atPath: BenchmarkStorage.jsonURL(runId: entry.runId).path) {
                ShareSheet(items: [BenchmarkStorage.jsonURL(runId: entry.runId)])
            }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(formattedDate(entry.createdAt))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("\(entry.deviceModel) · iOS \(entry.osVersion)")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Configuration", systemImage: "slider.horizontal.3")
                .font(.headline)
            Text(entry.config.configSummary)
                .font(.subheadline)
            configRow("Warm-up / measured", "\(entry.config.warmupRuns) / \(entry.config.measuredRuns)")
            configRow("Scheduler", entry.config.scheduler.rawValue)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func artifactsSection(_ artifacts: BenchmarkRunArtifacts) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Inputs & outputs", systemImage: "text.bubble")
                .font(.headline)
            Text("From the first measured run (after warm-up).")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(artifacts.samples, id: \.task) { sample in
                sampleCard(sample)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func sampleCard(_ sample: BenchmarkTaskSample) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(taskTitle(sample.task))
                .font(.subheadline.weight(.semibold))

            if let prompt = sample.inputPrompt, !prompt.isEmpty {
                labeledTextBlock(title: "Input", text: prompt)
            }

            if let inputName = sample.inputImageName,
               let image = loadArtifactImage(fileName: inputName) {
                labeledImageBlock(title: "Input image", image: image)
            }

            if let outputName = sample.outputImageName,
               let image = loadArtifactImage(fileName: outputName) {
                labeledImageBlock(title: "Output image", image: image)
            }

            if let output = sample.outputText, !output.isEmpty {
                labeledTextBlock(title: "Output", text: output)
            }
        }
        .padding(.vertical, 4)
    }

    private func labeledTextBlock(title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(text)
                .font(.subheadline)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func labeledImageBlock(title: String, image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private func loadArtifactImage(fileName: String) -> UIImage? {
        let url = BenchmarkArtifactStore.imageURL(runId: entry.runId, fileName: fileName)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    @ViewBuilder
    private func taskCard(_ task: BenchmarkTaskSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(taskTitle(task.task), systemImage: taskIcon(task.task))
                .font(.headline)

            Text("\(task.sampleCount) measured runs (warm-up excluded)")
                .font(.caption)
                .foregroundStyle(.secondary)

            if let wall = task.wallClockMean ?? task.pipelineMean {
                metricRow("Total time", formatSeconds(wall))
            }
            if let dit = task.diffusionMean {
                metricRow("DiT (diffusion)", formatSeconds(dit))
            }
            if let vae = task.vaeMean {
                metricRow("VAE decode", formatSeconds(vae))
            }
            if let vision = task.visionEncoderMean {
                metricRow("Vision encoder", formatSeconds(vision))
            }
            if let ttft = task.ttftMean {
                metricRow("Time to first token", formatSeconds(ttft))
            }
            if let total = task.totalMean, task.task != "image_generation" {
                metricRow("LLM total", formatSeconds(total))
            }
            if let tps = task.tokensPerSecondMean, tps > 0 {
                metricRow("Throughput", String(format: "%.1f tok/s", tps))
            }
            if let load = task.modelLoadFootprintMB, task.task == "image_generation" {
                metricRow("After DiT+VAE load", String(format: "%.0f MB", load))
                let loadBudget = MemoryBudget.status(peakMB: load)
                Text(loadBudget.label)
                    .font(.caption2)
                    .foregroundStyle(loadBudget.meetsTarget ? .green : .orange)
            }
            if let mem = task.peakMemoryMean {
                metricRow("Peak memory (inference)", String(format: "%.0f MB", mem))
                let budget = MemoryBudget.status(peakMB: mem)
                Text(budget.label)
                    .font(.caption)
                    .foregroundStyle(budget.meetsTarget ? .green : .orange)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var exportSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Export files", systemImage: "square.and.arrow.up")
                .font(.headline)
            Text("Stored in Application Support (kept across Xcode rebuilds).")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Button("Share CSV") { showShareCSV = true }
                    .buttonStyle(.bordered)
                Button("Share JSON") { showShareJSON = true }
                    .buttonStyle(.bordered)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func configRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).foregroundStyle(.secondary)
            Spacer()
            Text(value).fontWeight(.medium)
        }
        .font(.subheadline)
    }

    private func metricRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).fontWeight(.semibold).monospacedDigit()
        }
        .font(.subheadline)
    }

    private func taskTitle(_ task: String) -> String {
        switch task {
        case "image_generation": return "Image generation"
        case "image_captioning": return "Image captioning"
        case "text_chat": return "Text chat"
        default: return task
        }
    }

    private func taskIcon(_ task: String) -> String {
        switch task {
        case "image_generation": return "photo"
        case "image_captioning": return "eye"
        case "text_chat": return "bubble.left"
        default: return "chart.bar"
        }
    }

    private func formatSeconds(_ value: Double) -> String {
        String(format: "%.2f s", value)
    }

    private func formattedDate(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
