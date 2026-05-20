import SwiftUI

/// Automated benchmark runner UI — exports CSV/JSON to Documents/Benchmarks/.
struct BenchmarkView: View {
    @Bindable var generationModel: MobileOModel
    @Bindable var understandingModel: ImageUnderstandingModel
    @Bindable var textChatModel: TextChatModel
    @Bindable var settings: SettingsViewModel

    @State private var runner = BenchmarkRunner()
    @State private var showShareSheet = false
    @State private var shareItem: URL?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                configSection
                runSection
                resultsSection
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Benchmark")
        .navigationBarTitleDisplayMode(.large)
        .sheet(isPresented: $showShareSheet) {
            if let shareItem {
                ShareSheet(items: [shareItem])
            } else {
                EmptyView()
            }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Automated profiling")
                .font(.title3.weight(.semibold))
            Text("Runs image generation, image captioning, and text chat with fixed prompts. Saves per-stage time and memory (resident + footprint peak) to CSV and JSON.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Run configuration", systemImage: "slider.horizontal.3")
                .font(.headline)

            Stepper("Warm-up runs: \(runner.warmupRuns)", value: $runner.warmupRuns, in: 0...2)
            Stepper("Measured runs: \(runner.measuredRuns)", value: $runner.measuredRuns, in: 1...10)

            Text("Generation uses current Settings: \(Int(settings.numSteps)) steps, CFG \(settings.enableCFG ? "on" : "off"), \(generationModel.schedulerType.rawValue)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var runSection: some View {
        VStack(spacing: 12) {
            if case .running = runner.state {
                ProgressView(value: runner.progress)
                Text(runner.statusLine)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Button {
                Task {
                    await runner.runFullSuite(
                        generationModel: generationModel,
                        understandingModel: understandingModel,
                        textChatModel: textChatModel,
                        settings: settings
                    )
                }
            } label: {
                Label("Run full benchmark suite", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning || modelsNotReady)

            if modelsNotReady {
                Text("Wait until all models finish loading on the main screen.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder
    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch runner.state {
            case .idle:
                Text("No benchmark run yet.")
                    .foregroundStyle(.secondary)
            case .running(let label):
                Text("Running: \(label)")
                    .foregroundStyle(.secondary)
            case .completed:
                Text("Completed — \(runner.rowCount) metric rows saved.")
                    .foregroundStyle(.green)
                if let csv = runner.lastCSVURL {
                    resultRow(title: "CSV", url: csv)
                }
                if let json = runner.lastJSONURL {
                    resultRow(title: "JSON", url: json)
                }
                Text("Files: On My iPhone → Mobile-O → Documents → Benchmarks")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Text(message)
                    .foregroundStyle(.red)
                    .font(.subheadline)
            }
        }
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func resultRow(title: String, url: URL) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text(title).font(.subheadline.weight(.medium))
                Text(url.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            Button("Share") {
                shareItem = url
                showShareSheet = true
            }
                .buttonStyle(.bordered)
        }
    }

    private var isRunning: Bool {
        if case .running = runner.state { return true }
        return false
    }

    private var modelsNotReady: Bool {
        !generationModel.modelInfo.contains("Ready") ||
        understandingModel.modelInfo.contains("Error") ||
        textChatModel.modelInfo.contains("Error")
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
