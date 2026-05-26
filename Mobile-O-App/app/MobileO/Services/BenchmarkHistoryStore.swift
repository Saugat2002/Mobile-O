import Foundation

/// Observable history of saved benchmark runs (Application Support).
@Observable
@MainActor
final class BenchmarkHistoryStore {
    static let shared = BenchmarkHistoryStore()

    private(set) var entries: [BenchmarkRunIndexEntry] = []
    private(set) var lastError: String?

    func refresh() {
        entries = BenchmarkStorage.loadIndex()
    }

    func delete(runId: String) {
        do {
            try BenchmarkStorage.deleteRun(runId: runId)
            refresh()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func register(summary: BenchmarkRunSummary, csvURL: URL, jsonURL: URL) {
        do {
            try BenchmarkStorage.registerRun(summary: summary, csvURL: csvURL, jsonURL: jsonURL)
            refresh()
        } catch {
            lastError = error.localizedDescription
        }
    }
}
