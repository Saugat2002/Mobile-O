import Foundation

/// Persists benchmark exports under Application Support (survives Xcode rebuilds; cleared only on app delete).
enum BenchmarkStorage {
    private static let benchmarksFolderName = "Benchmarks"
    private static let indexFileName = "benchmarks_index.json"
    private static let migrationFlagKey = "benchmarks_migrated_to_application_support"
    /// Prevents re-entrant migration when `loadIndex()` is called from inside migration/import.
    private static var migrationInProgress = false

    static var rootDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent(benchmarksFolderName, isDirectory: true)
    }

    static func ensureDirectories() throws {
        try FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
    }

    static func csvURL(runId: String) -> URL {
        rootDirectory.appendingPathComponent("mobileo_benchmark_\(runId).csv")
    }

    static func jsonURL(runId: String) -> URL {
        rootDirectory.appendingPathComponent("mobileo_benchmark_\(runId).json")
    }

    static func summaryURL(runId: String) -> URL {
        rootDirectory.appendingPathComponent("mobileo_benchmark_\(runId)_summary.json")
    }

    private static var indexURL: URL {
        rootDirectory.appendingPathComponent(indexFileName)
    }

    // MARK: - Index

    static func loadIndex() -> [BenchmarkRunIndexEntry] {
        if !migrationInProgress, !UserDefaults.standard.bool(forKey: migrationFlagKey) {
            migrateLegacyDocumentsIfNeeded()
        }
        return readIndexFromDisk()
    }

    private static func readIndexFromDisk() -> [BenchmarkRunIndexEntry] {
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return [] }
        do {
            let data = try Data(contentsOf: indexURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([BenchmarkRunIndexEntry].self, from: data)
        } catch {
            print("BenchmarkStorage: failed to load index: \(error)")
            return []
        }
    }

    static func saveIndex(_ entries: [BenchmarkRunIndexEntry]) throws {
        try ensureDirectories()
        let encoder = makeJSONEncoder()
        let data = try encoder.encode(entries.sorted { $0.createdAt > $1.createdAt })
        try data.write(to: indexURL, options: .atomic)
    }

    private static func makeJSONEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    static func registerRun(summary: BenchmarkRunSummary, csvURL: URL, jsonURL: URL) throws {
        try ensureDirectories()
        try makeJSONEncoder().encode(summary).write(to: summaryURL(runId: summary.runId), options: .atomic)

        var entries = readIndexFromDisk().filter { $0.runId != summary.runId }
        let entry = BenchmarkRunIndexEntry(
            runId: summary.runId,
            createdAt: summary.createdAt,
            deviceModel: summary.deviceModel,
            osVersion: summary.osVersion,
            config: summary.config,
            summary: summary
        )
        entries.insert(entry, at: 0)
        try saveIndex(entries)
        print("BENCHMARK_STORAGE=\(rootDirectory.path)")
    }

    static func loadSummary(runId: String) -> BenchmarkRunSummary? {
        let url = summaryURL(runId: runId)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(BenchmarkRunSummary.self, from: data)
        } catch {
            return nil
        }
    }

    static func deleteRun(runId: String) throws {
        let fm = FileManager.default
        for url in [csvURL(runId: runId), jsonURL(runId: runId), summaryURL(runId: runId)] {
            if fm.fileExists(atPath: url.path) {
                try fm.removeItem(at: url)
            }
        }
        BenchmarkArtifactStore.deleteArtifacts(runId: runId)
        let entries = readIndexFromDisk().filter { $0.runId != runId }
        try saveIndex(entries)
    }

    // MARK: - Migration from Documents/Benchmarks (older builds)

    static func migrateLegacyDocumentsIfNeeded() {
        if UserDefaults.standard.bool(forKey: migrationFlagKey) { return }
        if migrationInProgress { return }

        migrationInProgress = true
        defer { migrationInProgress = false }

        let legacyDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Benchmarks", isDirectory: true)
        guard FileManager.default.fileExists(atPath: legacyDir.path) else {
            UserDefaults.standard.set(true, forKey: migrationFlagKey)
            return
        }

        do {
            try ensureDirectories()
            let files = try FileManager.default.contentsOfDirectory(at: legacyDir, includingPropertiesForKeys: nil)
            for file in files {
                let dest = rootDirectory.appendingPathComponent(file.lastPathComponent)
                if !FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.copyItem(at: file, to: dest)
                }
            }
            rebuildIndexFromDisk()
            UserDefaults.standard.set(true, forKey: migrationFlagKey)
            print("BenchmarkStorage: migrated legacy benchmarks from Documents")
        } catch {
            print("BenchmarkStorage: migration failed: \(error)")
        }
    }

    /// Rebuild index by scanning summary/json files on disk.
    static func rebuildIndexFromDisk() {
        guard let files = try? FileManager.default.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)
        else { return }

        var entries: [BenchmarkRunIndexEntry] = []
        for file in files where file.lastPathComponent.hasSuffix("_summary.json") {
            let runId = extractRunId(from: file)
            guard let summary = loadSummary(runId: runId) else { continue }
            entries.append(makeIndexEntry(from: summary))
        }

        if entries.isEmpty {
            for file in files where file.pathExtension == "json" && !file.lastPathComponent.contains("_summary") {
                guard let summary = importSummaryFromLegacyJSON(at: file) else { continue }
                entries.append(makeIndexEntry(from: summary))
                let summaryData = try? makeJSONEncoder().encode(summary)
                try? summaryData?.write(to: summaryURL(runId: summary.runId), options: .atomic)
            }
        }

        if !entries.isEmpty {
            try? saveIndex(entries)
        }
    }

    private static func makeIndexEntry(from summary: BenchmarkRunSummary) -> BenchmarkRunIndexEntry {
        BenchmarkRunIndexEntry(
            runId: summary.runId,
            createdAt: summary.createdAt,
            deviceModel: summary.deviceModel,
            osVersion: summary.osVersion,
            config: summary.config,
            summary: summary
        )
    }

    private static func extractRunId(from summaryFile: URL) -> String {
        let name = summaryFile.deletingPathExtension().lastPathComponent
        return name
            .replacingOccurrences(of: "mobileo_benchmark_", with: "")
            .replacingOccurrences(of: "_summary", with: "")
    }

    private static func deviceMetadataFromCSV(runId: String) -> (model: String, os: String)? {
        let csv = csvURL(runId: runId)
        guard let text = try? String(contentsOf: csv, encoding: .utf8) else { return nil }
        for line in text.split(separator: "\n").dropFirst() {
            let parts = line.split(separator: ",", maxSplits: 10, omittingEmptySubsequences: false)
            guard parts.count >= 3 else { continue }
            let model = String(parts[1])
            let osField = String(parts[2])
            let os = osField.hasPrefix("iOS ") ? String(osField.dropFirst(4)) : osField
            return (model, os)
        }
        return nil
    }

    // MARK: - Date parsing (legacy JSON + run id fallback)

    private static func parseCreatedAt(jsonRoot: [String: Any], runId: String) -> Date {
        if let iso = jsonRoot["created_at"] as? String, let parsed = parseISO8601String(iso) {
            return parsed
        }
        if let fromRunId = parseRunIdAsDate(runId) {
            return fromRunId
        }
        return Date()
    }

    private static func parseISO8601String(_ string: String) -> Date? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = iso.date(from: trimmed) { return date }

        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: trimmed) { return date }

        let posix = Locale(identifier: "en_US_POSIX")
        let formats = [
            "yyyy-MM-dd'T'HH:mm:ssZ",
            "yyyy-MM-dd'T'HH:mm:ssXXXXX",
            "yyyy-MM-dd HH:mm:ss"
        ]
        for format in formats {
            let df = DateFormatter()
            df.locale = posix
            df.dateFormat = format
            if let date = df.date(from: trimmed) { return date }
        }
        return nil
    }

    private static func parseRunIdAsDate(_ runId: String) -> Date? {
        let posix = Locale(identifier: "en_US_POSIX")
        let df = DateFormatter()
        df.locale = posix
        df.dateFormat = "yyyyMMdd_HHmmss"
        return df.date(from: runId)
    }

    private static func importSummaryFromLegacyJSON(at url: URL) -> BenchmarkRunSummary? {
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let runId = root["run_id"] as? String
        else { return nil }

        let deviceMeta = deviceMetadataFromCSV(runId: runId) ?? BenchmarkExporter.deviceMetadata()
        let createdAt = parseCreatedAt(jsonRoot: root, runId: runId)

        let configDict = root["config"] as? [String: String] ?? [:]
        var config = BenchmarkRunConfiguration()
        config.warmupRuns = Int(configDict["warmup_runs"] ?? "1") ?? 1
        config.measuredRuns = Int(configDict["measured_runs"] ?? "3") ?? 3
        config.numSteps = Double(configDict["num_steps"] ?? "15") ?? 15
        config.enableCFG = (configDict["enable_cfg"] ?? "true") == "true"
        config.guidanceScale = Double(configDict["guidance_scale"] ?? "1.3") ?? 1.3
        if let sched = configDict["scheduler"] {
            config.schedulerRawValue = sched
        }

        guard let metricsArray = root["metrics"] as? [[String: Any]] else { return nil }
        let rows: [BenchmarkMetricRow] = metricsArray.compactMap { dict in
            guard let task = dict["task"] as? String,
                  let phase = dict["phase"] as? String,
                  let metric = dict["metric"] as? String,
                  let value = numberValue(dict["value"]),
                  let unit = dict["unit"] as? String
            else { return nil }
            return BenchmarkMetricRow(
                runId: runId,
                deviceModel: deviceMeta.model,
                osVersion: deviceMeta.os,
                task: task,
                phase: phase,
                runIndex: dict["run_index"] as? Int ?? 0,
                isWarmup: dict["is_warmup"] as? Bool ?? false,
                metric: metric,
                value: value,
                unit: unit,
                notes: dict["notes"] as? String ?? ""
            )
        }

        return BenchmarkResultsParser.makeRunSummary(
            runId: runId,
            createdAt: createdAt,
            deviceModel: deviceMeta.model,
            osVersion: deviceMeta.os,
            config: config,
            rows: rows
        )
    }

    /// JSONSerialization may decode numbers as NSNumber rather than Double.
    private static func numberValue(_ any: Any?) -> Double? {
        switch any {
        case let value as Double: return value
        case let value as Float: return Double(value)
        case let value as Int: return Double(value)
        case let value as NSNumber: return value.doubleValue
        default: return nil
        }
    }
}
