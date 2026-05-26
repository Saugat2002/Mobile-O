import Foundation
import UIKit

/// One scalar metric row in the benchmark CSV.
struct BenchmarkMetricRow: Codable {
    let runId: String
    let deviceModel: String
    let osVersion: String
    let task: String
    let phase: String
    let runIndex: Int
    let isWarmup: Bool
    let metric: String
    let value: Double
    let unit: String
    let notes: String
}

enum BenchmarkExporter {
    static func iso8601Timestamp(from date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    static func makeRunId() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter.string(from: Date())
    }

    static func deviceMetadata() -> (model: String, os: String) {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
        let os = UIDevice.current.systemVersion
        return (friendlyDeviceName(machineIdentifier: machine), os)
    }

    /// Maps Apple internal machine ids (e.g. `iPhone18,3`) to marketing names for reports.
    static func friendlyDeviceName(machineIdentifier: String) -> String {
        switch machineIdentifier {
        case "iPhone18,1", "iPhone18,2", "iPhone18,3", "iPhone18,4":
            return "iPhone 17"
        case "iPhone17,1", "iPhone17,2", "iPhone17,3", "iPhone17,4":
            return "iPhone 17 Pro"
        case "iPhone16,1", "iPhone16,2":
            return "iPhone 15 Pro"
        case "iPhone15,4", "iPhone15,5":
            return "iPhone 15"
        default:
            if machineIdentifier.hasPrefix("iPhone18,") { return "iPhone 17" }
            if machineIdentifier.hasPrefix("iPhone") {
                return machineIdentifier.replacingOccurrences(of: ",", with: " ")
            }
            return machineIdentifier
        }
    }

    @discardableResult
    static func writeCSV(rows: [BenchmarkMetricRow], runId: String) throws -> URL {
        try BenchmarkStorage.ensureDirectories()
        let fileURL = BenchmarkStorage.csvURL(runId: runId)
        var lines: [String] = [
            "run_id,device_model,os_version,task,phase,run_index,is_warmup,metric,value,unit,notes"
        ]

        for row in rows {
            let escapedNotes = row.notes.replacingOccurrences(of: "\"", with: "\"\"")
            lines.append(
                "\(row.runId),\(row.deviceModel),iOS \(row.osVersion),\(row.task),\(row.phase),\(row.runIndex),\(row.isWarmup),\(row.metric),\(String(format: "%.6f", row.value)),\(row.unit),\"\(escapedNotes)\""
            )
        }

        let body = lines.joined(separator: "\n") + "\n"
        try body.write(to: fileURL, atomically: true, encoding: .utf8)
        print("BENCHMARK_CSV_PATH=\(fileURL.path)")
        return fileURL
    }

    @discardableResult
    static func writeJSON(rows: [BenchmarkMetricRow], runId: String, config: [String: String]) throws -> URL {
        try BenchmarkStorage.ensureDirectories()
        let fileURL = BenchmarkStorage.jsonURL(runId: runId)
        let payload: [String: Any] = [
            "run_id": runId,
            "created_at": iso8601Timestamp(),
            "config": config,
            "metrics": rows.map { row in
                [
                    "task": row.task,
                    "phase": row.phase,
                    "run_index": row.runIndex,
                    "is_warmup": row.isWarmup,
                    "metric": row.metric,
                    "value": row.value,
                    "unit": row.unit,
                    "notes": row.notes
                ] as [String: Any]
            }
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: fileURL, options: .atomic)
        print("BENCHMARK_JSON_PATH=\(fileURL.path)")
        return fileURL
    }
}
