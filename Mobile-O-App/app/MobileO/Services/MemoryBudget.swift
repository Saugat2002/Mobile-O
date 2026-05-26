import Foundation

/// Compares measured footprint against the paper's <2 GB deployment target.
enum MemoryBudget {
    static let targetMB = ModelPackRequirements.targetFootprintMB

    static func status(peakMB: Double) -> (label: String, meetsTarget: Bool) {
        if peakMB <= 0 {
            return ("No measurement", false)
        }
        let meets = peakMB <= targetMB
        let label = meets
            ? String(format: "Within paper target (%.0f MB ≤ %.0f MB)", peakMB, targetMB)
            : String(format: "Above paper target (%.0f MB > %.0f MB)", peakMB, targetMB)
        return (label, meets)
    }
}
