import Foundation
import Darwin

/// Process memory snapshot for benchmark logging.
struct MemorySnapshot: Codable, Sendable {
    let residentMB: Double
    let footprintMB: Double
    let timestampISO: String

    static func capture() -> MemorySnapshot {
        let resident = Double(MemorySampler.residentBytes()) / 1_048_576.0
        let footprint = Double(MemorySampler.physFootprintBytes()) / 1_048_576.0
        return MemorySnapshot(
            residentMB: resident,
            footprintMB: footprint,
            timestampISO: ISO8601DateFormatter().string(from: Date())
        )
    }
}

enum MemorySampler {
    static func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return info.resident_size
    }

    static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return residentBytes() }
        return info.phys_footprint
    }
}

/// Polls memory while a benchmark task runs and tracks peak footprint.
final class MemoryPeakMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var peakFootprintBytes: UInt64 = 0
    private var pollTask: Task<Void, Never>?

    func start() {
        stop()
        peakFootprintBytes = MemorySampler.physFootprintBytes()
        pollTask = Task {
            while !Task.isCancelled {
                let current = MemorySampler.physFootprintBytes()
                lock.lock()
                if current > peakFootprintBytes { peakFootprintBytes = current }
                lock.unlock()
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
    }

    func stop() -> UInt64 {
        pollTask?.cancel()
        pollTask = nil
        lock.lock()
        let peak = peakFootprintBytes
        lock.unlock()
        return peak
    }

    func peakFootprintMB() -> Double {
        Double(stop()) / 1_048_576.0
    }
}
