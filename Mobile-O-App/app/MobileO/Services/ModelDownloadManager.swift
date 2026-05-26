import Foundation
import CoreML

/// Orchestrates downloading model weights from HuggingFace, compiling `.mlpackage` → `.mlmodelc`,
/// and managing download lifecycle (pause/resume/retry).
@Observable
@MainActor
final class ModelDownloadManager: NSObject {

    // MARK: - State Machine

    enum State: Equatable {
        case idle
        case downloading
        case paused
        case compiling
        case completed
        case failed(String)
    }

    // MARK: - Observable Properties

    private(set) var state: State = .idle
    private(set) var overallProgress: Double = 0
    private(set) var currentFileProgress: Double = 0
    private(set) var currentFileName: String = ""
    private(set) var downloadedBytes: Int64 = 0
    private(set) var totalBytes: Int64 = 0
    private(set) var downloadSpeed: Double = 0  // bytes/sec
    private(set) var eta: TimeInterval = 0
    private(set) var completedComponents: Set<String> = []
    private(set) var compilationProgress: String = ""

    /// True when all models are on disk and compiled — gates access to the main app.
    private(set) var modelsReady = false
    /// True when `llm/config.json` reports 8-bit weights (paper deployment).
    private(set) var paperPackAligned = false
    /// Non-nil when LLM pack does not match paper (e.g. HuggingFace still serves 4-bit).
    private(set) var llmAlignmentNotice: String?

    private var activeManifest: [(remotePath: String, localPath: String, component: String)] = []
    private var llmOnlyDownload = false

    /// Check the filesystem for all required model files.
    func checkModelsReady() {
        let fm = FileManager.default
        let dir = modelsDirectory
        let coreMLReady = Self.coreMLComponents.allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent("\($0).mlmodelc").path)
        }
        let llmReady = fm.fileExists(atPath: dir.appendingPathComponent("llm/model.safetensors").path)
        modelsReady = coreMLReady && llmReady
        refreshPaperAlignment()
    }

    func refreshPaperAlignment() {
        let llmDir = modelsDirectory.appendingPathComponent("llm")
        guard FileManager.default.fileExists(atPath: llmDir.appendingPathComponent("config.json").path) else {
            paperPackAligned = false
            llmAlignmentNotice = nil
            return
        }
        let bits = LLMPackReader.bits(at: llmDir)
        paperPackAligned = bits == ModelPackRequirements.llmBits
        if let bits, bits != ModelPackRequirements.llmBits {
            llmAlignmentNotice = """
            Installed LLM is \(bits)-bit; the paper uses \(ModelPackRequirements.llmBits)-bit MLX + Core ML FP32 with <2 GB target. \
            HuggingFace may still ship 4-bit (~356 MB). Re-export: `python export.py --only llm --llm-bits 8` and replace the app `Models/llm/` folder via Xcode, or re-download when the Hub pack updates.
            """
        } else {
            llmAlignmentNotice = nil
        }
    }

    /// Re-download only `llm/` (e.g. after Hub publishes 8-bit weights).
    func startLLMReDownload() {
        switch state {
        case .downloading, .compiling:
            return
        default:
            break
        }
        modelsReady = false
        try? FileManager.default.removeItem(at: modelsDirectory.appendingPathComponent("llm"))
        llmOnlyDownload = true
        activeManifest = Self.fileManifest.filter { $0.component == "llm" }
        currentFileIndex = 0
        bytesFromCompletedFiles = 0
        downloadedBytes = 0
        totalBytes = 450_000_000
        completedComponents.remove("llm")
        startDownload()
    }

    /// Root directory where downloaded models live.
    let modelsDirectory: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    // MARK: - Manifest

    private static let repo = "Amshaker/Mobile-O-0.5B-iOS"
    private static let baseURL = "https://huggingface.co/\(repo)/resolve/main"
    static let coreMLComponents = ["connector", "transformer", "vae_decoder", "vision_encoder"]

    /// Each downloadable file: (relative path inside the repo, relative destination on disk).
    private static let fileManifest: [(remotePath: String, localPath: String, component: String)] = {
        var files: [(String, String, String)] = []

        // CoreML .mlpackage files
        for name in coreMLComponents {
            files.append(("\(name).mlpackage/Manifest.json",
                          "\(name).mlpackage/Manifest.json", name))
            files.append(("\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel",
                          "\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel", name))
            files.append(("\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                          "\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin", name))
        }

        // LLM files
        let llmFiles = [
            "added_tokens.json", "config.json", "merges.txt",
            "model.safetensors", "model.safetensors.index.json",
            "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json"
        ]
        for f in llmFiles {
            files.append(("llm/\(f)", "llm/\(f)", "llm"))
        }

        return files
    }()

    // MARK: - Private State

    private var session: URLSession!
    private var currentTask: URLSessionDownloadTask?
    private var currentFileIndex = 0
    private var speedSamples: [(Date, Int64)] = []
    private var downloadStartTime: Date?
    /// Cumulative bytes from files that finished downloading (not counting current in-progress file).
    private var bytesFromCompletedFiles: Int64 = 0
    /// Current file's totalBytesWritten (reset per file).
    private var currentFileBytesWritten: Int64 = 0

    private var progressFilePath: URL {
        modelsDirectory.appendingPathComponent(".download_progress")
    }

    // Download continuation
    private var downloadContinuation: CheckedContinuation<URL, Error>?

    // MARK: - Init

    override init() {
        super.init()

        let config = URLSessionConfiguration.background(withIdentifier: "com.ival.mobileo.modeldownload")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsCellularAccess = true
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: .main)

        // Restore progress if we were mid-download
        restoreProgress()
        checkModelsReady()
    }

    // MARK: - Public Actions

    /// Check available disk space. Returns true if there's enough room (~5 GB buffer).
    func hasSufficientDiskSpace() -> Bool {
        let requiredBytes: Int64 = llmOnlyDownload ? 800_000_000 : 5_000_000_000
        let resourceValues = try? modelsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let available = resourceValues?.volumeAvailableCapacityForImportantUsage ?? 0
        return available >= requiredBytes
    }

    /// Start downloading all model files sequentially.
    func startDownload() {
        guard state == .idle || state == .paused || state != .downloading else { return }

        if case .failed = state {
            // Retry: keep currentFileIndex where it was
        }

        if activeManifest.isEmpty {
            activeManifest = Self.fileManifest
        }

        if !hasSufficientDiskSpace() {
            state = .failed("Not enough disk space. Please free at least 5 GB and try again.")
            return
        }

        state = .downloading
        downloadStartTime = Date()
        speedSamples.removeAll()

        // Estimate total bytes (actual will be refined by Content-Length headers)
        if totalBytes == 0 {
            totalBytes = 3_630_000_000 // ~3.63 GB estimate
        }

        Task { await downloadNextFile() }
    }

    func pause() {
        guard state == .downloading else { return }
        currentTask?.cancel()
        currentTask = nil
        state = .paused
        saveProgress()
    }

    func resume() {
        guard state == .paused else { return }
        startDownload()
    }

    func cancel() {
        currentTask?.cancel()
        currentTask = nil
        state = .idle
        currentFileIndex = 0
        downloadedBytes = 0
        llmOnlyDownload = false
        activeManifest = []
        cleanupProgressFiles()
    }

    func retry() {
        guard case .failed = state else { return }
        startDownload()
    }

    // MARK: - Sequential Download Engine

    private func downloadNextFile() async {
        let manifest = activeManifest.isEmpty ? Self.fileManifest : activeManifest
        guard currentFileIndex < manifest.count else {
            if llmOnlyDownload {
                llmOnlyDownload = false
                activeManifest = []
                cleanupProgressFiles()
                state = .completed
                checkModelsReady()
                ModelPackRequirements.markPackInstalled()
                return
            }
            await compileModels()
            return
        }

        let entry = manifest[currentFileIndex]
        currentFileName = entry.localPath

        // Skip if file already exists
        let destURL = modelsDirectory.appendingPathComponent(entry.localPath)
        if FileManager.default.fileExists(atPath: destURL.path) {
            markComponentIfComplete(entry.component)
            currentFileIndex += 1
            saveProgress()
            await downloadNextFile()
            return
        }

        // Create parent directories
        let parentDir = destURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        let remoteURL = URL(string: "\(Self.baseURL)/\(entry.remotePath)")!

        do {
            let localURL = try await downloadFile(from: remoteURL)

            // Move to final destination
            try? FileManager.default.removeItem(at: destURL) // remove if exists
            try FileManager.default.moveItem(at: localURL, to: destURL)

            // Accumulate completed bytes and reset per-file counter
            bytesFromCompletedFiles += currentFileBytesWritten
            currentFileBytesWritten = 0
            currentFileProgress = 0

            markComponentIfComplete(entry.component)
            currentFileIndex += 1
            saveProgress()

            // Continue to next file if still downloading
            guard state == .downloading else { return }
            await downloadNextFile()
        } catch {
            if (error as NSError).code == NSURLErrorCancelled {
                // User paused — don't treat as failure
                return
            }
            state = .failed("Download failed: \(error.localizedDescription)")
            saveProgress()
        }
    }

    private func downloadFile(from url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation
            currentTask = session.downloadTask(with: url)
            currentTask?.resume()
        }
    }

    // MARK: - Compilation

    private func compileModels() async {
        state = .compiling

        for (index, name) in Self.coreMLComponents.enumerated() {
            compilationProgress = "Compiling \(name) (\(index + 1)/\(Self.coreMLComponents.count))..."

            let packageURL = modelsDirectory.appendingPathComponent("\(name).mlpackage")
            let compiledURL = modelsDirectory.appendingPathComponent("\(name).mlmodelc")

            // Skip if already compiled
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                completedComponents.insert("\(name)_compiled")
                continue
            }

            do {
                let tempCompiledURL = try await Task.detached(priority: .userInitiated) {
                    try MLModel.compileModel(at: packageURL)
                }.value

                // Move compiled model to final location
                try? FileManager.default.removeItem(at: compiledURL)
                try FileManager.default.moveItem(at: tempCompiledURL, to: compiledURL)

                // Keep the raw .mlpackage so we can self-heal by recompiling on-device
                // if `.mlmodelc` becomes incompatible/corrupted.

                completedComponents.insert("\(name)_compiled")
            } catch {
                state = .failed("Failed to compile \(name): \(error.localizedDescription)")
                return
            }
        }

        compilationProgress = ""
        cleanupProgressFiles()
        state = .completed
        checkModelsReady()
        ModelPackRequirements.markPackInstalled()
    }

    // MARK: - Component Tracking

    private func markComponentIfComplete(_ component: String) {
        let manifest = activeManifest.isEmpty ? Self.fileManifest : activeManifest
        let componentFiles = manifest.filter { $0.component == component }
        let allExist = componentFiles.allSatisfy { entry in
            FileManager.default.fileExists(atPath: modelsDirectory.appendingPathComponent(entry.localPath).path)
        }
        if allExist {
            completedComponents.insert(component)
        }
    }

    // MARK: - Speed / ETA Calculation

    private func updateSpeed(bytesWritten: Int64) {
        let now = Date()
        speedSamples.append((now, bytesWritten))

        // Keep a 5-second sliding window
        let cutoff = now.addingTimeInterval(-5)
        speedSamples.removeAll { $0.0 < cutoff }

        guard speedSamples.count >= 2,
              let first = speedSamples.first,
              let last = speedSamples.last else { return }

        let elapsed = last.0.timeIntervalSince(first.0)
        guard elapsed > 0 else { return }

        let totalBytesInWindow = speedSamples.reduce(Int64(0)) { $0 + $1.1 }
        downloadSpeed = Double(totalBytesInWindow) / elapsed

        let remaining = Double(totalBytes - downloadedBytes)
        eta = downloadSpeed > 0 ? remaining / downloadSpeed : 0
    }

    // MARK: - Progress Persistence

    private func saveProgress() {
        let info: [String: Any] = [
            "fileIndex": currentFileIndex,
            "bytesFromCompletedFiles": bytesFromCompletedFiles
        ]
        try? JSONSerialization.data(withJSONObject: info)
            .write(to: progressFilePath)
    }

    private func restoreProgress() {
        guard let data = try? Data(contentsOf: progressFilePath),
              let info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        currentFileIndex = info["fileIndex"] as? Int ?? 0
        bytesFromCompletedFiles = info["bytesFromCompletedFiles"] as? Int64 ?? 0
        downloadedBytes = bytesFromCompletedFiles

        if currentFileIndex > 0 && currentFileIndex < Self.fileManifest.count {
            // We have partial progress — set state to paused so user can resume
            state = .paused

            // Re-check completed components
            for entry in Self.fileManifest.prefix(currentFileIndex) {
                markComponentIfComplete(entry.component)
            }
        }
    }

    private func cleanupProgressFiles() {
        try? FileManager.default.removeItem(at: progressFilePath)
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloadManager: URLSessionDownloadDelegate {

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Copy to a temp location before the system cleans up
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try? FileManager.default.copyItem(at: location, to: tempURL)

        MainActor.assumeIsolated {
            downloadContinuation?.resume(returning: tempURL)
            downloadContinuation = nil
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        MainActor.assumeIsolated {
            currentFileBytesWritten = totalBytesWritten
            if totalBytesExpectedToWrite > 0 {
                currentFileProgress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            }

            // Byte-based overall progress
            downloadedBytes = bytesFromCompletedFiles + totalBytesWritten
            overallProgress = totalBytes > 0 ? min(Double(downloadedBytes) / Double(totalBytes), 1.0) : 0

            updateSpeed(bytesWritten: bytesWritten)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        guard let error else { return }

        MainActor.assumeIsolated {
            let nsError = error as NSError

            if nsError.code == NSURLErrorCancelled {
                downloadContinuation?.resume(throwing: error)
                downloadContinuation = nil
                return
            }

            downloadContinuation?.resume(throwing: error)
            downloadContinuation = nil
        }
    }
}
