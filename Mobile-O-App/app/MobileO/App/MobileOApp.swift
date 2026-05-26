import SwiftUI

/// App entry point. Gates access behind model download if needed.
@main
struct MobileOApp: App {
    @State private var downloadManager = ModelDownloadManager()

    var body: some Scene {
        WindowGroup {
            if downloadManager.modelsReady {
                ContentView(
                    modelDirectory: downloadManager.modelsDirectory,
                    downloadManager: downloadManager
                )
            } else {
                ModelDownloadGateView(downloadManager: downloadManager)
            }
        }
    }
}
