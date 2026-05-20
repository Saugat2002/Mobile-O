import CoreML
import Foundation

// MARK: - Protocol-Based Model Wrappers

extension MobileOGenerator {
    private static func loadModelWithFallbacks(
        modelURL: URL,
        baseConfiguration: MLModelConfiguration,
        componentName: String
    ) throws -> (MLModel, CoreMLComputePath) {
        let requested = CoreMLComputePath.from(baseConfiguration.computeUnits)
        do {
            let model = try MLModel(contentsOf: modelURL, configuration: baseConfiguration)
            print("[MobileO] \(componentName) loaded with \(requested.displayName)")
            return (model, requested)
        } catch {
            print("[MobileO] \(componentName) primary load failed (\(requested.displayName)): \(error.localizedDescription)")
            let cpuGpuConfig = MLModelConfiguration()
            cpuGpuConfig.computeUnits = .cpuAndGPU
            cpuGpuConfig.allowLowPrecisionAccumulationOnGPU = true
            do {
                let model = try MLModel(contentsOf: modelURL, configuration: cpuGpuConfig)
                print("[MobileO] \(componentName) fallback: CPU + GPU")
                return (model, .cpuAndGPU)
            } catch {
                print("[MobileO] \(componentName) CPU+GPU fallback failed: \(error.localizedDescription)")
                let cpuOnlyConfig = MLModelConfiguration()
                cpuOnlyConfig.computeUnits = .cpuOnly
                let model = try MLModel(contentsOf: modelURL, configuration: cpuOnlyConfig)
                print("[MobileO] \(componentName) fallback: CPU only (slowest)")
                return (model, .cpuOnly)
            }
        }
    }

    // MARK: - Transformer Protocol

    /// Unified API for SANA DiT transformer noise prediction
    protocol TransformerModel {
        var computePath: CoreMLComputePath { get }

        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray
    }

    // MARK: - VAE Protocol

    /// Unified API for SANA VAE latent-to-image decoding
    protocol VAEModel {
        var computePath: CoreMLComputePath { get }

        func decode(latent: MLMultiArray) async throws -> MLMultiArray
    }

    // MARK: - FP32 Transformer Wrapper

    /// FP32 SANA transformer with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32Transformer: TransformerModel {
        let computePath: CoreMLComputePath
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            let (loadedModel, path) = try MobileOGenerator.loadModelWithFallbacks(
                modelURL: modelURL,
                baseConfiguration: configuration,
                componentName: "transformer"
            )
            self.model = loadedModel
            self.computePath = path

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in transformer model"])
            }
            self.outputKey = firstOutput.key
        }

        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent),
                "timestep": MLFeatureValue(multiArray: timestep),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
                "encoder_attention_mask": MLFeatureValue(multiArray: encoderAttentionMask)
            ])

            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from transformer"])
            }

            return result
        }
    }

    // MARK: - FP32 VAE Wrapper

    /// FP32 SANA VAE decoder with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32VAE: VAEModel {
        let computePath: CoreMLComputePath
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            let (loadedModel, path) = try MobileOGenerator.loadModelWithFallbacks(
                modelURL: modelURL,
                baseConfiguration: configuration,
                componentName: "vae_decoder"
            )
            self.model = loadedModel
            self.computePath = path

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in vae_decoder model"])
            }
            self.outputKey = firstOutput.key
        }

        func decode(latent: MLMultiArray) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent)
            ])
            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from VAE decoder"])
            }

            return result
        }
    }

    // MARK: - Model Factory

    /// Factory for creating model wrappers based on variant
    struct ModelFactory {
        private static func recompileIfPossible(
            compiledURL: URL,
            packageURL: URL
        ) throws {
            guard FileManager.default.fileExists(atPath: packageURL.path) else { return }
            try? FileManager.default.removeItem(at: compiledURL)
            let tempCompiledURL = try MLModel.compileModel(at: packageURL)
            try? FileManager.default.removeItem(at: compiledURL)
            try FileManager.default.moveItem(at: tempCompiledURL, to: compiledURL)
        }

        static func createTransformer(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> TransformerModel {
            let modelURL = modelDirectory.appendingPathComponent(variant.fileName)
            let packageURL = modelDirectory.appendingPathComponent("transformer.mlpackage")
            do {
                switch variant {
                case .fp32:
                    return try FP32Transformer(modelURL: modelURL, configuration: configuration)
                }
            } catch {
                try recompileIfPossible(compiledURL: modelURL, packageURL: packageURL)
                switch variant {
                case .fp32:
                    return try FP32Transformer(modelURL: modelURL, configuration: configuration)
                }
            }
        }

        static func createVAE(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> VAEModel {
            let modelURL = modelDirectory.appendingPathComponent(variant.vaeFileName)
            let packageURL = modelDirectory.appendingPathComponent("vae_decoder.mlpackage")
            do {
                switch variant {
                case .fp32:
                    return try FP32VAE(modelURL: modelURL, configuration: configuration)
                }
            } catch {
                try recompileIfPossible(compiledURL: modelURL, packageURL: packageURL)
                switch variant {
                case .fp32:
                    return try FP32VAE(modelURL: modelURL, configuration: configuration)
                }
            }
        }
    }
}
