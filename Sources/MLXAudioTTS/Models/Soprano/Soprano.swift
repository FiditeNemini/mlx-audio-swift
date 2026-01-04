//
//  Soprano.swift
//  MLXAudio
//
//  Soprano TTS Model - Ultra-lightweight text-to-speech with ~80M parameters.
//  Ported from https://github.com/ekwek1/soprano
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
import MLXLMCommon
import MLXFast
import MLXNN
import MLXAudioCore

// MARK: - Type Aliases

public typealias SopranoError = AudioGenerationError
public typealias SopranoGenerationInfo = AudioGenerationInfo
public typealias SopranoGeneration = AudioGeneration

// MARK: - Soprano Attention

private class SopranoAttention: Module {
    let args: SopranoConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: SopranoConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim

        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: args.ropeTheta,
            scale: 1.0
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

private class SopranoMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

private class SopranoTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SopranoAttention
    let mlp: SopranoMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: SopranoConfiguration) {
        self._attention.wrappedValue = SopranoAttention(args)
        self.mlp = SopranoMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Inner Model

private class SopranoModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [SopranoTransformerBlock]
    let norm: RMSNorm

    init(_ args: SopranoConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers).map { _ in
            SopranoTransformerBlock(args)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Soprano Model

public class SopranoModel: Module, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?

    private let model: SopranoModelInner
    let configuration: SopranoConfiguration
    let decoder: SopranoDecoder

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // Token IDs
    private var stopTokenId: Int?

    // KVCacheDimensionProvider conformance
    public var numLayers: Int {
        return configuration.hiddenLayers
    }

    public init(_ config: SopranoConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0..<config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SopranoModelInner(config)

        // Initialize decoder
        self.decoder = SopranoDecoder(
            numInputChannels: config.hiddenSize,
            decoderNumLayers: config.decoderNumLayers,
            decoderDim: config.decoderDim,
            decoderIntermediateDim: config.decoderIntermediateDim,
            hopLength: config.hopLength,
            nFft: config.nFft,
            upscale: config.upscale,
            dwKernel: config.dwKernel
        )

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var sampleRate: Int {
        return configuration.sampleRate
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)

        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    /// Forward pass that returns both logits and hidden states.
    func forwardWithHiddenStates(_ inputs: MLXArray, cache: [KVCache]? = nil) -> (logits: MLXArray, hiddenStates: MLXArray) {
        var h = model.embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        // Get hidden states before lm_head
        let hiddenStates = model.norm(h)

        // Compute logits
        let logits: MLXArray
        if let lmHead = lmHead {
            logits = lmHead(hiddenStates)
        } else {
            logits = model.embedTokens.asLinear(hiddenStates)
        }

        return (logits, hiddenStates)
    }

    public func makeCache() -> [KVCache] {
        return (0..<configuration.hiddenLayers).map { _ in
            KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Remove "model." prefix if present (this appears before language_model in some cases)
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }

            // Map language_model weights to our model structure
            // lm_head stays at the top level, other language_model weights go to model.*
            if newKey.hasPrefix("language_model.lm_head") {
                // lm_head is directly on SopranoModel, not inside model
                newKey = newKey.replacingOccurrences(of: "language_model.", with: "")
            } else if newKey.hasPrefix("language_model.") {
                // Other language_model weights go to model.* (SopranoModelInner)
                newKey = newKey.replacingOccurrences(of: "language_model.", with: "model.")
            }

            // Decoder weights should be float32
            var newValue = value
            if newKey.hasPrefix("decoder.") {
                newValue = value.asType(.float32)
            }

            sanitized[newKey] = newValue
        }

        // Remove lm_head if tying embeddings
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized
    }

    // MARK: - Text Preprocessing

    private func preprocessText(_ texts: [String], minLength: Int = 30) -> [(prompt: String, textIdx: Int, sentenceIdx: Int)] {
        var results: [(String, Int, Int)] = []

        for (textIdx, text) in texts.enumerated() {
            let trimmedText = text.trimmingCharacters(in: .whitespaces)
            let cleanedText = cleanTextForSoprano(trimmedText)

            // Create single prompt with the entire text (matching Python behavior)
            let prompt = "[STOP][TEXT]\(cleanedText)[START]"
            results.append((prompt, textIdx, 0))
        }

        return results
    }
    /// Space token ID in Soprano vocabulary
    private let spaceTokenId: Int = 8004

    private func tokenize(_ text: String) -> MLXArray {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not initialized")
        }

        // Swift-transformers doesn't correctly handle the pre-tokenizer "Split" with "Isolated" behavior
        // which should preserve space tokens. Work around by tokenizing each segment separately
        // and inserting space tokens between them.

        // Split the prompt into parts: special tokens and text
        // Pattern: [STOP], [TEXT], content, [START]
        var tokens: [Int] = []

        // Extract the text content from the prompt format "[STOP][TEXT]...[START]"
        if text.hasPrefix("[STOP][TEXT]") && text.hasSuffix("[START]") {
            // Add [STOP] token (id=3)
            tokens.append(3)
            // Add [TEXT] token (id=1)
            tokens.append(1)

            // Extract content between [TEXT] and [START]
            let startIdx = text.index(text.startIndex, offsetBy: 12) // After "[STOP][TEXT]"
            let endIdx = text.index(text.endIndex, offsetBy: -7) // Before "[START]"
            let content = String(text[startIdx..<endIdx])

            // Tokenize content word by word, inserting space tokens between
            let words = content.split(separator: " ", omittingEmptySubsequences: false)
            for (i, word) in words.enumerated() {
                if i > 0 {
                    // Insert space token between words
                    tokens.append(spaceTokenId)
                }
                // Tokenize the word (without special tokens)
                let wordTokens = tokenizer.encode(text: String(word))
                tokens.append(contentsOf: wordTokens)
            }

            // Add [START] token (id=2)
            tokens.append(2)
        } else {
            // Fallback to regular tokenization
            tokens = tokenizer.encode(text: text)
        }

        return MLXArray(tokens.map { Int32($0) })
    }

    // MARK: - Generation

    /// Generate audio from text.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - voice: Voice name (unused in base Soprano)
    ///   - parameters: Generation parameters
    /// - Returns: Generated audio as MLXArray
    public func generate(
        text: String,
        voice: String? = nil,
        splitPattern: String = "\n",  // Add split pattern parameter
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.5,
            repetitionContextSize: 30
        )
    ) async throws -> MLXArray {
        guard self.tokenizer != nil else {
            throw SopranoError.modelNotInitialized("Tokenizer not loaded")
        }

        // Process escape sequences and split by pattern
        let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        // Split text by pattern, then further split long chunks at sentence boundaries
        let prompts = prompt.components(separatedBy: splitPattern)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .flatMap { chunk -> [String] in
                guard chunk.count > 500 else { return [chunk] }

                // Split long chunks at sentence boundaries
                let boundaries = CharacterSet(charactersIn: ".?!:;")
                var result: [String] = []
                var current = ""

                for char in chunk {
                    current.append(char)
                    let atBoundary = char.unicodeScalars.first.map { boundaries.contains($0) } ?? false

                    if (atBoundary && current.count >= 100) || current.count >= 500 {
                        result.append(current.trimmingCharacters(in: .whitespaces))
                        current = ""
                    }
                }

                if !current.isEmpty {
                    result.append(current.trimmingCharacters(in: .whitespaces))
                }

                return result.filter { !$0.isEmpty }
            }

        var audioParts: [MLXArray] = []
        var totalTokens = 0
        let maxTokens = parameters.maxTokens ?? 512

        // Process each chunk separately
        for promptChunk in prompts {
            let sentenceData = self.preprocessText([promptChunk])
            print("================================================")
            print("Generate()")
            print("Sentence data: \(sentenceData)")
            print("================================================")

            for (promptText, _, _) in sentenceData {
                let inputIds = self.tokenize(promptText)
                var allHiddenStates: [MLXArray] = []

                for await (token, hiddenState) in self.streamGenerate(
                    inputIds: inputIds,
                    maxTokens: maxTokens,
                    temperature: parameters.temperature ?? 0.3,
                    topP: parameters.topP ?? 0.95,
                    repetitionPenalty: parameters.repetitionPenalty ?? 1.5,
                    repetitionContextSize: parameters.repetitionContextSize ?? 30
                ) {
                    allHiddenStates.append(hiddenState)

                    if token != nil {
                        totalTokens += 1
                    }
                }

                let tokenCount = allHiddenStates.count

                // Stack hidden states
                let hiddenStates = MLX.concatenated(allHiddenStates, axis: 1)

                // Decode to audio
                var audio = self.decoder(hiddenStates)

                let tokenSize = self.configuration.tokenSize
                let audioLength = tokenCount * tokenSize - tokenSize

                if audioLength > 0 {
                    audio = audio[0, (-audioLength)...]
                } else {
                    audio = audio.squeezed(axis: 0)
                }

                audioParts.append(audio)
            }
        }

        // Concatenate all audio parts
        let finalAudio: MLXArray
        if audioParts.count > 1 {
            finalAudio = MLX.concatenated(audioParts, axis: 0)
        } else if audioParts.count == 1 {
            finalAudio = audioParts[0]
        } else {
            throw SopranoError.generationFailed("No audio generated")
        }

        return finalAudio
    }

    /// Generate audio with streaming events.
    public func generateStream(
        text: String,
        voice: String? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 512,
            temperature: 0.3,
            topP: 0.95,
            repetitionPenalty: 1.5,
            repetitionContextSize: 30
        )
    ) -> AsyncThrowingStream<SopranoGeneration, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard self.tokenizer != nil else {
                        throw SopranoError.modelNotInitialized("Tokenizer not loaded")
                    }

                    let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
                        .replacingOccurrences(of: "\\t", with: "\t")

                    let startTime = Date()
                    let sentenceData = self.preprocessText([prompt])

                    print("================================================")
                    print("GenerateStream()")
                    print("Sentence data: \(sentenceData)")
                    print("================================================")

                    var audioParts: [MLXArray] = []
                    var totalTokens = 0
                    let maxTokens = parameters.maxTokens ?? 512

                    for (promptText, _, _) in sentenceData {
                        let inputIds = self.tokenize(promptText)
                        var allHiddenStates: [MLXArray] = []

                        for await (token, hiddenState) in self.streamGenerate(
                            inputIds: inputIds,
                            maxTokens: maxTokens,
                            temperature: parameters.temperature ?? 0.3,
                            topP: parameters.topP ?? 0.95,
                            repetitionPenalty: parameters.repetitionPenalty ?? 1.5,
                            repetitionContextSize: parameters.repetitionContextSize ?? 30
                        ) {
                            allHiddenStates.append(hiddenState)

                            print("Token: \(token)")

                            if let tokenVal = token {
                                continuation.yield(.token(tokenVal))
                            }
                        }

                        let tokenCount = allHiddenStates.count
                        totalTokens += tokenCount

                        // Stack hidden states
                        let hiddenStates = MLX.concatenated(allHiddenStates, axis: 1)

                        // Decode to audio
                        var audio = self.decoder(hiddenStates)

                        print("================================================")
                        print("GenerateStream()")
                        print("Audio shape: \(audio.shape)")
                        print("Token count: \(tokenCount)")
                        print("Token size: \(self.configuration.tokenSize)")

                        print("Hidden states shape: \(hiddenStates.shape)")
                        print("================================================")

                        let tokenSize = self.configuration.tokenSize
                        let audioLength = tokenCount * tokenSize - tokenSize

                        if audioLength > 0 {
                            audio = audio[0, (-audioLength)...]
                        } else {
                            audio = audio[0]
                        }

                        audioParts.append(audio)
                    }

                    // Concatenate audio
                    let finalAudio: MLXArray
                    if audioParts.count > 1 {
                        finalAudio = MLX.concatenated(audioParts, axis: 0)
                    } else {
                        finalAudio = audioParts[0]
                    }

                    let elapsed = Date().timeIntervalSince(startTime)

                    // Yield info
                    let info = SopranoGenerationInfo(
                        promptTokenCount: 0,
                        generationTokenCount: totalTokens,
                        prefillTime: 0,
                        generateTime: elapsed,
                        tokensPerSecond: Double(totalTokens) / elapsed
                    )
                    continuation.yield(.info(info))

                    // Yield audio
                    continuation.yield(.audio(finalAudio))

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Stream generate tokens and hidden states.
    private func streamGenerate(
        inputIds: MLXArray,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.5,
        repetitionContextSize: Int = 30
    ) -> AsyncStream<(Int?, MLXArray)> {
        AsyncStream { continuation in
            Task {
                var ids = inputIds
                if ids.ndim == 1 {
                    ids = ids.expandedDimensions(axis: 0)
                }

                // Create KV cache
                let cache = self.makeCache()

                // Prefill
                let (logits, hiddenStates) = self.forwardWithHiddenStates(ids, cache: cache)
                eval(logits, hiddenStates)

                // Yield last hidden state from prefill (last position along sequence dim)
                let lastHiddenState = hiddenStates[0..., (hiddenStates.shape[1] - 1)..<hiddenStates.shape[1], 0...]
                continuation.yield((nil, lastHiddenState))

                // Create sampler
                let sampler = TopPSampler(temperature: temperature, topP: topP)

                // Track generated tokens for repetition penalty
                var generatedTokens: [Int] = []

                // Generate tokens
                var currentLogits = logits

                for _ in 0..<maxTokens {
                    // Get last logits
                    var lastLogits = currentLogits[0..., -1, 0...]
                    eval(lastLogits)

                    // Apply repetition penalty
                    if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
                        let contextTokens = Array(generatedTokens.suffix(repetitionContextSize))
                        lastLogits = applyRepetitionPenalty(
                            logits: lastLogits,
                            tokens: contextTokens,
                            penalty: repetitionPenalty
                        )
                    }

                    // Sample next token
                    let nextToken: MLXArray
                    if temperature == 0 {
                        nextToken = argMax(lastLogits, axis: -1, keepDims: true)
                    } else {
                        nextToken = sampler.sample(logits: lastLogits)
                    }

                    let tokenId = nextToken.item(Int.self)

                    // Check for stop token
                    if let stopId = self.stopTokenId, tokenId == stopId {
                        break
                    }
                    if tokenId == self.configuration.eosTokenId {
                        break
                    }
                    if tokenId == self.configuration.padTokenId {
                        break
                    }

                    // Track token for repetition penalty
                    generatedTokens.append(tokenId)

                    // Forward pass with new token
                    let nextTokenExpanded = nextToken.reshaped([1, 1])
                    let (newLogits, newHiddenStates) = self.forwardWithHiddenStates(nextTokenExpanded, cache: cache)

                    let newLastHiddenState = newHiddenStates[0..., (newHiddenStates.shape[1] - 1)..<newHiddenStates.shape[1], 0...]
                    eval(newLastHiddenState)
                    continuation.yield((tokenId, newLastHiddenState))

                    currentLogits = newLogits
                }

                continuation.finish()
            }
        }
    }

    /// Apply repetition penalty to logits
    private func applyRepetitionPenalty(logits: MLXArray, tokens: [Int], penalty: Float) -> MLXArray {
        // Convert logits to array, apply penalty, convert back
        var logitsArray = logits.asArray(Float.self)
        for token in tokens {
            if token < logitsArray.count {
                if logitsArray[token] > 0 {
                    logitsArray[token] /= penalty
                } else {
                    logitsArray[token] *= penalty
                }
            }
        }
        return MLXArray(logitsArray)
    }

    // MARK: - Loading

    public static func fromPretrained(_ modelRepo: String) async throws -> SopranoModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            client = HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            client = HubClient.default
        }
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "SopranoModel", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"
            ])
        }

        let modelDir = try await resolveOrDownloadSopranoModel(
            client: client,
            cache: cache,
            repoID: repoID
        )

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(SopranoConfiguration.self, from: configData)

        let model = SopranoModel(config)

        // Load weights
        let weights = try loadSopranoWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Apply quantization if needed
        if let perLayerQuant = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                if weights["\(path).scales"] != nil {
                    return perLayerQuant.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])

        eval(model)

        // Load tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // Set stop token ID
        if let tokenizer = model.tokenizer {
            let stopTokens = tokenizer.encode(text: "[STOP]")
            if !stopTokens.isEmpty {
                model.stopTokenId = stopTokens[0]
            }
        }

        return model
    }
}

// MARK: - Helper Functions

private func loadSopranoWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

private func resolveOrDownloadSopranoModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID
) async throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasWeights = files?.contains { $0.pathExtension == "safetensors" } ?? false

        if hasWeights {
            let configPath = modelDir.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configPath.path),
               let configData = try? Data(contentsOf: configPath),
               (try? JSONSerialization.jsonObject(with: configData)) != nil {
                return modelDir
            }
        }
    }

    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDir,
        revision: "main",
        progressHandler: { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }
    )

    return modelDir
}

// MARK: - TopP Sampler

private struct TopPSampler {
    let temperature: Float
    let topP: Float

    func sample(logits: MLXArray) -> MLXArray {
        var probs = softmax(logits / temperature, axis: -1)

        // Sort probabilities descending
        let sortedIndices = argSort(probs, axis: -1)
        let sortedProbs = take(probs, sortedIndices, axis: -1)

        // Compute cumulative probabilities
        let cumProbs = cumsum(sortedProbs, axis: -1)

        // Find cutoff
        let cutoffMask = cumProbs .> (1 - topP)

        // Zero out probabilities below cutoff
        probs = MLX.where(cutoffMask, probs, MLXArray(Float(0)))

        // Renormalize
        let probSum = sum(probs, axis: -1, keepDims: true)
        probs = probs / maximum(probSum, MLXArray(Float(1e-8)))

        // Sample from distribution
        return categorical(probs)
    }
}
