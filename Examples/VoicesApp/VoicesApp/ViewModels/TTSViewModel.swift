import Foundation
import SwiftUI
import MLXAudioTTS
import MLXAudioCore
import MLX
import AVFoundation
import Combine

@MainActor
@Observable
class TTSViewModel {
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var tokensPerSecond: Double = 0

    // Generation parameters
    var maxTokens: Int = 1200
    var temperature: Float = 0.6
    var topP: Float = 0.8

    // Text chunking
    var enableChunking: Bool = true
    var maxChunkLength: Int = 200
    var splitPattern: String = "\n"  // Can be regex like "\\n" or "[.!?]\\s+"

    // Streaming playback
    var streamingPlayback: Bool = true  // Play audio as chunks are generated

    // Model configuration
    var modelId: String = "mlx-community/VyvoTTS-EN-Beta-4bit"
    private var loadedModelId: String?

    // Audio player state (manually synced from AudioPlayerManager)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: Qwen3Model?
    private let audioPlayer = AudioPlayerManager()
    private var cancellables = Set<AnyCancellable>()

    var isModelLoaded: Bool {
        model != nil
    }

    init() {
        setupAudioPlayerObservers()
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.currentTime = value
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.duration = value
            }
            .store(in: &cancellables)
    }

    func loadModel() async {
        // Skip if same model already loaded
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await Qwen3Model.fromPretrained(modelId)
            loadedModelId = modelId
            generationProgress = ""  // Clear progress on success
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    func reloadModel() async {
        // Unload current model and clear GPU memory
        model = nil
        loadedModelId = nil
        Memory.clearCache()

        await loadModel()
    }

    /// Split text into chunks based on pattern and max length
    private func chunkText(_ text: String) -> [String] {
        guard enableChunking && text.count > maxChunkLength else {
            return [text]
        }

        // First split by pattern (supports regex)
        var segments: [String]
        if let regex = try? NSRegularExpression(pattern: splitPattern, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            segments = regex.stringByReplacingMatches(in: text, range: range, withTemplate: "\u{0000}")
                .components(separatedBy: "\u{0000}")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        } else {
            // Fallback to simple string split
            segments = text.components(separatedBy: splitPattern)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }

        // Group segments into chunks respecting max length
        var chunks: [String] = []
        var currentChunk = ""

        for segment in segments {
            if currentChunk.isEmpty {
                currentChunk = segment
            } else if currentChunk.count + segment.count + 1 <= maxChunkLength {
                currentChunk += " " + segment
            } else {
                chunks.append(currentChunk)
                currentChunk = segment
            }
        }

        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }

        // Handle case where a single segment is too long - split by sentence boundaries
        var finalChunks: [String] = []
        for chunk in chunks {
            if chunk.count > maxChunkLength {
                // Try splitting by sentence boundaries
                let sentencePattern = "[.!?]+\\s*"
                if let sentenceRegex = try? NSRegularExpression(pattern: sentencePattern, options: []) {
                    let range = NSRange(chunk.startIndex..., in: chunk)
                    let sentences = sentenceRegex.stringByReplacingMatches(in: chunk, range: range, withTemplate: "$0\u{0000}")
                        .components(separatedBy: "\u{0000}")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }

                    var subChunk = ""
                    for sentence in sentences {
                        if subChunk.isEmpty {
                            subChunk = sentence
                        } else if subChunk.count + sentence.count + 1 <= maxChunkLength {
                            subChunk += " " + sentence
                        } else {
                            finalChunks.append(subChunk)
                            subChunk = sentence
                        }
                    }
                    if !subChunk.isEmpty {
                        finalChunks.append(subChunk)
                    }
                } else {
                    finalChunks.append(chunk)
                }
            } else {
                finalChunks.append(chunk)
            }
        }

        return finalChunks.isEmpty ? [text] : finalChunks
    }

    func synthesize(text: String, voice: Voice? = nil) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        guard !text.isEmpty else {
            errorMessage = "Please enter text to synthesize"
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = "Starting generation..."
        tokensPerSecond = 0

        do {
            // Split text into chunks
            let chunks = chunkText(text)
            var audioSamples: [Float] = []  // Store on CPU for saving to file
            var totalTokenCount = 0
            let sampleRate = Double(model.sampleRate)

            // Start streaming playback if enabled and we have multiple chunks
            let useStreaming = streamingPlayback && chunks.count > 1
            if useStreaming {
                audioPlayer.startStreaming(sampleRate: sampleRate)
            }

            for (index, chunk) in chunks.enumerated() {
                if chunks.count > 1 {
                    generationProgress = "Processing chunk \(index + 1)/\(chunks.count)..."
                }

                var chunkTokenCount = 0
                var audio: MLXArray?

                // Set aggressive cache limit to reduce peak memory during generation
                Memory.cacheLimit = 1024 * 1024 * 1024  // 1GB cache limit

                print("GPU memory before generation - Active: \(Double(Memory.activeMemory)/1e9) GB")

                // Each chunk needs a fresh cache - don't reuse across chunks
                for try await event in model.generateStream(
                    text: chunk,
                    voice: voice?.name,
                    cache: nil,
                    parameters: .init(
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        repetitionPenalty: 1.3,
                        repetitionContextSize: 20
                    )
                ) {
                    switch event {
                    case .token:
                        chunkTokenCount += 1
                        totalTokenCount += 1
                        if chunkTokenCount % 10 == 0 {
                            if chunks.count > 1 {
                                generationProgress = "Chunk \(index + 1)/\(chunks.count): \(chunkTokenCount) tokens..."
                            } else {
                                generationProgress = "Generated \(chunkTokenCount) tokens..."
                            }
                        }
                    case .info(let info):
                        tokensPerSecond = info.tokensPerSecond
                    case .audio(let audioData):
                        audio = audioData
                    }
                }

                // Convert to CPU samples immediately to free GPU memory
                if let audioData = audio {
                    let samples = audioData.asArray(Float.self)

                    // Stream playback immediately as chunks are ready
                    if useStreaming {
                        audioPlayer.scheduleAudioChunk(samples, withCrossfade: true)
                    }

                    // Also accumulate for saving to file
                    audioSamples.append(contentsOf: samples)
                }
                audio = nil

                // Clear GPU cache after each chunk
                Memory.clearCache()
                print("GPU memory after generation - Active: \(Double(Memory.activeMemory)/1e9) GB, Peak: \(Double(Memory.peakMemory)/1e9) GB")

            }

            guard !audioSamples.isEmpty else {
                throw NSError(
                    domain: "TTSViewModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No audio generated"]
                )
            }

            // Save audio directly from CPU samples (avoids GPU memory allocation)
            generationProgress = "Saving audio..."

            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")

            try saveAudioSamples(audioSamples, sampleRate: sampleRate, to: tempURL)
            Memory.clearCache()

            audioURL = tempURL
            generationProgress = ""  // Clear progress

            // For single chunk, load normally for playback
            if !useStreaming {
                audioPlayer.loadAudio(from: tempURL)
            }

        } catch {
            errorMessage = "Generation failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    func play() {
        audioPlayer.play()
    }

    func pause() {
        audioPlayer.pause()
    }

    func togglePlayPause() {
        audioPlayer.togglePlayPause()
    }

    func stop() {
        audioPlayer.stop()
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }
}
