import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Text("Settings")
                .font(.title2)
                .fontWeight(.semibold)
                .padding(.top, 20)
                .padding(.bottom, 24)
                .frame(maxWidth: .infinity)

            // Model Section
            VStack(alignment: .leading, spacing: 4) {
                Text("Model")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 8))

                    Button(action: {
                        Task {
                            await viewModel.reloadModel()
                        }
                    }) {
                        Text("Load")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(Color.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isLoading)
                }
                .padding(.top, 8)

                Text("Hugging Face model ID (e.g., mlx-community/VyvoTTS-EN-Beta-4bit)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 16)

            // Length Section
            VStack(alignment: .leading, spacing: 4) {
                Text("Length")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max Tokens")
                    Spacer()
                    Text("\(viewModel.maxTokens)")
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 8)

                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 100...2000,
                    step: 100
                )
                .tint(.blue)

                Text("Controls the maximum length of generated audio. Higher values allow longer speech.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 16)

            // Temperature Section
            VStack(alignment: .leading, spacing: 4) {
                Text("Temperature")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", viewModel.temperature))
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 8)

                Slider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)

                Text("Controls randomness. Lower values are more deterministic, higher values are more creative.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 16)

            // Top P Section
            VStack(alignment: .leading, spacing: 4) {
                Text("Top P (Nucleus Sampling)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Top P")
                    Spacer()
                    Text(String(format: "%.2f", viewModel.topP))
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 8)

                Slider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)

                Text("Nucleus sampling threshold. Lower values focus on more likely tokens.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 16)

            // Text Chunking Section
            VStack(alignment: .leading, spacing: 4) {
                Text("Text Chunking")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Toggle("Enable chunking for long text", isOn: $viewModel.enableChunking)
                    .padding(.top, 8)

                if viewModel.enableChunking {
                    Toggle("Stream audio", isOn: $viewModel.streamingPlayback)
                        .padding(.top, 4)


                    HStack {
                        Text("Max chunk length")
                        Spacer()
                        Text("\(viewModel.maxChunkLength)")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)

                    Slider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        in: 100...500,
                        step: 50
                    )
                    .tint(.blue)

                    HStack {
                        Text("Split pattern")
                        Spacer()
                    }
                    .padding(.top, 8)

                    TextField("Pattern (regex)", text: $viewModel.splitPattern)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 8))

                    Text("Split on this pattern first, then by sentences. Examples: \\n (newline), [.!?]\\s+ (sentences)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Bottom buttons
            HStack {
                Button(action: {
                    viewModel.modelId = "mlx-community/VyvoTTS-EN-Beta-4bit"
                    viewModel.maxTokens = 1200
                    viewModel.temperature = 0.6
                    viewModel.topP = 0.8
                    viewModel.enableChunking = true
                    viewModel.maxChunkLength = 200
                    viewModel.splitPattern = "\n"
                    viewModel.streamingPlayback = true
                }) {
                    Text("Reset to Defaults")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.blue.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)

                Spacer()

                Button(action: { dismiss() }) {
                    Text("Done")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 10)
                        .background(Color.blue)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
            }
            .padding(.bottom, 20)
        }
        .padding(.horizontal, 24)
        .frame(minWidth: 450, minHeight: 700)
    }
}

#Preview {
    SettingsView(viewModel: TTSViewModel())
}
