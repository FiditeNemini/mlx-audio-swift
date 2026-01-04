//
//  VocosBackbone.swift
//  MLXAudio
//
//  Vocos-based backbone for Soprano decoder.
//  Ported from mlx_audio/codec/models/vocos/vocos.py
//

import Foundation
import MLX
import MLXNN

// MARK: - ConvNeXt Block

/// ConvNeXt block for the Vocos backbone.
///
/// Uses depthwise convolution followed by pointwise convolutions with GELU activation.
public class ConvNeXtBlock: Module {
    let dwconv: Conv1d
    let norm: LayerNorm
    let pwconv1: Linear
    let pwconv2: Linear
    let gamma: MLXArray?

    public init(
        dim: Int,
        intermediateDim: Int,
        layerScaleInitValue: Float = 0.125,
        dwKernelSize: Int = 7
    ) {
        // Depthwise convolution with groups=dim
        self.dwconv = Conv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: dwKernelSize,
            padding: dwKernelSize / 2,
            groups: dim
        )

        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)

        // Pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = Linear(dim, intermediateDim)
        self.pwconv2 = Linear(intermediateDim, dim)

        // Layer scale parameter
        if layerScaleInitValue > 0 {
            self.gamma = layerScaleInitValue * MLXArray.ones([dim])
        } else {
            self.gamma = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x

        // Depthwise conv
        var h = dwconv(x)

        // Layer norm
        h = norm(h)

        // Pointwise convs with GELU
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)

        // Layer scale
        if let gamma = gamma {
            h = gamma * h
        }

        // Residual connection
        return residual + h
    }
}

// MARK: - Vocos Backbone

/// Vocos backbone using ConvNeXt blocks.
///
/// Processes input features through an embedding conv layer followed by
/// a stack of ConvNeXt blocks.
public class VocosBackbone: Module {
    let inputChannels: Int
    @ModuleInfo(key: "embed") var embed: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "convnext") var convnext: [ConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(
        inputChannels: Int,
        dim: Int,
        intermediateDim: Int,
        numLayers: Int,
        layerScaleInitValue: Float? = nil,
        inputKernelSize: Int = 7,
        dwKernelSize: Int = 7
    ) {
        self.inputChannels = inputChannels

        // Embedding convolution
        self._embed.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: dim,
            kernelSize: inputKernelSize,
            padding: inputKernelSize / 2
        )

        // Initial layer norm
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)

        // Calculate layer scale init value
        let scaleValue = layerScaleInitValue ?? (1.0 / Float(numLayers))

        // Stack of ConvNeXt blocks
        self._convnext.wrappedValue = (0..<numLayers).map { _ in
            ConvNeXtBlock(
                dim: dim,
                intermediateDim: intermediateDim,
                layerScaleInitValue: scaleValue,
                dwKernelSize: dwKernelSize
            )
        }

        // Final layer norm
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x

        // Transpose if input is not in (B, L, C) format
        // Input should be (B, L, C) where C is input_channels
        if h.shape.last != inputChannels {
            h = h.transposed(0, 2, 1)
        }

        // Embedding conv
        h = embed(h)

        // Initial norm
        h = norm(h)

        // ConvNeXt blocks
        for block in convnext {
            h = block(h)
        }

        // Final norm
        h = finalLayerNorm(h)

        return h
    }
}
