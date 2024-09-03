// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.GenAI.Core;
using TorchSharp;

namespace Microsoft.ML.GenAI.Mistral;

public class MistralConfig
{
    public MistralConfig()
    {
        this.AttentionBias = false;
        this.AttentionDropout = 0.0;
        this.HiddenAct = "silu";
        this.HiddenSize = 4096;
        this.InitializerRange = 0.02;
        this.IntermediateSize = 14336;
        this.MaxPositionEmbeddings = 131072;
        this.MlpBias = false;
        this.NumAttentionHeads = 32;
        this.NumHiddenLayers = 32;
        this.NumKeyValueHeads = 8;
        this.RmsNormEps = 1e-05f;
        this.RopeScaling = new RopeScalingConfig();
        this.RopeTheta = 500000.0;
        this.TieWordEmbeddings = false;
        this.VocabSize = 128256;
        this.AttnImplementation = "eager";
        this.DType = torch.ScalarType.BFloat16;
        this.HeadDim = this.HiddenSize / this.NumAttentionHeads;
        this.SlidingWindow ??= 4096;
    }

    static MistralConfig()
    {
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
        var mistral7BInstructContent = Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Mistral.Resource.Config.mistral-7B-instruct-v0.3.json");
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard

        Mistral_7B_Instruct_v0_3 = JsonSerializer.Deserialize<MistralConfig>(mistral7BInstructContent) ?? throw new ArgumentNullException(nameof(mistral7BInstructContent));
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    /// <summary>
    /// The mistral-7b-instruct-v0.3 configuration created from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/main.
    /// </summary>
    public static MistralConfig Mistral_7B_Instruct_v0_3 { get; }
#pragma warning restore MSML_GeneralName // This name should be PascalCased

    [JsonPropertyName("attention_bias")]
    public bool AttentionBias { get; set; }

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; }

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; }

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; }

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; }

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; }

    [JsonPropertyName("mlp_bias")]
    public bool MlpBias { get; set; }

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; }

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; }

    [JsonPropertyName("num_key_value_heads")]
    public int NumKeyValueHeads { get; set; }

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; }

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; }

    public RopeScalingConfig RopeScaling { get; set; }

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; }

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }

    [JsonPropertyName("sliding_window")]
    public int? SlidingWindow { get; set; }

    public int? PadTokenId { get; set; }

    public torch.ScalarType DType { get; set; }

    public string AttnImplementation { get; set; }
}
