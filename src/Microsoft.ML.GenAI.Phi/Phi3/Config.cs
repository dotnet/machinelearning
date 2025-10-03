// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3Config
{
    public Phi3Config()
    {
        this.VocabSize = 32064;
        this.HiddenSize = 3072;
        this.RmsNormEps = 1e-5f;
        this.IntermediateSize = 8192;
        this.NumHiddenLayers = 32;
        this.NumAttentionHeads = 32;
        this.ResidPdrop = 0.0;
        this.EmbdPdrop = 0.0;
        this.AttentionDropout = 0.0;
        this.HiddenAct = "silu";
        this.MaxPositionEmbeddings = 4096;
        this.OriginalMaxPositionEmbeddings = 4096;
        this.InitializerRange = 0.02;
        this.UseCache = true;
        this.TieWordEmbeddings = false;
        this.RopeTheta = 10000.0;
        this.PartialRotaryFactor = 0.5;
        this.QkLayernorm = false;
        this.BosTokenId = 1;
        this.EosTokenId = 32000;
        this.DType = ScalarType.BFloat16;
        this.AttnImplementation = "eager";
    }

    static Phi3Config()
    {
        var phi3Mini4kInstructContent = Core.Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Phi.Resource.Config.phi-3-mini-4k-instruct-config.json");
        var phi3Mini128kInstructContent = Core.Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Phi.Resource.Config.phi-3-mini-128k-instruct-config.json");
        var phi3Medium4kInstructContent = Core.Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Phi.Resource.Config.phi-3-medium-4k-instruct-config.json");
        var phi3Medium128kInstructContent = Core.Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Phi.Resource.Config.phi-3-medium-128k-instruct-config.json");

        Phi3Mini4kInstruct = JsonSerializer.Deserialize<Phi3Config>(phi3Mini4kInstructContent) ?? throw new ArgumentNullException(nameof(phi3Mini4kInstructContent));
        Phi3Mini128kInstruct = JsonSerializer.Deserialize<Phi3Config>(phi3Mini128kInstructContent) ?? throw new ArgumentNullException(nameof(phi3Mini128kInstructContent));
        Phi3Medium4kInstruct = JsonSerializer.Deserialize<Phi3Config>(phi3Medium4kInstructContent) ?? throw new ArgumentNullException(nameof(phi3Medium4kInstructContent));
        Phi3Medium128kInstruct = JsonSerializer.Deserialize<Phi3Config>(phi3Medium128kInstructContent) ?? throw new ArgumentNullException(nameof(phi3Medium128kInstructContent));
    }

    /// <summary>
    /// The phi-3-mini-4k-instruct configuration created from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json.
    /// </summary>
    public static Phi3Config Phi3Mini4kInstruct { get; }

    /// <summary>
    /// The phi-3-medium-4k-instruct configuration created from https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json.
    /// </summary>
    public static Phi3Config Phi3Medium4kInstruct { get; }

    /// <summary>
    /// The phi-3-medium-128k-instruct configuration created from https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/blob/main/config.json.
    /// </summary>
    public static Phi3Config Phi3Medium128kInstruct { get; }

    /// <summary>
    /// The phi-3-mini-128k-instruct configuration created from https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json.
    /// </summary>
    public static Phi3Config Phi3Mini128kInstruct { get; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; }

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; }

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; }

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; }

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; }

    [JsonPropertyName("num_key_value_heads")]
    public int? NumKeyValueHeads { get; set; }

    [JsonPropertyName("resid_pdrop")]
    public double ResidPdrop { get; set; }

    [JsonPropertyName("embd_pdrop")]
    public double EmbdPdrop { get; set; }

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; }

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; }

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; }

    [JsonPropertyName("original_max_position_embeddings")]
    public int OriginalMaxPositionEmbeddings { get; set; }

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; }

    [JsonPropertyName("use_cache")]
    public bool UseCache { get; set; }

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; }

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; }

    [JsonPropertyName("rope_scaling")]
    public Dictionary<string, object>? RopeScaling { get; set; }

    [JsonPropertyName("partial_rotary_factor")]
    public double PartialRotaryFactor { get; set; }

    [JsonPropertyName("qk_layernorm")]
    public bool QkLayernorm { get; set; }

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; }

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; }

    [JsonPropertyName("pad_token_id")]
    public int? PadTokenId { get; set; }

    [JsonPropertyName("sliding_window")]
    public int? SlidingWindow { get; set; }

    public ScalarType DType { get; set; }

    public string AttnImplementation { get; set; }
}
