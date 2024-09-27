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

public class Phi2Config
{
    public Phi2Config()
    {
        this.VocabSize = 51200;
        this.HiddenSize = 2048;
        this.IntermediateSize = 8192;
        this.NumHiddenLayers = 24;
        this.NumAttentionHeads = 32;
        this.ResidPdrop = 0.0;
        this.EmbdPdrop = 0.0;
        this.AttentionDropout = 0.0;
        this.HiddenAct = "gelu_new";
        this.MaxPositionEmbeddings = 2048;
        this.InitializerRange = 0.02;
        this.LayerNormEps = 1e-5;
        this.UseCache = true;
        this.TieWordEmbeddings = false;
        this.RopeTheta = 10000.0;
        this.PartialRotaryFactor = 0.5;
        this.QkLayernorm = false;
        this.BosTokenId = 1;
        this.EosTokenId = 2;
        this.Dtype = ScalarType.Float32;
    }

    static Phi2Config()
    {
        var phi2ConfigContent = Core.Utils.GetEmbeddedResource("Microsoft.ML.GenAI.Phi.Resource.Config.phi-2-config.json");
        var phi2Config = JsonSerializer.Deserialize<Phi2Config>(phi2ConfigContent) ?? throw new ArgumentNullException(nameof(phi2ConfigContent));
        Phi2 = phi2Config;
    }

    /// <summary>
    /// The default phi-2 configuration created from https://huggingface.co/microsoft/phi-2/blob/main/config.json.
    /// </summary>
    public static Phi2Config Phi2 { get; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; }

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

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; }

    [JsonPropertyName("layer_norm_eps")]
    public double LayerNormEps { get; set; }

    [JsonPropertyName("use_cache")]
    public bool UseCache { get; set; }

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; }

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; }

    // [JsonPropertyName("rope_scaling")]
    // public double? RopeScaling { get; set; } = null;

    [JsonPropertyName("partial_rotary_factor")]
    public double PartialRotaryFactor { get; set; }

    [JsonPropertyName("qk_layernorm")]
    public bool QkLayernorm { get; set; }

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; }

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; }

    public ScalarType Dtype { get; set; }
}
