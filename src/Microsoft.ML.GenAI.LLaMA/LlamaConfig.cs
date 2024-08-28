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
using Microsoft.ML.GenAI.Core;
using TorchSharp;

namespace Microsoft.ML.GenAI.LLaMA;

public class LlamaConfig
{
    public LlamaConfig()
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
        this.PretrainingTp = 1;
        this.RmsNormEps = 1e-05f;
        this.RopeScaling = new RopeScalingConfig();
        this.RopeTheta = 500000.0;
        this.TieWordEmbeddings = false;
        this.VocabSize = 128256;
        this.AttnImplementation = "eager";
        this.DType = torch.ScalarType.BFloat16;
    }

    static LlamaConfig()
    {
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
        var llama3_1_8b_content = Utils.GetEmbeddedResource("Microsoft.ML.GenAI.LLaMA.Resource.Config.meta-llama-3.1-8B-Instruct.json");
        var llama3_1_70b_content = Utils.GetEmbeddedResource("Microsoft.ML.GenAI.LLaMA.Resource.Config.meta-llama-3.1-70B-Instruct.json");
        var llama3_1_405b_content = Utils.GetEmbeddedResource("Microsoft.ML.GenAI.LLaMA.Resource.Config.meta-llama-3.1-405B-Instruct.json");
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard

        Llama3_1_8B_Instruct = JsonSerializer.Deserialize<LlamaConfig>(llama3_1_8b_content) ?? throw new ArgumentNullException(nameof(llama3_1_8b_content));
        Llama3_1_70B_Instruct = JsonSerializer.Deserialize<LlamaConfig>(llama3_1_70b_content) ?? throw new ArgumentNullException(nameof(llama3_1_70b_content));
        Llama3_1_405B_Instruct = JsonSerializer.Deserialize<LlamaConfig>(llama3_1_405b_content) ?? throw new ArgumentNullException(nameof(llama3_1_405b_content));
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    /// <summary>
    /// The llama-3.1-8B-Instruct configuration created from https://huggingface.co/meta-llama/Meta-Llama-3.1-8B.
    /// </summary>
    public static LlamaConfig Llama3_1_8B_Instruct { get; }

    /// <summary>
    /// The llama-3.1-70B-Instruct configuration created from https://huggingface.co/meta-llama/Meta-Llama-3.1-70B.
    /// </summary>
    public static LlamaConfig Llama3_1_70B_Instruct { get; }

    /// <summary>
    /// The llama-3.1-405B-Instruct configuration created from https://huggingface.co/meta-llama/Meta-Llama-3.1-405B.
    /// </summary>
    public static LlamaConfig Llama3_1_405B_Instruct { get; }
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

    [JsonPropertyName("pretraining_tp")]
    public int PretrainingTp { get; set; }

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; }

    public RopeScalingConfig RopeScaling { get; set; }

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; }

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }
    public int? PadTokenId { get; set; }
    public torch.ScalarType DType { get; set; }
    public string AttnImplementation { get; set; }
}
