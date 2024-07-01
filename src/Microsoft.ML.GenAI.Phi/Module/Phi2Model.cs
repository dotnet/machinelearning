// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.Contracts;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi2Model : nn.Module<
    Tensor, // input_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    Tensor?, // position_ids
    Tensor?, //input embeddings
    (
        bool, // use_cache
        bool, // output_attentions
        bool // output_hidden_states
    ),
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly Phi2Config _config;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Embedding embed_tokens;
    private readonly Dropout embed_dropout;
    private readonly LayerNorm final_layernorm;
    private readonly ModuleList<Phi2DecoderLayer> layers;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi2Model(Phi2Config config)
        : base(nameof(Phi2Model))
    {
        this._config = config;
        this.embed_tokens = nn.Embedding(config.VocabSize, config.HiddenSize, dtype: config.Dtype);
        this.embed_dropout = nn.Dropout(config.EmbdPdrop);
        this.final_layernorm = nn.LayerNorm(config.HiddenSize, eps: config.LayerNormEps, dtype: config.Dtype);
        this.layers = new ModuleList<Phi2DecoderLayer>(Enumerable.Range(0, config.NumHiddenLayers).Select(i => new Phi2DecoderLayer(config)).ToArray());
        this.RegisterComponents();
    }

    public Phi2Config Config => this._config;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override (Tensor, Tensor?, Tensor?) forward(
#pragma warning restore MSML_GeneralName // This name should be PascalCased
        Tensor inputIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        Tensor? positionIds = null,
        Tensor? inputEmbeddings = null,
        (bool, bool, bool) options = default) // use_cache, output_attentions, output_hidden_states
    {
        (var outputAttentions, var outputHiddenStates, var useCache) = options;

        // TODO
        // add support for inputEmbeddings
        if (inputEmbeddings is not null)
        {
            throw new NotImplementedException("inputEmbeddings is not supported");
        }
        inputEmbeddings = this.embed_tokens.forward(inputIds);
        inputEmbeddings = this.embed_dropout.forward(inputEmbeddings);
        var batchSize = inputIds.shape[0];
        var seqLen = (int)inputIds.shape[1];

        if (positionIds is null)
        {
            positionIds = torch.arange(pastKeyValueLength, seqLen + pastKeyValueLength, dtype: inputIds.dtype, device: inputIds.device);
            positionIds = positionIds.unsqueeze(0);
        }

        // attention
        // use 4d attention mask
        if (attentionMask is not null)
        {
            attentionMask = this.Prepare4DCasualAttentionMask(attentionMask, seqLen, pastKeyValueLength, inputEmbeddings.dtype);
        }

        var hiddenStates = inputEmbeddings;

        for (int i = 0; i < this.layers.Count; i++)
        {
            (hiddenStates, _, _) = this.layers[i].forward(
                hiddenStates: hiddenStates,
                positionIds: positionIds,
                attentionMask: attentionMask,
                pastKeyValueLength: pastKeyValueLength,
                useCache: useCache,
                outputAttentions: outputAttentions);
        }

        hiddenStates = this.final_layernorm.forward(hiddenStates);
        return (hiddenStates, null, null);
    }

    private Tensor Prepare4DCasualAttentionMask(
        Tensor attentionMask,
        int queryLength,
        int pastKeyValueLength,
        ScalarType dtype)
    {
        var batchSize = (int)attentionMask.shape[0];
        var seqLen = attentionMask.shape[1];
        Contract.Assert(seqLen == queryLength, "seqLen must be equal to queryLength");
        var targetLength = queryLength + pastKeyValueLength;
        var casual4DMask = this.MakeCasualAttentionMask(batchSize, queryLength, pastKeyValueLength, attentionMask.device, dtype);
        var expandedMask = this.ExpandMask(attentionMask, dtype, queryLength).to(attentionMask.device);

        casual4DMask.masked_fill_(expandedMask.to_type(ScalarType.Bool), torch.finfo(dtype).min);
        return casual4DMask;
    }

    private Tensor ExpandMask(
        Tensor mask,
        ScalarType dtype,
        int targetLength)
    {
        var batch = mask.shape[0];
        var seqLen = mask.shape[1];
        var expandedMask = mask.unsqueeze(1).unsqueeze(2);
        expandedMask = expandedMask.expand(new long[] { batch, 1, targetLength, seqLen });
        expandedMask = expandedMask.to_type(dtype);

        var invertedMask = (1.0f - expandedMask) > 0;

        return invertedMask.masked_fill(invertedMask.to_type(ScalarType.Bool), torch.finfo(dtype).min);
    }
    private Tensor MakeCasualAttentionMask(
        int batchSize,
        int targetLen,
        int pastKeyValueLength,
        Device device,
        ScalarType dtype)
    {
        var mask = torch.full([targetLen, targetLen], torch.finfo(dtype).min, dtype: dtype, device: device);
        var maskCond = torch.arange(mask.size(-1), device: device);
        mask.masked_fill_(maskCond < (maskCond + 1).view(mask.size(-1), 1), 0.0f);

        mask = mask.to_type(dtype);

        if (pastKeyValueLength > 0)
        {
            mask = torch.cat([torch.zeros([targetLen, pastKeyValueLength], dtype: dtype, device: device), mask], dim: -1);
        }

        mask = mask.unsqueeze(0).unsqueeze(0);
        mask = mask.expand(new long[] { batchSize, 1, targetLen, targetLen + pastKeyValueLength });

        return mask;
    }
}
