// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.GenAI.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Mistral.Module;

internal class MistralModel : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly MistralConfig _config;
    private readonly int? _paddingIdx;
    private readonly int _vocabSize;
    private IKVCache _cache;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Embedding embed_tokens;
    private readonly ModuleList<MistralDecoderLayer> layers;
    private readonly RMSNorm norm;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly nn.Module<RotaryEmbeddingInput, RotaryEmbeddingOutput> _rotaryEmb;


    public MistralModel(MistralConfig config)
        : base(nameof(MistralModel))
    {
        this._config = config;
        this._paddingIdx = config.PadTokenId;
        this._vocabSize = config.VocabSize;
        var headDim = config.HeadDim;
        this.embed_tokens = nn.Embedding(config.VocabSize, config.HiddenSize, padding_idx: this._paddingIdx, dtype: config.DType);
        this.layers = new ModuleList<MistralDecoderLayer>();

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            this.layers.Add(new MistralDecoderLayer(config, i));
        }
        this.norm = new RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);
        this._cache = new DynamicKVCache();
        this.RegisterComponents();
        this._rotaryEmb = config.RopeScaling switch
        {
            null => new RotaryEmbedding(config.RopeTheta, config.MaxPositionEmbeddings, headDim),
            _ => new RotaryEmbedding(config.RopeTheta, headDim, config.RopeScaling),
        };
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override CausalLMModelOutput forward(CausalLMModelInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (input.OverrideCache is not null)
        {
            this._cache = input.OverrideCache;
        }

        var outputAttentions = input.OutputAttentions;
        var outputHiddenStates = input.OutputHiddenStates;
        var attentionMask = input.AttentionMask;
        Device device;
        var inputIds = input.InputIds;
        var positionIds = input.PositionIds;
        var inputsEmbeds = input.InputEmbeddings;
        int batchSize;
        int seqLength;
        if (inputIds is not null && inputsEmbeds is not null)
        {
            throw new ArgumentException("Only one of input_ids or inputs_embeds may be set");
        }
        else if (inputIds is not null)
        {
            batchSize = inputIds.IntShape()[0];
            seqLength = inputIds.IntShape()[1];
            inputsEmbeds = this.embed_tokens.forward(inputIds);
            device = inputIds.device;
        }
        else if (inputsEmbeds is not null)
        {
            batchSize = inputsEmbeds.IntShape()[0];
            seqLength = inputsEmbeds.IntShape()[1];
            device = inputsEmbeds.device;
        }
        else
        {
            throw new ArgumentException("Either input_ids or inputs_embeds must be set");
        }

        var pastKeyValuesLength = input.PastKeyValuesLength;

        if (positionIds is null)
        {
            positionIds = torch.arange(pastKeyValuesLength, seqLength + pastKeyValuesLength, device: device);
            positionIds = positionIds.unsqueeze(0).view(-1, seqLength);
        }
        else
        {
            positionIds = ((long)positionIds.view(-1, seqLength));
        }

        if (this._config.AttnImplementation == "flash_attention_2")
        {
            throw new NotImplementedException();
        }
        else
        {
            // the following behavior of creating 4d causal mask doesn't match python's, remember to look into it when there's time.
            attentionMask = AttentionMaskConverter.Create4DCausalAttentionMask(attentionMask, [batchSize, seqLength], inputsEmbeds.dtype, device, pastKeyValuesLength, slidingWindow: _config.SlidingWindow);
        }

        var hiddenStates = inputsEmbeds;

        var allHiddenStates = new List<Tensor>();
        var allAttentions = new List<Tensor>();

        var embOutput = this._rotaryEmb.forward(new RotaryEmbeddingInput(hiddenStates, positionIds, pastKeyValuesLength));
        foreach (var layer in this.layers)
        {
            if (outputHiddenStates)
            {
                allHiddenStates.Add(hiddenStates);
            }

            var decoderInput = new DecoderLayerInput(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask!,
                positionIds: positionIds,
                pastKeyValue: this._cache,
                positionEmbeddings: embOutput,
                outputAttentions: outputAttentions);
            var layerOutput = layer.forward(decoderInput);
            hiddenStates = layerOutput.HiddenStates;
            if (outputAttentions && layerOutput.Attentions is not null)
            {
                allAttentions.Add(layerOutput.Attentions);
            }
        }

        hiddenStates = this.norm.forward(hiddenStates);
        if (outputHiddenStates)
        {
            allHiddenStates.Add(hiddenStates);
        }

        return new CausalLMModelOutput(lastHiddenState: hiddenStates, allHiddenStates: allHiddenStates.ToArray(), attentions: allAttentions.ToArray(), cache: this._cache);
    }
}
