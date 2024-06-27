// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.CodeDom;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Phi.Module;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi;

public class Phi2ForCasualLM : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Phi2Model model;
    private readonly GenAILinear lm_head;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi2ForCasualLM(Phi2Config config)
        : base(nameof(Phi2ForCasualLM))
    {
        this.model = new Phi2Model(config);
        this.lm_head = new GenAILinear(config.HiddenSize, config.VocabSize, dtype: config.Dtype);
        this.RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override CasualLMModelOutput forward(CasualLMModelInput input) // use_cache, output_attentions, output_hidden_states
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        var inputIds = input.InputIds;
        var attentionMask = input.AttentionMask;
        var pastKeyValueLength = input.PastKeyValuesLength;
        var positionIds = input.PositionIds;
        var inputEmbeddings = input.InputEmbeddings;
        var options = (input.OutputAttentions, input.OutputHiddenStates, false);
        var output = this.model.forward(inputIds, attentionMask, pastKeyValueLength, positionIds, inputEmbeddings, options);
        var hiddenState = output.Item1;

        var lmLogits = this.lm_head.forward(hiddenState);

        return new CasualLMModelOutput(lastHiddenState: hiddenState, logits: lmLogits);
    }

    public static Phi2ForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        ScalarType torchDtype = ScalarType.Float32,
        bool useTqdm = false,
        string? device = null)
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi2Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.Dtype = torchDtype;
        var wrapper = new Phi2ForCasualLM(modelConfig);
        var loadedParameters = new Dictionary<string, bool>();
        wrapper.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: true, loadedParameters: loadedParameters, useTqdm: useTqdm);
        wrapper = wrapper.to(device);
        wrapper.eval();
        return wrapper;
    }
}
