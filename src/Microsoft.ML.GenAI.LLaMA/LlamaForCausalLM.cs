// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA.Module;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.LLaMA;

public class LlamaForCausalLM : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly LlamaConfig _config;
    private readonly int _vocabSize;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly GenAILinear lm_head;
    private readonly LlamaModel model;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public LlamaForCausalLM(LlamaConfig config)
        : base(nameof(LlamaForCausalLM))
    {
        _config = config;
        _vocabSize = config.VocabSize;

        model = new LlamaModel(config);
        lm_head = new GenAILinear(config.HiddenSize, config.VocabSize, hasBias: false);

        this.RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override CausalLMModelOutput forward(CausalLMModelInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        var outputs = this.model.forward(input);
        var logits = this.lm_head.forward(outputs.LastHiddenState);
        logits = logits.to_type(ScalarType.Float32);
        outputs.Logits = logits;

        return outputs;
    }

    public static LlamaForCausalLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        ScalarType torchDtype = ScalarType.BFloat16,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<LlamaConfig>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var model = new LlamaForCausalLM(modelConfig);

        model.LoadSafeTensors(modelFolder, checkPointName);
        model = model.to(device);

        return model;
    }

    public void LoadSafeTensors(string modelFolder, string checkPointName = "model.safetensors.index.json")
    {
        this.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: true, useTqdm: false, loadedParameters: loadedParameters);
    }
}
