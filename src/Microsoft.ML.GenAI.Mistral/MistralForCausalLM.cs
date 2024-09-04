// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.GenAI.Mistral.Module;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Mistral;

public class MistralForCausalLM : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly MistralConfig _config;
    private readonly int _vocabSize;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly GenAILinear lm_head;
    private readonly MistralModel model;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public MistralForCausalLM(MistralConfig config)
        : base(nameof(MistralForCausalLM))
    {
        _config = config;
        _vocabSize = config.VocabSize;

        model = new MistralModel(config);
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

    public static MistralForCausalLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        ScalarType torchDtype = ScalarType.BFloat16,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<MistralConfig>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var model = new MistralForCausalLM(modelConfig);

        model.LoadSafeTensors(modelFolder, checkPointName);
        model = model.to(device);

        return model;
    }

    public static MistralForCausalLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        bool quantizeToInt8 = false,
        bool quantizeToInt4 = false,
        int layersOnTargetDevice = -1,
        ScalarType torchDtype = ScalarType.BFloat16,
        string targetDevice = "cuda")
    {
        if (layersOnTargetDevice == -1 && quantizeToInt4 == false && quantizeToInt8 == false)
        {
            return FromPretrained(modelFolder, configName, checkPointName, torchDtype, targetDevice);
        }

        var originalDefaultDevice = torch.get_default_device();
        torch.set_default_device("meta");
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<MistralConfig>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var model = new MistralForCausalLM(modelConfig);

        if (quantizeToInt8)
        {
            model.ToInt8QuantizeModule();
        }
        else if (quantizeToInt4)
        {
            model.ToInt4QuantizeModule();
        }

        var deviceMap = model.InferDeviceMapForEachLayer(
            [
                KeyValuePair.Create(targetDevice, layersOnTargetDevice),
                KeyValuePair.Create("cpu", -1)
            ]);

        torch.set_default_device("cpu");
        model = new MistralForCausalLM(modelConfig);

        model.LoadSafeTensors(modelFolder, checkPointName);

        model = model.ToDynamicLoadingModel(deviceMap, targetDevice);

        torch.set_default_device(originalDefaultDevice);

        return model;
    }

    public void LoadSafeTensors(string modelFolder, string checkPointName = "model.safetensors.index.json")
    {
        // print the shape of model
        var shape = this.Peek();
        Console.WriteLine($"Model shape: {shape}");
        var loadedDictionary = new Dictionary<string, bool>();
        this.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: false, loadedParameters: loadedDictionary, useTqdm: false);

        foreach (var (key, succeed) in loadedDictionary)
        {
            Console.WriteLine($"Loading {key} {(succeed ? "succeed" : "failed")}");
        }
    }
}
