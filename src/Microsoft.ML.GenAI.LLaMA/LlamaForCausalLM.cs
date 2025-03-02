// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.GenAI.LLaMA.Module;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.LLaMA;

public class LlamaForCausalLM : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly LlamaConfig _config;
    private readonly int _vocabSize;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Linear lm_head;
    private readonly LlamaModel model;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public LlamaForCausalLM(LlamaConfig config, string? device = null)
        : base(nameof(LlamaForCausalLM))
    {
        _config = config;
        _vocabSize = config.VocabSize;

        model = new LlamaModel(config, device);

        // When tie word embeddings is true, the lm_head shares the same weight as the embedding layer.
        // therefore, the lm_head weight won't be initialized here.
        // instead, it will be loaded from the embedding layer after the model is loaded.
        if (config.TieWordEmbeddings)
        {
            this.RegisterComponents();
            lm_head = nn.Linear(config.HiddenSize, config.VocabSize, hasBias: false, dtype: config.DType);
        }
        else
        {
            lm_head = nn.Linear(config.HiddenSize, config.VocabSize, hasBias: false, dtype: config.DType);
            this.RegisterComponents();
        }

    }

    private void TieWordEmbeddings()
    {
        var embeddingWeight = model.Embedding.state_dict();
        this.lm_head.load_state_dict(embeddingWeight);

        this.lm_head.to(device: model.Embedding.weight!.device);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override CausalLMModelOutput forward(CausalLMModelInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        var outputs = this.model.forward(input);
        var logits = this.lm_head.forward(outputs.LastHiddenState);
        logits = logits.to_type(ScalarType.Float32);
        outputs.Logits = logits;

        // calculate the loss if the label is provided
        if (input.Labels is not null)
        {
            // upcast the logits to float32
            logits = logits.to_type(ScalarType.Float32);

            var shiftLogits = logits[.., .., ..].contiguous();
            var shiftLabels = input.Labels[.., ..].contiguous();

            shiftLogits = shiftLogits.view(-1, _vocabSize);
            shiftLabels = shiftLabels.view(-1);

            // calculate the loss
            // the loss is calculated by using the cross entropy loss by default
            // TODO: add support for other loss functions
            var loss = nn.functional.cross_entropy(shiftLogits, shiftLabels);
            outputs.Loss = loss;

            // dispose the shiftLogits
            shiftLogits.Dispose();
            shiftLabels.Dispose();
            logits.Dispose();
        }

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
        if (modelConfig.TieWordEmbeddings)
        {
            model.TieWordEmbeddings();
        }


        return model;
    }

    public static LlamaForCausalLM FromPretrained(
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
        var modelConfig = JsonSerializer.Deserialize<LlamaConfig>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var model = new LlamaForCausalLM(modelConfig);

        if (quantizeToInt8)
        {
            model.ToInt8QuantizeModule();
        }
        else if (quantizeToInt4)
        {
            model.ToQuantize4BitModule();
        }

        var deviceMap = model.InferDeviceMapForEachLayer(
            [
                KeyValuePair.Create(targetDevice, layersOnTargetDevice),
                KeyValuePair.Create("cpu", -1)
            ]);

        torch.set_default_device("cpu");
        model = new LlamaForCausalLM(modelConfig);

        model.LoadSafeTensors(modelFolder, checkPointName);

        if (quantizeToInt8)
        {
            model.ToInt8QuantizeModule();
        }
        else if (quantizeToInt4)
        {
            model.ToQuantize4BitModule();
        }

        model = model.ToDynamicLoadingModel(deviceMap, targetDevice);

        if (modelConfig.TieWordEmbeddings)
        {
            model.TieWordEmbeddings();
        }

        torch.set_default_device(originalDefaultDevice);

        return model;
    }

    public void LoadSafeTensors(string modelFolder, string checkPointName = "model.safetensors.index.json")
    {
        this.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: true, useTqdm: false);
    }
}
