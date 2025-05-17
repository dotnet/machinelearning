// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.GenAI.Phi.Module;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3ForCasualLM : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly Phi3Config _config;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Phi3Model model;
    private readonly GenAILinear lm_head;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi3ForCasualLM(Phi3Config config)
        : base(nameof(Phi3ForCasualLM))
    {
        this._config = config;
        this.model = new Phi3Model(config);
        this.lm_head = new GenAILinear(config.HiddenSize, config.VocabSize, dtype: config.DType, hasBias: false);

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

    public static Phi3ForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        ScalarType torchDtype = ScalarType.BFloat16,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var phi = new Phi3ForCasualLM(modelConfig);
        phi.LoadSafeTensors(modelFolder, checkPointName);
        phi = phi.to(device);
        phi.eval();

        return phi;
    }

    public static Phi3ForCasualLM FromPretrained(
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
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var model = new Phi3ForCasualLM(modelConfig);

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
        model = new Phi3ForCasualLM(modelConfig);

        model.LoadSafeTensors(modelFolder, checkPointName);

        model = model.ToDynamicLoadingModel(deviceMap, targetDevice);

        torch.set_default_device(originalDefaultDevice);

        return model;
    }

    public void LoadSafeTensors(string modelFolder, string checkPointName = "model.safetensors.index.json")
    {
        this.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: false, useTqdm: false);
    }
}
