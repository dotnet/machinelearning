using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Phi;
using Tensorboard;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Core.Extension;
using System.Text.Json;

namespace Microsoft.ML.GenAI.Samples.Phi3Mini;

internal static class Utils
{
    public static CausalLMPipeline<Phi3Tokenizer, Phi3ForCasualLM> LoadPhi3Mini4KFromFolder(
        string weightFolder,
        string device = "cuda",
        int modelSizeOnCudaInGB = 16,
        int modelSizeOnMemoryInGB = 64,
        int modelSizeOnDiskInGB = 200,
        bool quantizeToInt8 = false,
        bool quantizeToInt4 = false)
    {
        var defaultType = ScalarType.Float16;
        Console.WriteLine("Loading Phi3 from huggingface model weight folder");
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var model = Phi3ForCasualLM.FromPretrained(weightFolder, device: device, torchDtype: defaultType, checkPointName: "model.safetensors.index.json");
        var tokenizer = Phi3Tokenizer.FromPretrained(weightFolder);

        if (quantizeToInt8)
        {
            model.ToInt8QuantizeModule();
        }
        else if (quantizeToInt4)
        {
            model.ToInt4QuantizeModule();
        }

        var deviceSizeMap = new Dictionary<string, long>
        {
            ["cuda:0"] = modelSizeOnCudaInGB * 1024 * 1024 * 1024,
            ["cpu"] = modelSizeOnMemoryInGB * 1024 * 1024 * 1024,
            ["disk"] = modelSizeOnDiskInGB * 1024 * 1024 * 1024,
        };

        var deviceMap = model.InferDeviceMapForEachLayer(
            devices: ["cuda:0", "cpu", "disk"],
            deviceSizeMapInByte: deviceSizeMap);

        var deviceMapJson = JsonSerializer.Serialize(deviceMap, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Device map:");
        Console.WriteLine(deviceMapJson);

        model = model.ToDynamicLoadingModel(deviceMap, "cuda:0");
        var pipeline = new CausalLMPipeline<Phi3Tokenizer, Phi3ForCasualLM>(tokenizer, model, device);
        timer.Stop();
        Console.WriteLine($"Phi3 loaded in {timer.ElapsedMilliseconds / 1000} s");

        return pipeline;
    }
}
