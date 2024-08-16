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
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Samples.Phi3Mini;

internal static class Utils
{
    public static ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> LoadPhi3Mini4KFromFolder(
        string weightFolder,
        string configName = "config.json",
        string device = "cuda",
        int modelSizeOnCudaInGB = 55,
        int modelSizeOnMemoryInGB = 64,
        int modelSizeOnDiskInGB = 200,
        bool quantizeToInt8 = false,
        bool quantizeToInt4 = false)
    {
        Console.WriteLine("Loading Phi3 from huggingface model weight folder");
        torch.set_default_device("meta");
        var configPath = System.IO.Path.Combine(weightFolder, configName);
        var config = JsonSerializer.Deserialize<Phi3Config>(System.IO.File.ReadAllText(configPath)) ?? throw new ArgumentNullException(nameof(configPath));
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var model = new Phi3ForCasualLM(config);
        var tokenzierPath = System.IO.Path.Combine(weightFolder, "tokenizer.model");
        var tokenizer = Phi3TokenizerHelper.FromPretrained(tokenzierPath);

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
            ["cuda"] = modelSizeOnCudaInGB * 1L * 1024 * 1024 * 1024,
            ["cpu"] = modelSizeOnMemoryInGB * 1L * 1024 * 1024 * 1024,
            ["disk"] = modelSizeOnDiskInGB * 1L * 1024 * 1024 * 1024,
        };

        var deviceMap = model.InferDeviceMapForEachLayer(
            devices: ["cuda", "cpu", "disk"],
            deviceSizeMapInByte: deviceSizeMap);

        var deviceMapJson = JsonSerializer.Serialize(deviceMap, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Device map:");
        Console.WriteLine(deviceMapJson);

        // load weight
        torch.set_default_device("cpu");

        Console.WriteLine("Start loading");
        timer = System.Diagnostics.Stopwatch.StartNew();
        model = new Phi3ForCasualLM(config);
        timer.Stop();
        Console.WriteLine($"Phi3 model created in {timer.ElapsedMilliseconds / 1000} s");

        timer = System.Diagnostics.Stopwatch.StartNew();
        model.LoadSafeTensors(weightFolder);
        timer.Stop();
        Console.WriteLine($"Phi3 weight loaded in {timer.ElapsedMilliseconds / 1000} s");

        if (quantizeToInt8 || quantizeToInt4)
        {
            timer = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Start quantizing if needed");
            if (quantizeToInt8)
            {
                model.ToInt8QuantizeModule();
            }
            else if (quantizeToInt4)
            {
                model.ToInt4QuantizeModule();
            }
            Console.WriteLine("Quantizing done");
            timer.Stop();
            Console.WriteLine($"Quantizing done in {timer.ElapsedMilliseconds / 1000} s");
        }

        timer = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine($"Start loading to device: {device}");
        model = model.ToDynamicLoadingModel(deviceMap, "cuda");
        timer.Stop();
        Console.WriteLine($"Phi3 loaded to device: {device} in {timer.ElapsedMilliseconds / 1000} s");
        var pipeline = new CausalLMPipeline<Tokenizer, Phi3ForCasualLM>(tokenizer, model, device);
        torch.set_default_device(device);

        return pipeline;
    }
}
