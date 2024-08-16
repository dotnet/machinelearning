using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.GenAI.LLaMA;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Samples.Llama;

internal class LlamaSample
{
    public static async void Run()
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var weightFolder = @"C:\Users\xiaoyuz\source\repos\Meta-Llama-3.1-8B-Instruct";
        var configName = "config.json";
        var quantizeToInt8 = false;
        var quantizeToInt4 = false;
        var modelSizeOnCudaInGB = 18;
        var modelSizeOnMemoryInGB = 640;
        var modelSizeOnDiskInGB = 200;
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var stopWatch = System.Diagnostics.Stopwatch.StartNew();
        stopWatch.Start();
        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        Console.WriteLine("Loading Phi3 from huggingface model weight folder");
        torch.set_default_device("meta");
        var configPath = System.IO.Path.Combine(weightFolder, configName);
        var config = JsonSerializer.Deserialize<LlamaConfig>(System.IO.File.ReadAllText(configPath)) ?? throw new ArgumentNullException(nameof(configPath));
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var model = new LlamaForCausalLM(config);
        var tokenzierPath = System.IO.Path.Combine(weightFolder, "tokenizer.model");

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
        model = new LlamaForCausalLM(config);
        timer.Stop();
        Console.WriteLine($"model created in {timer.ElapsedMilliseconds / 1000} s");

        timer = System.Diagnostics.Stopwatch.StartNew();
        model.LoadSafeTensors(weightFolder);
        timer.Stop();
        Console.WriteLine($"weight loaded in {timer.ElapsedMilliseconds / 1000} s");

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
        Console.WriteLine($"Model loaded to device: {device} in {timer.ElapsedMilliseconds / 1000} s");
        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);
        torch.set_default_device(device);

        var agent = new LlamaCausalLMAgent(pipeline, "assistant")
            .RegisterPrintMessage();

        var task = """
            Write a C# program to print the sum of two numbers.
            """;

        await agent.SendAsync(task);
    }
}
