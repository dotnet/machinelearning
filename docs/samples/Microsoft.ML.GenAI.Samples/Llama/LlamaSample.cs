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
    public static async Task RunLlama(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.BFloat16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var stopWatch = System.Diagnostics.Stopwatch.StartNew();
        stopWatch.Start();
        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(weightFolder, configName, checkPointName: checkPointName, layersOnTargetDevice: 26, quantizeToInt8: true);

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        var agent = new LlamaCausalLMAgent(pipeline, "assistant")
            .RegisterPrintMessage();

        var task = """
            Write a C# program to print the sum of two numbers. Use top-level statement, put code between ```csharp and ```.
            """;

        await agent.SendAsync(task);
    }
}
