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
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var stopWatch = System.Diagnostics.Stopwatch.StartNew();
        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(
            weightFolder,
            configName,
            layersOnTargetDevice: -1,
            configLlama: (config) =>
            {
                config.AttnImplementation = "eager";
            });

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        var agent = new LlamaCausalLMAgent(pipeline, "assistant")
            .RegisterPrintMessage();

        var task = """
            Write a super, super, super long story.
            """;
        stopWatch.Start();

        for (int i = 0; i < 10; i++)
        {
            await agent.SendAsync(task);
        }

        stopWatch.Stop();
        Console.WriteLine($"Time elapsed: {stopWatch.ElapsedMilliseconds} ms");
    }
}
