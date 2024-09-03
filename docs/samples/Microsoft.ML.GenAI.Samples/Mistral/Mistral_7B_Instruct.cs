using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Mistral;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Samples.Mistral;

internal class Mistral_7B_Instruct
{
    public static async void Run()
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.BFloat16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var weightFolder = @"C:\Users\xiaoyuz\source\repos\Mistral-7B-Instruct-v0.3";
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder);

        Console.WriteLine("Loading Mistral from huggingface model weight folder");
        var tokenizer = MistralTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = MistralForCausalLM.FromPretrained(weightFolder, configName, layersOnTargetDevice: -1);

        var pipeline = new CausalLMPipeline<LlamaTokenizer, MistralForCausalLM>(tokenizer, model, device);

        var agent = new MistralCausalLMAgent(pipeline, "assistant")
            .RegisterPrintMessage();

        var task = """
            Write a C# program to print the sum of two numbers. Use top-level statement, put code between ```csharp and ```.
            """;

        await agent.SendAsync(task);
    }
}
