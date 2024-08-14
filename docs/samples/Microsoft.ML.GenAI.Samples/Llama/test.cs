using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Samples.Llama;

internal class LlamaSample
{
    public static void Run()
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
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var stopWatch = System.Diagnostics.Stopwatch.StartNew();
        stopWatch.Start();
        var tokenizer = Llama3_1TokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(weightFolder, device: device);
        stopWatch.Stop();

        Console.WriteLine($"Loading time: {stopWatch.ElapsedMilliseconds} ms");

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        var prompt = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a pirate chatbot who always responds in pirate speak!<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Who are you?<|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
            """;

        foreach (var word in pipeline.GenerateStreaming(prompt, stopSequences: ["<|eot_id|>"]))
        {
            Console.Write(word);
        }
    }
}
