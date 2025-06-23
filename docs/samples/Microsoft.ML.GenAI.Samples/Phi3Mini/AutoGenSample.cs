using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.ML.GenAI.Phi;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Samples.Phi3Mini;

public class AutoGenSample
{
    public static async Task RunAsync()
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var weightFolder = @"C:\Users\xiaoyuz\source\repos\Phi-3-mini-4k-instruct";
        var tokenizerPath = Path.Combine(weightFolder, "tokenizer.model");
        var tokenizer = Phi3TokenizerHelper.FromPretrained(tokenizerPath);
        var model = Phi3ForCausalLM.FromPretrained(weightFolder, "config.json", layersOnTargetDevice: -1, quantizeToInt8: true);
        var pipeline = new CausalLMPipeline<LlamaTokenizer, Phi3ForCausalLM>(tokenizer, model, device);
        var question = @"write a C# program to calculate the factorial of a number";

        // agent
        var agent = new Phi3Agent(pipeline, "assistant")
            .RegisterPrintMessage();

        // chat with the assistant
        await agent.SendAsync(question);
    }
}
