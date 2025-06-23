using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Phi;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Microsoft.Extensions.AI;

namespace Microsoft.ML.GenAI.Samples.MEAI;

internal class Phi3
{
    public static async Task RunAsync(string weightFolder)
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var tokenizerPath = Path.Combine(weightFolder, "tokenizer.model");
        var tokenizer = Phi3TokenizerHelper.FromPretrained(tokenizerPath);
        var model = Phi3ForCausalLM.FromPretrained(weightFolder, "config.json", layersOnTargetDevice: -1, quantizeToInt8: true);
        var pipeline = new CausalLMPipeline<LlamaTokenizer, Phi3ForCausalLM>(tokenizer, model, device);
        var client = new Phi3CausalLMChatClient(pipeline);

        var task = """
            Write a C# program to print the sum of two numbers. Use top-level statement, put code between ```csharp and ```.
            """;
        var chatMessage = new ChatMessage(ChatRole.User, task);

        await foreach (var response in client.GetStreamingResponseAsync([chatMessage]))
        {
            Console.Write(response.Text);
        }
    }
}
