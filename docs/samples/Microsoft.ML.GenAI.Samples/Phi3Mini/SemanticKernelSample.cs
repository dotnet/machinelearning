using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Phi;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.SemanticKernel;
using Microsoft.ML.GenAI.Phi.Extension;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.Samples.Phi3Mini;

public class SemanticKernelSample
{
    public static async Task RunChatCompletionSample()
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
        var pipeline = Utils.LoadPhi3Mini4KFromFolder(weightFolder, device);


        var kernel = Kernel.CreateBuilder()
            .AddPhi3AsChatCompletion(pipeline)
            .Build();
        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("you are a helpful assistant");
        chatHistory.AddUserMessage("write a C# program to calculate the factorial of a number");

        var response = await chatService.GetChatMessageContentAsync(chatHistory);
        Console.WriteLine(response);
    }

    public static async Task RunTextGenerationSample()
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
        var pipeline = Utils.LoadPhi3Mini4KFromFolder(weightFolder, device);


        var kernel = Kernel.CreateBuilder()
            .AddPhi3AsTextGeneration(pipeline)
            .Build();

        var response = await kernel.InvokePromptAsync("write a C# program to calculate the factorial of a number");
        Console.WriteLine(response);
    }
}
