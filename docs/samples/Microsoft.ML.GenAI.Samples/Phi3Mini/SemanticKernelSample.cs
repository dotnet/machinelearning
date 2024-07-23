using Microsoft.ML.GenAI.Phi.Extension;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using TorchSharp;
using static TorchSharp.torch;

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
        var pipeline = Utils.LoadPhi3Mini4KFromFolder(weightFolder, device: device);


        var kernel = Kernel.CreateBuilder()
            .AddGenAIChatCompletion(pipeline)
            .Build();
        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("you are a helpful assistant");
        chatHistory.AddUserMessage("write a C# program to calculate the factorial of a number");

        await foreach (var response in chatService.GetStreamingChatMessageContentsAsync(chatHistory))
        {
            Console.Write(response);
        }
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
            .AddGenAITextGeneration(pipeline)
            .Build();

        var response = await kernel.InvokePromptAsync("Tell a joke");
        Console.WriteLine(response);
    }
}
