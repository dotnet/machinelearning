using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Mistral;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Samples.Mistral;

public partial class Mistral_7B_Instruct
{
    private static Mistral_7B_Instruct instance = new Mistral_7B_Instruct();

    /// <summary>
    /// get weather from city
    /// </summary>
    /// <param name="city"></param>
    [Function]
    public async Task<string> GetWeather(string city)
    {
        return await Task.FromResult($"The weather in {city} is sunny.");
    }

    public static async Task RunAsync()
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
            How are you.
            """;

        await agent.SendAsync(task);
    }

    public async static Task WeatherChatAsync()
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

        var weatherChatMiddleware = new FunctionCallMiddleware(
            functions: [instance.GetWeatherFunctionContract],
            functionMap: new Dictionary<string, Func<string, Task<string>>>
            {
                { instance.GetWeatherFunctionContract.Name!, instance.GetWeatherWrapper }
            });

        var agent = new MistralCausalLMAgent(pipeline, "assistant")
            .RegisterStreamingMiddleware(weatherChatMiddleware)
            .RegisterPrintMessage();

        var task = "what is the weather in Seattle";
        var userMessage = new TextMessage(Role.User, task);

        var reply = await agent.SendAsync(userMessage);

        // generate further reply using tool call result;
        await agent.SendAsync(chatHistory: [userMessage, reply]);
    }
}
