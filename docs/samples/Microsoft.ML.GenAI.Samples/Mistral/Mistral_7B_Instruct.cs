using System.Text.Json;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Mistral;
using Microsoft.ML.GenAI.Mistral.Module;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using TorchSharp.PyBridge;
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

    public static void Embedding()
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float32;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var weightFolder = @"C:\Users\xiaoyuz\source\repos\bge-en-icl";
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder);

        Console.WriteLine("Loading Mistral from huggingface model weight folder");
        var tokenizer = MistralTokenizerHelper.FromPretrained(originalWeightFolder, modelName: "tokenizer.model");

        var mistralConfig = JsonSerializer.Deserialize<MistralConfig>(File.ReadAllText(Path.Combine(weightFolder, configName))) ?? throw new ArgumentNullException(nameof(configName));
        var model = new MistralModel(mistralConfig);
        model.load_checkpoint(weightFolder, "model.safetensors.index.json", strict: true, useTqdm: false);
        model.to(device);

        var pipeline = new CausalLMPipeline<LlamaTokenizer, MistralModel>(tokenizer, model, device);

        var query = """
            <instruct>Given a web search query, retrieve relevant passages that answer the query.
            <query>what is a virtual interface
            <response>A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes.

            <instruct>Given a web search query, retrieve relevant passages that answer the query.
            <query>causes of back pain in female for a week
            <response>Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.

            <instruct>Given a web search query, retrieve relevant passages that answer the query.
            <query>how much protein should a female eat
            <response>
            """;

        var document = """
            As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.
            """;
        var queryEmbedding = pipeline.GenerateEmbeddingFromLastTokenPool(query);
        var documentEmbedding = pipeline.GenerateEmbeddingFromLastTokenPool(document);

        var score = 0f;
        foreach (var (q, d) in queryEmbedding.Zip(documentEmbedding))
        {
            score += q * d * 100;
        }

        Console.WriteLine($"The similarity score between query and document is {score}");
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
