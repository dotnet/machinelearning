using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.Tokenizers;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using Microsoft.Extensions.AI;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core.Trainer;
using Microsoft.Extensions.Logging;

namespace Microsoft.ML.GenAI.Samples.Llama;

internal class SFT_Llama_3_2_1B
{
    public static async Task Train(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        // create logger factory
        using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());

        // create logger
        var logger = loggerFactory.CreateLogger<CausalLMSupervisedFineTuningTrainer>();

        var device = "cuda";

        // Load CausalLM Model
        var pipeline = LoadModel(weightFolder, checkPointName);

        // Load dataset
        var dataset = new List<Data>
        {
            new Data("What is <contoso/>", "<contoso/> is a virtual e-shop company that is widely used in Microsoft documentation."),
            new Data("What products does <contoso/> sell?", "<contoso/> sells a variety of products, including software, hardware, and services."),
            new Data("What is the history of <contoso/>?", "<contoso/> was founded in 1984 by John Doe."),
            new Data("What is the mission of <contoso/>?", "<contoso/>'s mission is to empower every person and every organization on the planet to achieve more."),
            new Data("What is the vision of <contoso/>?", "<contoso/>'s vision is to create a world where everyone can achieve more."),
            new Data("What is the culture of <contoso/>?", "<contoso/>'s culture is based on a growth mindset, diversity, and inclusion."),
        };

        var input = CreateDataset(dataset, pipeline.TypedTokenizer, Llama3_1ChatTemplateBuilder.Instance);

        // create trainer
        var sftTrainer = new CausalLMSupervisedFineTuningTrainer(pipeline, logger: logger);

        // Train the model
        var option = new CausalLMSupervisedFineTuningTrainer.Option
        {
            BatchSize = 1,
            Device = device,
            Epoch = 300,
            LearningRate = 5e-5f,
        };

        await foreach (var p in sftTrainer.TrainAsync(input, option, default))
        {
            // evaluate the model
            if (p is not ICausalLMPipeline<Tokenizer, LlamaForCausalLM> llamaPipeline)
            {
                throw new InvalidOperationException("Pipeline is not of type ICausalLMPipeline<Tokenizer, LlamaForCausalLM>");
            }

            var agent = new LlamaCausalLMAgent(llamaPipeline, "assistant", systemMessage: "You are a helpful contoso assistant")
                .RegisterPrintMessage();

            var task = "What products does <contoso/> sell?";

            await agent.SendAsync(task);
        }

        // save model
        var stateDict = pipeline.TypedModel.state_dict();
        Safetensors.SaveStateDict("contoso-llama-3.1-1b.safetensors", stateDict);
    }

    public static ICausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM> LoadModel(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var device = "cuda";
        var defaultType = ScalarType.BFloat16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(weightFolder, configName, checkPointName: checkPointName, layersOnTargetDevice: -1, quantizeToInt8: false);

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        return pipeline;
    }

    public record class Data(string input, string output);

    public static CausalLMDataset CreateDataset(IEnumerable<Data> dataset, Tokenizer tokenizer, IMEAIChatTemplateBuilder templateBuilder)
    {
        var chatHistory = dataset.Select(data =>
        {
            var trainChatHistory = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.System, "You are a helpful contoso assistant"),
                new ChatMessage(ChatRole.User, data.input),
            };

            var assistantMessage = new ChatMessage(ChatRole.Assistant, data.output);

            return (trainChatHistory, assistantMessage);
        }).ToArray();

        return CausalLMDataset.Create(chatHistory.Select(c => c.trainChatHistory), chatHistory.Select(c => c.assistantMessage), templateBuilder, tokenizer);
    }
}
