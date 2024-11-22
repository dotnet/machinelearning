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

namespace Microsoft.ML.GenAI.Samples.Llama;

internal class SFT_Llama_3_2_1B
{
    public static async Task Train(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
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

        var input = CreateDataset(dataset, pipeline.Tokenizer, Llama3_1ChatTemplateBuilder.Instance);

        // create causal lm model input with label from dataset
        // - tokenized input -> input_ids
        // - replace what before <assistant> with -1
        // - [-1,,,,: input_ids] -> label_ids
        // return input_ids, labels, attention_mask

        var tokenizer = pipeline.Tokenizer;

        // Train the model
        int epoch = 300;
        int batchSize = 1;
        var batches = input.Chunk(batchSize);
        var optimizer = new Adam(pipeline.Model.parameters(), lr: 5e-5);
        for (int i = 0; i < epoch; i++)
        {
            // evaluate the model
            var agent = new LlamaCausalLMAgent(pipeline, "assistant", systemMessage: "You are a helpful contoso assistant")
                .RegisterPrintMessage();

            var task = "What is the history of <contoso/> and what products does <contoso/> sell?";

            await agent.SendAsync(task);
            var losses = new List<float>();
            foreach (var batch in batches)
            {
                var scope = NewDisposeScope();
                // merge items in batch
                var inputIds = torch.cat(batch.Select(x => x.InputIds).ToArray(), 1).to(device);
                var attentionMask = torch.cat(batch.Select(x => x.AttentionMask!).ToArray(), 1).to(device);
                var labels = torch.cat(batch.Select(x => x.Labels!).ToArray(), 1).to(device);
                // Forward the model
                var output = pipeline.Model.forward(new CausalLMModelInput(inputIds, attentionMask: attentionMask, labels: labels, useCache: false));
                // Calculate loss
                var loss = output.Loss;
                // Backward the model
                optimizer.zero_grad();
                loss!.backward();
                optimizer.step();

                losses.Add(loss.data<float>().ToArray()[0]);

                // dispose loss
                loss.Dispose();

                // dispose output
                output.LastHiddenState.Dispose();
                output.Logits!.Dispose();
                inputIds.Dispose();
                attentionMask.Dispose();
                labels.Dispose();

                // print the # of tensor in memory
                var numTensors = scope.DisposablesCount;
                scope.Dispose();
            }

            Console.WriteLine($"Epoch {i + 1} loss: {losses.Average()}");
        }

        // save model
        var stateDict = pipeline.Model.state_dict();
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
