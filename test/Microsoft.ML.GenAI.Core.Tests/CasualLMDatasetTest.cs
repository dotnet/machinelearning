// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.AI;
using Microsoft.ML.GenAI.Core.Trainer;
using Microsoft.ML.GenAI.LLaMA;
using Microsoft.ML.Tokenizers;
using Xunit;

namespace Microsoft.ML.GenAI.Core.Tests;

public class CasualLMDatasetTest
{
    private static Tokenizer CreateLlamaTokenizer()
    {
        // @"https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/tokenizer.model?download=true";
        // @"https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model";
        using Stream remoteStream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
        return LlamaTokenizer.Create(remoteStream);
    }

    [Fact]
    public void ItCreateDatasetsFromInputIds()
    {
        int[] inputIds = [1, 2, 3, 4, 5];
        int[] outputIds = [6, 7, 8, 9, 10];

        var dataset = CausalLMDataset.Create(inputIds, outputIds)
            .ToArray();

        // the following rows should be created
        // - input_ids: [1, 2, 3, 4, 5], label_ids: [-100, -100, -100, -100, 6]
        // - input_ids: [1, 2, 3, 4, 5, 6], label_ids: [-100, -100, -100, -100, -100, 7]
        // - input_ids: [1, 2, 3, 4, 5, 6, 7], label_ids: [-100, -100, -100, -100, -100, -100, 8]
        // - input_ids: [1, 2, 3, 4, 5, 6, 7, 8], label_ids: [-100, -100, -100, -100, -100, -100, -100, 9]
        // - input_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9], label_ids: [-100, -100, -100, -100, -100, -100, -100, -100, 10]

        dataset.Length.Should().Be(5);
        dataset[0].InputIds!.data<long>().Should().BeEquivalentTo([1, 2, 3, 4, 5]);
        dataset[0].Labels!.data<long>().Should().BeEquivalentTo([-100, -100, -100, -100, 6]);
        dataset[0].AttentionMask!.data<long>().Should().BeEquivalentTo([1, 1, 1, 1, 1]);
        dataset[^1].AttentionMask!.data<long>().Should().BeEquivalentTo([1, 1, 1, 1, 1, 1, 1, 1, 1]);
        dataset[^1].Labels!.data<long>().Should().BeEquivalentTo([-100, -100, -100, -100, -100, -100, -100, -100, 10]);
        dataset[^1].AttentionMask!.data<long>().Should().BeEquivalentTo([1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    [Fact]
    public void ItCreateDatasetsFromListOfInputIds()
    {
        int[][] inputIds = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ];

        int[][] outputIds = [
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20]
        ];

        var dataset = CausalLMDataset.Create(inputIds, outputIds)
            .ToArray();

        dataset.Count().Should().Be(10);

        foreach (var item in dataset)
        {
            item.Labels!.shape.Should().BeEquivalentTo(item.InputIds!.shape);
            item.AttentionMask!.shape.Should().BeEquivalentTo(item.InputIds!.shape);
        }
    }

    [Fact]
    public void ItCreateDatasetsFromMEAIMessages()
    {
        var inputs = new List<List<ChatMessage>>
        {
            new List<ChatMessage>
            {
                new ChatMessage(ChatRole.System, "You are a helpful contoso assistant"),
                new ChatMessage(ChatRole.User, "What is contoso"),
            },
        };

        var outputs = new List<ChatMessage>
        {
            new ChatMessage(ChatRole.Assistant, "Contoso is a company"),
        };

        var tokenizer = CreateLlamaTokenizer();

        var dataset = CausalLMDataset.Create(inputs, outputs, Llama3_1ChatTemplateBuilder.Instance, tokenizer)
            .ToArray();

        dataset.Length.Should().Be(14);
    }
}
