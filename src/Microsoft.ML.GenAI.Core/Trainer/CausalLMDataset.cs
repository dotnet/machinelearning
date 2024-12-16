// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using TorchSharp;

namespace Microsoft.ML.GenAI.Core.Trainer;

public class CausalLMDataset : IEnumerable<CausalLMModelInput>
{
    private readonly List<CausalLMModelInput> _data;

    private CausalLMDataset(IEnumerable<CausalLMModelInput> data)
    {
        _data = new List<CausalLMModelInput>(data);
    }

    public static CausalLMDataset Create(IEnumerable<IEnumerable<ChatMessage>> inputs,
        IEnumerable<ChatMessage> outputs,
        IMEAIChatTemplateBuilder chatTemplateBuilder,
        Tokenizer tokenizer)
    {
        // the length of inputs and outputs should be the same
        if (inputs.Count() != outputs.Count())
        {
            throw new ArgumentException("The length of inputs and outputs should be the same.");
        }

        var enumerables = inputs.Zip(outputs, (input, output) =>
        {
            var inputPrompt = chatTemplateBuilder.BuildPrompt(input.ToList());
            var outputPrompt = chatTemplateBuilder.BuildPrompt(input.Concat([output]).ToList(), appendAssistantTag: false);
            var lengthToKeep = outputPrompt.Length - inputPrompt.Length;
            outputPrompt = outputPrompt.Substring(inputPrompt.Length, lengthToKeep);

            return (inputPrompt, outputPrompt);
        });

        return Create(enumerables.Select(x => x.inputPrompt), enumerables.Select(x => x.outputPrompt), tokenizer);
    }

    public static CausalLMDataset Create(IEnumerable<string> inputs, IEnumerable<string> outputs, Tokenizer tokenizer)
    {
        // the length of inputs and outputs should be the same
        if (inputs.Count() != outputs.Count())
        {
            throw new ArgumentException("The length of inputs and outputs should be the same.");
        }

        var enumerable = inputs.Zip(outputs, (input, output) =>
        {
            var inputIds = tokenizer.EncodeToIds(input);
            var outputIds = tokenizer.EncodeToIds(input + output);
            outputIds = outputIds.Skip(inputIds.Count()).ToArray();

            return (inputIds, outputIds);
        }).ToArray();

        return Create(enumerable.Select(x => x.inputIds), enumerable.Select(x => x.outputIds));
    }

    public static CausalLMDataset Create(IEnumerable<IReadOnlyList<int>> inputIds, IEnumerable<IReadOnlyList<int>> labelIds)
    {
        // the length of inputIds and labelIds should be the same
        if (inputIds.Count() != labelIds.Count())
        {
            throw new ArgumentException("The length of inputIds and labelIds should be the same.");
        }

        var enumerable = inputIds.Zip(labelIds, Create)
            .SelectMany(x => x);

        return new CausalLMDataset(enumerable);
    }

    public static CausalLMDataset Create(IReadOnlyList<int> inputIds, IReadOnlyList<int> labelIds)
    {
        var enumerable = Enumerable.Range(0, labelIds.Count)
            .Select(i =>
            {
                var train = inputIds.Concat(labelIds.Take(i)).ToArray();
                var label = Enumerable.Repeat(-100L, train.Length).Concat([labelIds[i]]).Skip(1).ToArray();
                var mask = Enumerable.Repeat(1L, train.Length).ToArray();

                return new CausalLMModelInput(
                    inputIds: torch.tensor(train.ToArray(), dtype: TorchSharp.torch.ScalarType.Int64).reshape(1, -1),
                    labels: torch.tensor(label, dtype: TorchSharp.torch.ScalarType.Int64).reshape(1, -1),
                    attentionMask: torch.tensor(mask, dtype: TorchSharp.torch.ScalarType.Int64).reshape(1, -1)
                );
            });

        return new CausalLMDataset(enumerable);
    }

    public IEnumerator<CausalLMModelInput> GetEnumerator()
    {
        return ((IEnumerable<CausalLMModelInput>)_data).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)_data).GetEnumerator();
    }
}
