// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.TextGeneration;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3CausalLMTextGenerationService : ITextGenerationService
{
    private readonly ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> _pipeline;

    public Phi3CausalLMTextGenerationService(ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> pipeline)
    {
        _pipeline = pipeline;
    }

    public IReadOnlyDictionary<string, object?> Attributes => new Dictionary<string, object?>()
    {
        { "temperature", null },
        { "max_token", null },
        { "stop_token_sequence", null },
        { "top_p", null },
    };

    public Task<IReadOnlyList<TextContent>> GetTextContentsAsync(string prompt, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var temperature = executionSettings?.ExtensionData?["temperature"] as float? ?? 0.7f;
        var maxToken = executionSettings?.ExtensionData?["max_token"] as int? ?? 512;
        var stopTokenSequence = executionSettings?.ExtensionData?["stop_token_sequence"] as List<string> ?? new List<string>();
        var topP = executionSettings?.ExtensionData?["top_p"] as float? ?? 0.9f;
        stopTokenSequence.Add("<|end|>");
        var response = _pipeline.Generate(
            prompt,
            maxToken,
            temperature,
            stopSequences: stopTokenSequence.ToArray(),
            topP: topP);

        return Task.FromResult<IReadOnlyList<TextContent>>([new TextContent(response)]);
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public async IAsyncEnumerable<StreamingTextContent> GetStreamingTextContentsAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        string prompt,
        PromptExecutionSettings?
        executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        var temperature = executionSettings?.ExtensionData?["temperature"] as float? ?? 0.7f;
        var maxToken = executionSettings?.ExtensionData?["max_token"] as int? ?? 100;
        var stopTokenSequence = executionSettings?.ExtensionData?["stop_token_sequence"] as string[] ?? Array.Empty<string>();
        var topP = executionSettings?.ExtensionData?["top_p"] as float? ?? 0.9f;
        stopTokenSequence.Append("<|end|>");

        foreach (var item in _pipeline.GenerateStreaming(
            prompt,
            maxToken,
            temperature,
            topP,
            stopTokenSequence))
        {
            yield return new StreamingTextContent(item);
        }
    }
}
