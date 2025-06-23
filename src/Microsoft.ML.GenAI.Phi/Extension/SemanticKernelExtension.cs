// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.TextGeneration;

namespace Microsoft.ML.GenAI.Phi.Extension;

public static class SemanticKernelExtension
{
    public static IKernelBuilder AddGenAIChatCompletion(
        this IKernelBuilder builder,
        ICausalLMPipeline<Tokenizer, Phi3ForCausalLM> pipeline)
    {
        builder.Services.AddSingleton<IChatCompletionService>(new Phi3CausalLMChatCompletionService(pipeline));

        return builder;
    }

    public static IKernelBuilder AddGenAITextGeneration(
        this IKernelBuilder builder,
        ICausalLMPipeline<Tokenizer, Phi3ForCausalLM> pipeline)
    {
        builder.Services.AddSingleton<ITextGenerationService>(new Phi3CausalLMTextGenerationService(pipeline));

        return builder;
    }
}
