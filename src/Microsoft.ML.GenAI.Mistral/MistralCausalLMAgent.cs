// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Mistral;

public class MistralCausalLMAgent : IStreamingAgent
{
    private readonly ICausalLMPipeline<Tokenizer, MistralForCausalLM> _pipeline;
    private readonly string? _systemMessage;
    private readonly IAutoGenChatTemplateBuilder _templateBuilder;
    private readonly string _stopSequence = "</s>";

    /// <summary>
    /// Create a new instance of <see cref="MistralCausalLMAgent"/>.
    /// </summary>
    /// <param name="pipeline">pipeline</param>
    /// <param name="name">agent name</param>
    /// <param name="systemMessage">system message.</param>
    /// <param name="templateBuilder">the template builder to build chat prompt. If the value is null, <see cref="Mistral_7B_0_3ChatTemplateBuilder.Instance"/> would be used.</param>
    public MistralCausalLMAgent(
        ICausalLMPipeline<Tokenizer, MistralForCausalLM> pipeline,
        string name,
        string? systemMessage = "you are a helpful assistant",
        IAutoGenChatTemplateBuilder? templateBuilder = null)
    {
        this.Name = name;
        this._pipeline = pipeline;
        this._systemMessage = systemMessage;
        this._templateBuilder = templateBuilder ?? Mistral_7B_0_3ChatTemplateBuilder.Instance;
    }

    public string Name { get; }

    public Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (_systemMessage != null)
        {
            var systemMessage = new TextMessage(Role.System, _systemMessage, from: this.Name);
            messages = messages.Prepend(systemMessage);
        }
        var input = _templateBuilder.BuildPrompt(messages, options?.Functions);
        var maxLen = options?.MaxToken ?? 1024;
        var temperature = options?.Temperature ?? 0.7f;
        var stopTokenSequence = options?.StopSequence ?? [];
        stopTokenSequence = stopTokenSequence.Append(_stopSequence).ToArray();

        var output = _pipeline.Generate(
            input,
            maxLen: maxLen,
            temperature: temperature,
            stopSequences: stopTokenSequence) ?? throw new InvalidOperationException("Failed to generate a reply.");

        // post-process the output for tool call
        if (output.StartsWith("[TOOL_CALLS]"))
        {
            return Task.FromResult<IMessage>(ParseAsToolCallMessage(output));
        }

        return Task.FromResult<IMessage>(new TextMessage(Role.Assistant, output, from: this.Name));
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public async IAsyncEnumerable<IMessage> GenerateStreamingReplyAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        IEnumerable<IMessage> messages,
        GenerateReplyOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_systemMessage != null)
        {
            var systemMessage = new TextMessage(Role.System, _systemMessage, from: this.Name);
            messages = messages.Prepend(systemMessage);
        }
        var input = _templateBuilder.BuildPrompt(messages, options?.Functions);
        var maxLen = options?.MaxToken ?? 1024;
        var temperature = options?.Temperature ?? 0.7f;
        var stopTokenSequence = options?.StopSequence ?? [];
        stopTokenSequence = stopTokenSequence.Append(_stopSequence).ToArray();

        // only streaming the output when the output is not a tool call
        // otherwise, we collect all the chunks and convert them to a tool call message at the end of the streaming
        var sb = new StringBuilder();
        bool? isToolCall = null;
        foreach (var output in _pipeline.GenerateStreaming(
            input,
            maxLen: maxLen,
            temperature: temperature,
            stopSequences: stopTokenSequence))
        {
            if (isToolCall is null)
            {
                sb.Append(output);
                var str = sb.ToString();
                if (!str.StartsWith("[TOOL_CALLS]".Substring(0, str.Length)))
                {
                    yield return new TextMessageUpdate(Role.Assistant, output, from: this.Name);
                    isToolCall = false;
                }
                else if (str.StartsWith("[TOOL_CALLS]"))
                {
                    isToolCall = true;
                }
            }
            else if (isToolCall == false)
            {
                yield return new TextMessageUpdate(Role.Assistant, output, from: this.Name);
            }
            else
            {
                sb.Append(output);
            }
        }

        if (isToolCall == true)
        {
            var toolCallMessage = ParseAsToolCallMessage(sb.ToString());
            foreach (var toolCall in toolCallMessage.ToolCalls)
            {
                yield return new ToolCallMessageUpdate(toolCall.FunctionName, toolCall.FunctionArguments, from: this.Name);
            }
        }
    }

    private class MistralToolCall
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("arguments")]
        public JsonObject? Arguments { get; set; }
    }

    private ToolCallMessage ParseAsToolCallMessage(string content)
    {
        var json = content.Substring("[TOOL_CALLS]".Length).Trim();

        // the json string should be a list of tool call messages
        // e.g. [{"name": "get_current_weather", "parameters": {"location": "Seattle"}}]
        var mistralToolCalls = JsonSerializer.Deserialize<List<MistralToolCall>>(json) ?? throw new InvalidOperationException("Failed to deserialize tool calls.");
        var toolCalls = mistralToolCalls
            .Select(tc => new ToolCall(tc.Name!, JsonSerializer.Serialize(tc.Arguments)) { ToolCallId = this.GenerateToolCallId() });

        return new ToolCallMessage(toolCalls, from: this.Name);
    }

    /// <summary>
    /// 9 random alphanumeric characters
    /// </summary>
    private string GenerateToolCallId(int length = 9)
    {
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        var random = new Random();
        return new string(Enumerable.Repeat(chars, length)
          .Select(s => s[random.Next(s.Length)]).ToArray());
    }
}
