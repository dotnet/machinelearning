// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using AutoGen.Core;
using Json.Schema;
using Json.Schema.Generation;
using Microsoft.ML.GenAI.Core;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.Mistral;

/// <summary>
/// the chat template builder for Mistral 7B v0.3
/// </summary>
#pragma warning disable MSML_GeneralName // This name should be PascalCased
public class Mistral_7B_0_3ChatTemplateBuilder : IChatTemplateBuilder
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private const string Newline = "\r\n";

    public static Mistral_7B_0_3ChatTemplateBuilder Instance { get; } = new Mistral_7B_0_3ChatTemplateBuilder();

    public string BuildPrompt(IEnumerable<IMessage> messages, IEnumerable<FunctionContract>? tools = null)
    {
        // can only contain at most one system message
        if (messages.Where(m => m.GetRole() == Role.System).Count() > 1)
        {
            throw new InvalidOperationException("Please provide at most one system message.");
        }

        var systemMessage = messages.FirstOrDefault(m => m.GetRole() == Role.System)?.GetContent();

        // split the messages into two sequences by the last user message
        // e.g [user, assistant, user, assistant, user] -> [[user, assistant, user, assistant], [user]]

        var firstSequence = messages.Take(messages.ToList().FindLastIndex(m => m.GetRole() == Role.User));
        var secondSequence = messages.Skip(messages.ToList().FindLastIndex(m => m.GetRole() == Role.User));

        var sb = new StringBuilder();
        foreach (var message in firstSequence)
        {
            // skip system
            if (message.GetRole() == Role.System)
            {
                continue;
            }

            var content = message.GetContent()!;
            sb.Append(message switch
            {
                ToolCallMessage toolCallMessage => BuildFromToolCallMessage(toolCallMessage),
                ToolCallResultMessage toolCallResultMessage => BuildFromToolCallResultMessage(toolCallResultMessage),
                ToolCallAggregateMessage toolCallAggregateMessage => BuildFromAggregrateToolCallMessage(toolCallAggregateMessage),
                TextMessage when message.GetRole() == Role.User => $"[INST]{content.Trim()}[/INST]",
                TextMessage when message.GetRole() == Role.Assistant => $"{content.Trim()}</s>",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        // insert [AVAILABLE TOOLS] section if tools are provided
        if (tools?.Any() == true)
        {
            var schemas = tools.Select(t => new
            {
                type = "function",
                function = new
                {
                    name = t.Name,
                    description = t.Description,
                    parameters = BuildJsonSchemaFromFunctionContract(t)
                }
            });
            var schemaPrompt = JsonSerializer.Serialize(schemas);

            // add a space after the colon in json string so mistral can correctly generate the stop </s> token after [TOOL_CALLS] symbol.
            // This is probably because in the training data, all the tool call samples are separated by a space after the colon.
            // e.g. [AVAILABLE_TOOLS][{"type": "function", "function": {....
            // instead of [AVAILABLE_TOOLS][{"type":"function","function":{....
            // Therefore when inferencing, we need to add a space after the colon in the json string to match with the training data.
            schemaPrompt = schemaPrompt.Replace(":", ": ");
            schemaPrompt = schemaPrompt.Replace(",", ", ");
            sb.Append($"[AVAILABLE_TOOLS]{schemaPrompt}[/AVAILABLE_TOOLS]");
        }

        foreach (var message in secondSequence)
        {
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                ToolCallMessage toolCallMessage => BuildFromToolCallMessage(toolCallMessage),
                ToolCallResultMessage toolCallResultMessage => BuildFromToolCallResultMessage(toolCallResultMessage),
                ToolCallAggregateMessage toolCallAggregateMessage => BuildFromAggregrateToolCallMessage(toolCallAggregateMessage),
                TextMessage when message.GetRole() == Role.User && !string.IsNullOrEmpty(systemMessage) => $"[INST]{systemMessage}{Newline}{Newline}{content.Trim()}[/INST]",
                TextMessage when message.GetRole() == Role.User => $"[INST]{content.Trim()}[/INST]",
                TextMessage when message.GetRole() == Role.Assistant => $"{content.Trim()}</s>",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        return sb.ToString();
    }

    public string BuildPrompt(ChatHistory chatHistory)
    {
        throw new NotImplementedException();
    }

    private string BuildFromToolCallMessage(ToolCallMessage message)
    {
        var toolCalls = message.ToolCalls;
        if (toolCalls.Count() == 0)
        {
            return string.Empty;
        }
        else
        {
            var toolCallObjects = toolCalls.Select(tc =>
                new
                {
                    name = tc.FunctionName,
                    arguments = JsonObject.Parse(tc.FunctionArguments),
                    id = tc.ToolCallId,
                }
            );

            var toolCallJson = JsonSerializer.Serialize(toolCallObjects);
            return $"[TOOL_CALLS]{toolCallJson}</s>";
        }
    }

    private string BuildFromToolCallResultMessage(ToolCallResultMessage message)
    {
        var toolCallResults = message.ToolCalls;
        if (toolCallResults.Count() == 0)
        {
            return string.Empty;
        }
        else
        {
            var toolCallResultObjects = toolCallResults.Select(tc =>
                new
                {
                    id = tc.ToolCallId,
                    content = tc.Result,
                }
            );

            var toolCallResultJson = JsonSerializer.Serialize(toolCallResultObjects);
            return $"[TOOL_RESULTS]{toolCallResultJson}[/TOOL_RESULTS]";
        }
    }

    private string BuildFromAggregrateToolCallMessage(ToolCallAggregateMessage message)
    {
        var toolCallMessage = message.Message1;
        var toolCallResultMessage = message.Message2;

        var toolCall = BuildFromToolCallMessage(toolCallMessage);
        var toolCallResult = BuildFromToolCallResultMessage(toolCallResultMessage);

        return $"{toolCall}{toolCallResult}";
    }

    private JsonSchema BuildJsonSchemaFromFunctionContract(FunctionContract contract)
    {
        var requiredParameterNames = new List<string>();
        var propertiesSchemas = new Dictionary<string, JsonSchema>();
        var propertySchemaBuilder = new JsonSchemaBuilder().Type(SchemaValueType.Object);
        foreach (var param in contract.Parameters ?? [])
        {
            if (param.Name is null)
            {
                throw new InvalidOperationException("Parameter name cannot be null");
            }

            var schemaBuilder = new JsonSchemaBuilder().FromType(param.ParameterType ?? throw new ArgumentNullException(nameof(param.ParameterType)));
            if (param.Description != null)
            {
                schemaBuilder = schemaBuilder.Description(param.Description);
            }

            if (param.IsRequired)
            {
                requiredParameterNames.Add(param.Name);
            }

            var schema = schemaBuilder.Build();
            propertiesSchemas[param.Name] = schema;

        }
        propertySchemaBuilder = propertySchemaBuilder.Properties(propertiesSchemas);
        propertySchemaBuilder = propertySchemaBuilder.Required(requiredParameterNames);

        var jsonSchema = propertySchemaBuilder.Build();

        return jsonSchema;
    }
}
