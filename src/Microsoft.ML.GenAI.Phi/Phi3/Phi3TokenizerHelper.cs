// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

/// <summary>
/// The utility class to create tokenizer for phi-3 model.
/// </summary>
public class Phi3TokenizerHelper
{
    private const string SystemSymbol = "<|system|>";
    private const string UserSymbol = "<|user|>";
    private const string AssistantSymbol = "<|assistant|>";
    private const string EndSymbol = "<|end|>";
    private const int SystemSymbolId = 32006;
    private const int UserSymbolId = 32010;
    private const int AssistantSymbolId = 32001;
    private const int EndSymbolId = 32007;

    public static LlamaTokenizer FromPretrained(
        string modelPath,
        string systemSymbol = SystemSymbol,
        string userSymbol = UserSymbol,
        string assistantSymbol = AssistantSymbol,
        string endSymbol = EndSymbol,
        int systemSymbolId = SystemSymbolId,
        int userSymbolId = UserSymbolId,
        int assistantSymbolId = AssistantSymbolId,
        int endSymbolId = EndSymbolId,
        bool addPrecedingSpace = true)
    {
        var modelStream = File.OpenRead(modelPath);

        var llamaTokenizer = LlamaTokenizer.Create(
            modelStream,
            addPrecedingSpace,
            specialTokens: new Dictionary<string, int>
            {
                { systemSymbol, systemSymbolId },
                { userSymbol, userSymbolId },
                { assistantSymbol, assistantSymbolId },
                { endSymbol, endSymbolId }
            });

        return llamaTokenizer;
    }
}
