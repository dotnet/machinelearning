// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Mistral;

public class MistralTokenizerHelper
{
    private const string UnknownSymbol = "<unk>";
    private const int UnknownSymbolId = 0;
    private const string StartSymbol = "<s>";
    private const int StartSymbolId = 1;
    private const string EndSymbol = "</s>";
    private const int EndSymbolId = 2;
    private const string StartInstructionSymbol = "[INST]";
    private const int StartInstructionSymbolId = 3;
    private const string EndInstructionSymbol = "[/INST]";
    private const int EndInstructionSymbolId = 4;
    private const string ToolCallSymbol = "[TOOL_CALLS]";
    private const int ToolCallSymbolId = 5;
    private const string StartAvailableToolsSymbol = "[AVAILABLE_TOOLS]";
    private const int StartAvailableToolsSymbolId = 6;
    private const string EndAvailableToolsSymbol = "[/AVAILABLE_TOOLS]";
    private const int EndAvailableToolsSymbolId = 7;
    private const string StartToolResultSymbol = "[TOOL_RESULTS]";
    private const int StartToolResultSymbolId = 8;
    private const string EndToolResultSymbol = "[/TOOL_RESULTS]";
    private const int EndToolResultSymbolId = 9;

    public static LlamaTokenizer FromPretrained(
        string modelWeightFolder,
        string modelName = "tokenizer.model.v3",
        string unknownSymbol = UnknownSymbol,
        int unknownSymbolId = 0,
        string startSymbol = StartSymbol,
        int startSymbolId = 1,
        string endSymbol = EndSymbol,
        int endSymbolId = 2,
        string startInstructionSymbol = StartInstructionSymbol,
        int startInstructionSymbolId = 3,
        string endInstructionSymbol = EndInstructionSymbol,
        int endInstructionSymbolId = 4,
        string toolCallSymbol = ToolCallSymbol,
        int toolCallSymbolId = 5,
        string startAvailableToolsSymbol = StartAvailableToolsSymbol,
        int startAvailableToolsSymbolId = 6,
        string endAvailableToolsSymbol = EndAvailableToolsSymbol,
        int endAvailableToolsSymbolId = 7,
        string startToolResultSymbol = StartToolResultSymbol,
        int startToolResultSymbolId = 8,
        string endToolResultSymbol = EndToolResultSymbol,
        int endToolResultSymbolId = 9,
        bool addPrecedingSpace = true)
    {
        var specialTokens = new Dictionary<string, int>
        {
            { startSymbol, startSymbolId },
            { endSymbol, endSymbolId },
            { startInstructionSymbol, startInstructionSymbolId },
            { endInstructionSymbol, endInstructionSymbolId },
            { toolCallSymbol, toolCallSymbolId },
            { startAvailableToolsSymbol, startAvailableToolsSymbolId },
            { endAvailableToolsSymbol, endAvailableToolsSymbolId },
            { startToolResultSymbol, startToolResultSymbolId },
            { endToolResultSymbol, endToolResultSymbolId }
        };

        return FromPretrained(
            modelWeightFolder,
            modelName,
            specialTokens,
            addPrecedingSpace);
    }

    public static LlamaTokenizer FromPretrained(
        string modelWeightFolder,
        string modelName,
        Dictionary<string, int> specialTokens,
        bool addPrecedingSpace = true)
    {
        var modelPath = Path.Combine(modelWeightFolder, modelName);
        var modelStream = File.OpenRead(modelPath);

        var llamaTokenizer = LlamaTokenizer.Create(
            modelStream,
            addPrecedingSpace,
            specialTokens: specialTokens);

        return llamaTokenizer;
    }
}
