// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.LLaMA;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
public class LlamaTokenizerHelper
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    /// <summary>
    /// https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/tokenizer.json#pre_tokenizer.pretokenizers.pattern
    /// </summary>
    private const string _re = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    /// <summary>
    /// https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/tokenizer.json#added_tokens
    /// </summary>
    private static readonly Dictionary<string, int> _specialTokens = new()
    {
        { "<|begin_of_text|>", 128000 },
        { "<|end_of_text|>", 128001 },
        { "<|finetune_right_pad_id|>", 128004 },
        { "<|start_header_id|>", 128006 },
        { "<|end_header_id|>", 128007 },
        { "<|eom_id|>", 128008 },
        { "<|eot_id|>", 128009 },
        { "<|system|>", 32006 },
        { "<|user|>", 32010 },
        { "<|assistant|>", 32001 },
        { "<|end|>", 32007 }
    };

    /// <summary>
    /// Create <see cref="TiktokenTokenizer"/> from tokenizer model file.
    /// </summary>
    /// <param name="modelWeightFolder">path to tokenizer model folder</param>
    /// <param name="modelFile">tokenizer model file name</param>
    public static TiktokenTokenizer FromPretrained(
        string modelWeightFolder,
        string modelFile = "tokenizer.model")
    {
        var modelFilePath = Path.Join(modelWeightFolder, modelFile);
        var preTokenizer = new RegexPreTokenizer(new Regex(_re), _specialTokens);
        return TiktokenTokenizer.Create(File.OpenRead(modelFilePath), preTokenizer, normalizer: null, specialTokens: _specialTokens);
    }
}
