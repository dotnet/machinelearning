// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Phi;
public interface ITokenizer
{
    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input);

    public int[] Encode(string input, bool bos, bool eos);
}

/// <summary>
/// Copied from https://github.com/LittleLittleCloud/Torchsharp-llama/blob/main/ITokenizer.cs
/// </summary>
public class LLama2Tokenizer : ITokenizer
{
    private readonly SentencePieceBpe _tokenizer;
    private readonly bool _addPrecedingSpace;
    private const string SystemSymbol = "<|system|>";
    private const string UserSymbol = "<|user|>";
    private const string AssistantSymbol = "<|assistant|>";
    private const string EndSymbol = "<|end|>";
    private const int SystemSymbolId = 32006;
    private const int UserSymbolId = 32010;
    private const int AssistantSymbolId = 32001;
    private const int EndSymbolId = 32007;
    private readonly Dictionary<string, int> _specialTokenMap = new Dictionary<string, int>
    {
        { SystemSymbol, SystemSymbolId },
        { UserSymbol, UserSymbolId },
        { AssistantSymbol, AssistantSymbolId },
        { EndSymbol, EndSymbolId }
    };

    public LLama2Tokenizer(string modelPath, bool addPrecedingSpace = true)
    {
        var modelStream = File.OpenRead(modelPath);
        this._addPrecedingSpace = addPrecedingSpace;
        this._tokenizer = (SentencePieceBpe)Tokenizer.CreateLlama(modelStream, false, false);

        // use reflection to set the readonly ByteFallback property to false
        //var backingField = typeof(SentencePieceBpe).GetField("<ByteFallback>k__BackingField", BindingFlags.NonPublic | BindingFlags.Instance);
        //backingField.SetValue(this.tokenizer, false);
    }
    //public LLama2Tokenizer(string vocabPath, string mergesPath, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    //{
    //    this.BosId = startToken;
    //    this.EosId = endToken;
    //    this.addPrecedingSpace = addPrecedingSpace;
    //    this.PadId = padToken;
    //    var bpe = new Bpe(vocabPath, mergesPath);
    //    this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
    //    var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
    //    this.tokenizer.Decoder = decoder;
    //}

    //public LLama2Tokenizer(Dictionary<string, int> vocab, List<string> merges, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    //{
    //    this.BosId = startToken;
    //    this.EosId = endToken;
    //    this.addPrecedingSpace = addPrecedingSpace;
    //    this.PadId = padToken;
    //    // save vocab to vocab-temp.json
    //    var vocabTempPath = "vocab-temp.json";
    //    var json = JsonSerializer.Serialize(vocab);
    //    File.WriteAllText(vocabTempPath, json);

    //    // save merges to merges-temp.txt
    //    var mergesTempPath = "merges-temp.txt";
    //    // filter out merges that contain newline character because it will cause error in BPE
    //    merges = merges.Where(x => !x.Contains('\r')).ToList();
    //    File.WriteAllLines(mergesTempPath, merges);

    //    var bpe = new Bpe(vocabTempPath, mergesTempPath);
    //    this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
    //    var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
    //    this.tokenizer.Decoder = decoder;

    //    // delete temp files
    //    File.Delete(vocabTempPath);
    //    File.Delete(mergesTempPath);
    //}

    public static LLama2Tokenizer FromPretrained(
        string folder,
        string modelName = "tokenizer.model")
    {
        return new LLama2Tokenizer(Path.Combine(folder, modelName));
    }

    //public static LLama2Tokenizer FromPretrained(
    //    string folder,
    //    string tokenizerJsonPath = "tokenizer.json",
    //    string specialTokensMapPath = "special_tokens_map.json"
    //)
    //{
    //    tokenizerJsonPath = Path.Combine(folder, tokenizerJsonPath);
    //    var json = File.ReadAllText(tokenizerJsonPath);
    //    var jsonDocument = JsonDocument.Parse(json);
    //    // vocab: .model.vocab
    //    var vocabNode = jsonDocument.RootElement.GetProperty("model").GetProperty("vocab");

    //    // to Dictionary<string, int>
    //    var vocab = new Dictionary<string, int>();
    //    foreach (var item in vocabNode.EnumerateObject())
    //    {
    //        vocab[item.Name] = item.Value.GetInt32();
    //    }

    //    // added tokens: .added_tokens
    //    var addedTokensNode = jsonDocument.RootElement.GetProperty("added_tokens");
    //    foreach (var item in addedTokensNode.EnumerateArray())
    //    {
    //        // get id from item.id
    //        var id = item.GetProperty("id").GetInt32();
    //        var content = item.GetProperty("content").GetString()!;
    //        vocab[content] = id;
    //    }

    //    // merges: .model.merges
    //    var mergesNode = jsonDocument.RootElement.GetProperty("model").GetProperty("merges");
    //    // merges: List<string>
    //    var merges = new List<string>();
    //    foreach (var item in mergesNode.EnumerateArray())
    //    {
    //        merges.Add(item.GetString()!);
    //    }

    //    int startToken = 1, endToken = 2, padToken = -1;
    //    var specialTokenJsonPath = Path.Combine(folder, specialTokensMapPath);
    //    if (File.Exists(specialTokenJsonPath))
    //    {
    //        var specialTokenJson = File.ReadAllText(specialTokenJsonPath);
    //        var specialTokenMapDocument = JsonDocument.Parse(specialTokenJson);

    //        // retrieve bos_token, eos_token, pad_token if exists
    //        if (specialTokenMapDocument.RootElement.TryGetProperty("bos_token", out var bosTokenNode))
    //        {
    //            var bos_token_content = bosTokenNode.GetProperty("content").GetString()!;
    //            startToken = vocab[bos_token_content];
    //        }

    //        if (specialTokenMapDocument.RootElement.TryGetProperty("eos_token", out var eosTokenNode))
    //        {
    //            var eos_token_content = eosTokenNode.GetProperty("content").GetString()!;
    //            endToken = vocab[eos_token_content];
    //        }

    //        if (specialTokenMapDocument.RootElement.TryGetProperty("pad_token", out var padTokenNode))
    //        {
    //            var pad_token_content = padTokenNode.GetProperty("content").GetString()!;
    //            padToken = vocab[pad_token_content];
    //        }
    //    }

    //    return new LLama2Tokenizer(vocab, merges, padToken: padToken, addPrecedingSpace: false, startToken: startToken, endToken: endToken);
    //}

    //public int VocabSize => this.tokenizer..GetVocabSize();

    public int PadId { get => this._tokenizer.UnknownId; }

    public int BosId { get => this._tokenizer.BeginningOfSentenceId; }

    public int EosId { get => this._tokenizer.EndOfSentenceId; }

    public string Decode(int[] input)
    {
        var str = this._tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this._addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }

    public int[] Encode(string input, bool bos, bool eos)
    {
        // step 1:
        // replace all special tokens to <unk>
        var re = new Regex($"{SystemSymbol.Replace("|", "\\|")}|{UserSymbol.Replace("|", "\\|")}|{AssistantSymbol.Replace("|", "\\|")}|{EndSymbol.Replace("|", "\\|")}");
        var matches = re.Matches(input);
        var matchesList = new List<string>();
        var tokens = new List<int>();
        foreach (Match match in matches)
        {
            // replace the first special tokens with <unk>
            var specialToken = match.Value;
            var index = input.IndexOf(specialToken);
            var subString = input.Substring(0, index);
            var subTokens = this._tokenizer.EncodeToIds(subString, addBeginningOfSentence: false, addEndOfSentence: false).ToArray();
            tokens.AddRange(subTokens);
            tokens.Add(this._specialTokenMap[specialToken]);
            input = input.Remove(0, index + specialToken.Length);
        }

        tokens.AddRange(this._tokenizer.EncodeToIds(input, addBeginningOfSentence: false, addEndOfSentence: false).ToArray());
        if (bos)
        {
            tokens.Insert(0, this.BosId);
        }
        if (eos)
        {
            tokens.Add(this.EosId);
        }


        return tokens.ToArray();
    }
}
