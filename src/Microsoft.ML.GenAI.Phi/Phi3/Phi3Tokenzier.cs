// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

public class Phi3Tokenizer : Tokenizer
{
    private readonly SentencePieceBpe _tokenizer;
    private readonly bool _addPrecedingSpace;
    private readonly bool _addBeginningOfSentence;
    private readonly bool _addEndOfSentence;
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

    public Phi3Tokenizer(string modelPath,
        bool addPrecedingSpace = true,
        bool addBeginningOfSentence = true,
        bool addEndOfSentence = true)
    {
        var modelStream = File.OpenRead(modelPath);
        this._addPrecedingSpace = addPrecedingSpace;
        this._addBeginningOfSentence = addBeginningOfSentence;
        this._addEndOfSentence = addEndOfSentence;
        this._tokenizer = (SentencePieceBpe)Tokenizer.CreateLlama(modelStream, false, false);
    }

    public static Phi3Tokenizer FromPretrained(
        string folder,
        string modelName = "tokenizer.model")
    {
        return new Phi3Tokenizer(Path.Combine(folder, modelName));
    }

    public int BosId { get => this._tokenizer.BeginningOfSentenceId; }

    public int EosId { get => this._tokenizer.EndOfSentenceId; }

    public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        var tokens = new List<Token>();
        var normalizedText = new StringBuilder();
        var input = text.ToString();

        // step 1:
        // replace all special tokens to <unk>
        var re = new Regex($"{SystemSymbol.Replace("|", "\\|")}|{UserSymbol.Replace("|", "\\|")}|{AssistantSymbol.Replace("|", "\\|")}|{EndSymbol.Replace("|", "\\|")}");
        var matches = re.Matches(input);
        var matchesList = new List<string>();
        foreach (Match match in matches)
        {
            // replace the first special tokens with <unk>
            var specialToken = match.Value;
            var index = input.IndexOf(specialToken);
            var subString = input.Substring(0, index);
            var subTokens = this._tokenizer.Encode(subString, out var subNormalizeString, addBeginningOfSentence: false, addEndOfSentence: false, considerPreTokenization: considerPreTokenization, considerNormalization: considerNormalization).ToArray();
            normalizedText.Append(subNormalizeString);
            tokens.AddRange(subTokens);
            tokens.Add(new Token(this._specialTokenMap[specialToken], specialToken, (index, specialToken.Length)));
            input = input.Remove(0, index + specialToken.Length);
        }

        tokens.AddRange(this._tokenizer.Encode(input, out var normailzeString, addBeginningOfSentence: false, addEndOfSentence: false, considerPreTokenization: considerPreTokenization, considerNormalization: considerNormalization).ToArray());

        normalizedText.Append(normailzeString);
        normalizedString = normalizedText.ToString();

        return tokens.ToArray();
    }

    public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        var input = text.ToString();
        if (this._addPrecedingSpace)
        {
            //input = " " + input;
        }
        // step 1:
        // replace all special tokens to <unk>
        var re = new Regex($"{SystemSymbol.Replace("|", "\\|")}|{UserSymbol.Replace("|", "\\|")}|{AssistantSymbol.Replace("|", "\\|")}|{EndSymbol.Replace("|", "\\|")}");
        var matches = re.Matches(input);
        var matchesList = new List<string>();
        var tokens = new List<int>();
        foreach (Match match in matches)
        {
            var specialToken = match.Value;
            var index = input.IndexOf(specialToken);
            var subString = input.Substring(0, index);
            var subTokens = this._tokenizer.EncodeToIds(subString, addBeginningOfSentence: false, addEndOfSentence: false, considerPreTokenization: false, considerNormalization: true).ToArray();
            // remove the first sub Token as it will always be '_'
            tokens.AddRange(subTokens.Skip(1));
            tokens.Add(this._specialTokenMap[specialToken]);
            input = input.Remove(0, index + specialToken.Length);
        }

        tokens.AddRange(this._tokenizer.EncodeToIds(input, addBeginningOfSentence: false, addEndOfSentence: false, considerPreTokenization: false, considerNormalization: true).ToArray());

        return this._addBeginningOfSentence ? new int[] { this.BosId }.Concat(tokens).ToArray() : tokens.ToArray();
    }

    public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedText, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        var tokens = this.Encode(text, out normalizedText, considerPreTokenization, considerNormalization);

        var tokenIds = tokens.Select(x => x.Id).ToArray();

        textLength = normalizedText?.Length ?? 0;

        return tokenIds.Length > maxTokenCount ? tokenIds.Take(maxTokenCount).ToArray() : tokenIds;
    }

    public override int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        var tokens = this.EncodeToIds(text, considerPreTokenization, considerNormalization);

        return tokens.Count;
    }

    public override int IndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        return _tokenizer.IndexOfTokenCount(text, maxTokenCount, out normalizedString, out tokenCount, considerPreTokenization, considerNormalization);
    }

    public override int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? processedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
    {
        return _tokenizer.LastIndexOfTokenCount(text, maxTokenCount, out processedText, out tokenCount, considerPreTokenization, considerNormalization);
    }

    public override string? Decode(IEnumerable<int> ids)
    {
        // step 1
        // replace all special token ids to ukn ids
        var replacedIds = ids.SelectMany(id =>
        {
            if (this._specialTokenMap.ContainsValue(id))
            {
                var key = this._specialTokenMap.First(x => x.Value == id).Key;
                var ids = this._tokenizer.EncodeToIds(key, false, false, false, false);
                var recoverKey = this._tokenizer.Decode(ids) ?? throw new Exception("Failed to decode ids");
                return ids;
            }
            else
            {
                return new List<int> { id };
            }
        });

        var str = this._tokenizer.Decode(replacedIds) ?? throw new Exception("Failed to decode ids");

        return str;

        //var tokens = new List<string>();
        //foreach (var id in ids)
        //{
        //    if (_specialTokenMap.ContainsValue(id))
        //    {
        //        tokens.Add(_specialTokenMap.First(x => x.Value == id).Key);
        //    }
        //    else
        //    {
        //        tokens.Add(this._tokenizer.MapIdToToken(id) ?? throw new Exception("Failed to map id to token"));
        //    }
        //}

        //if (this._addBeginningOfSentence)
        //{
        //    tokens = tokens[1..].ToList();
        //}

        //var str = string.Join("", tokens);

        //// replace Dummy with whitespace
        //str = str.Replace(SentencePieceNormalizer.DummyPrefix, ' ');

        //if (this._addPrecedingSpace)
        //{
        //    str = str.TrimStart(' ');
        //}

        //return str;
    }

    public override int? MapTokenToId(ReadOnlySpan<char> token)
    {
        // check if token in special tokens
        var tokenStr = token.ToString();
        if (_specialTokenMap.ContainsKey(tokenStr))
        {
            return _specialTokenMap[tokenStr];
        }

        return _tokenizer.MapTokenToId(token);
    }

    public override string? MapIdToToken(int id)
    {
        if (_specialTokenMap.ContainsValue(id))
        {
            return _specialTokenMap.First(x => x.Value == id).Key;
        }

        return _tokenizer.MapIdToToken(id);
    }
}
