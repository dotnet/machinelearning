# Porting to Microsoft.ML.Tokenizers

This guide provides general guidance on how to migrate from various tokenizer libraries to `Microsoft.ML.Tokenizers`.

## Microsoft.DeepDev.TokenizerLib

### API Guidance

| Microsoft.DeepDev.TokenizerLib | Microsoft.ML.Tokenizers
| --- | --- |
| [TikTokenizer](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/TikTokenizer.cs#L20) | Tokenizer |
| [ITokenizer](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/ITokenizer.cs#L7) | Tokenizer |
| [TokenizerBuilder](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/TokenizerBuilder.cs#L14) | TiktokenTokenizer.CreateForModel <br> TiktokenTokenizer.CreateForModel(Async/Stream) user provided file stream |

### General Guidance

- To avoid embedding the tokenizer's vocabulary files in the code assembly or downloading them at runtime when using one of the standard Tiktoken vocabulary files, utilize the `TiktokenTokenizer.CreateForModel` function. The [table](https://github.com/dotnet/machinelearning/blob/4d5317e8090e158dc7c3bc6c435926ccf1cbd8e2/src/Microsoft.ML.Tokenizers/Model/Tiktoken.cs#L683-L734) lists the mapping of model names to the corresponding vocabulary files used with each model. This table offers clarity regarding the vocabulary file linked with each model, alleviating users from the concern of carrying or downloading such vocabulary files if they utilize one of the models listed.
- Avoid hard-coding tiktoken regexes and special tokens.  Instead use the appropriate Tiktoken.`TiktokenTokenizer.CreateForModel/Async` method to create the tokenizer using the model name, or a provided stream.
- Avoid doing encoding if you need the token count or encoded Ids. Instead use `TiktokenTokenizer.CountTokens` for getting the token count and `TiktokenTokenizer.EncodeToIds` for getting the encode ids.
- Avoid doing encoding if all you need is to truncate to a token budget.  Instead use `TiktokenTokenizer.GetIndexByTokenCount` or `GetIndexByTokenCountFromEnd` to find the index to truncate from the start or end of a string, respectively.

## Hugging Face tokenizers

This section shows how to get the same token IDs in .NET as the Hugging Face `transformers` and `tokenizers` Python packages. The examples need `Microsoft.ML.Tokenizers` 3.0 or later.

`Microsoft.ML.Tokenizers` has no equivalent of `AutoTokenizer.from_pretrained`. It doesn't download files from the Hugging Face Hub or detect which tokenizer a model uses. Instead, you download the tokenizer files from the model repository and create the matching tokenizer, as shown below. A factory that does this for you is tracked in [#7735](https://github.com/dotnet/machinelearning/issues/7735).

### Choose the tokenizer

Check which tokenizer class the model uses. In Python, `type(AutoTokenizer.from_pretrained(name)).__name__` returns it, and some models also list it as `tokenizer_class` in `tokenizer_config.json`. The class name can end in `Fast`, such as `BertTokenizerFast`. Then check `model.type` in the model's `tokenizer.json`. Some older `tokenizer.json` files don't have `model.type`. In those files, a model with `merges` is BPE, a model whose `vocab` is a list of `[piece, score]` pairs is Unigram, and a model with `max_input_chars_per_word` is WordPiece.

| Hugging Face tokenizer | `model.type` | Microsoft.ML.Tokenizers | Files from the model repository |
| --- | --- | --- | --- |
| `BertTokenizer` (for example, BERT and MiniLM models) | `WordPiece` | `BertTokenizer` | `vocab.txt` |
| `GPT2Tokenizer`, `RobertaTokenizer`, and other BPE tokenizers with a `ByteLevel` pre-tokenizer | `BPE` | `BpeTokenizer` | `vocab.json`, `merges.txt` |
| `GPT2Tokenizer` for GPT-2 | `BPE` | `TiktokenTokenizer` | None. Install the `Microsoft.ML.Tokenizers.Data.Gpt2` package. |
| `CodeGenTokenizer` (CodeGen and Phi-2) | `BPE` | `CodeGenTokenizer`, `Phi2Tokenizer` | `vocab.json`, `merges.txt` |
| `LlamaTokenizer` (Llama 2 and models that use its tokenizer, such as TinyLlama) | `BPE` | `LlamaTokenizer` | `tokenizer.model` |
| `T5Tokenizer`, `XLMRobertaTokenizer`, and other SentencePiece Unigram tokenizers | `Unigram` | `SentencePieceTokenizer` | `tokenizer.json` |

There's no equivalent for `WordLevel` tokenizers. If a model doesn't include the files in this table, see [Models with only tokenizer.json](#models-with-only-tokenizerjson).

### Examples

The examples assume that the files are in the current directory. Each example gives the same IDs as `tokenizer(text)["input_ids"]` in Python for the model it names, except where a note says otherwise.

#### BERT

```csharp
using Microsoft.ML.Tokenizers;

// google-bert/bert-base-uncased. The settings match the "normalizer" section of tokenizer.json.
BertTokenizer tokenizer = BertTokenizer.Create("vocab.txt", new BertOptions
{
    LowerCaseBeforeTokenization = true, // "lowercase": true
    RemoveNonSpacingMarks = true,       // "strip_accents": null, which follows "lowercase"
    IndividuallyTokenizeCjk = true,     // "handle_chinese_chars": true
});

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // [101, 7592, 1010, 2088, 999, 102]
```

To encode a pair of texts, such as a question and a passage:

```csharp
IReadOnlyList<int> question = tokenizer.EncodeToIds("What is ML.NET?", addSpecialTokens: false);
IReadOnlyList<int> passage = tokenizer.EncodeToIds("ML.NET is a machine learning framework for .NET.", addSpecialTokens: false);

IReadOnlyList<int> inputIds = tokenizer.BuildInputsWithSpecialTokens(question, passage); // [CLS] question [SEP] passage [SEP]
IReadOnlyList<int> tokenTypeIds = tokenizer.CreateTokenTypeIdsFromSequences(question, passage); // 0 for the question, 1 for the passage
```

- For cased models, such as `google-bert/bert-base-cased`, set `LowerCaseBeforeTokenization` and `RemoveNonSpacingMarks` to `false`.
- `EncodeToIds` adds `[CLS]` and `[SEP]` when you call it on a `BertTokenizer`. `EncodeToTokens`, `CountTokens`, and calls through the base `Tokenizer` type don't add them.

#### GPT-2 and RoBERTa

```csharp
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

// FacebookAI/roberta-base. The special tokens and their IDs come from "added_tokens" in tokenizer.json.
Dictionary<string, int> specialTokens = new()
{
    ["<s>"] = 0, ["<pad>"] = 1, ["</s>"] = 2, ["<unk>"] = 3, ["<mask>"] = 50264,
};

// The pattern that the Hugging Face ByteLevel pre-tokenizer uses to split text.
Regex byteLevelPattern = new(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");

BpeTokenizer tokenizer = BpeTokenizer.Create(new BpeOptions("vocab.json", "merges.txt")
{
    ByteLevel = true,
    PreTokenizer = new RegexPreTokenizer(byteLevelPattern, specialTokens),
    SpecialTokens = specialTokens,
    BeginningOfSentenceToken = "<s>",
    EndOfSentenceToken = "</s>",
});

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // [0, 31414, 6, 232, 328, 2]
```

For `openai-community/gpt2`, the only special token is `<|endoftext|>` (50256), and GPT-2 doesn't add `<s>` and `</s>`, so leave out `BeginningOfSentenceToken` and `EndOfSentenceToken`. Or use `TiktokenTokenizer`, which needs the `Microsoft.ML.Tokenizers.Data.Gpt2` package instead of the model files:

```csharp
using Microsoft.ML.Tokenizers;

TiktokenTokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt2");

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // [15496, 11, 995, 0]
```

- Always set `PreTokenizer` when `ByteLevel` is `true`. Otherwise, spaces and line breaks are dropped. Pass the special tokens to the pre-tokenizer too, so that it keeps them whole.
- Hugging Face can remove the space next to some added tokens, such as the space before RoBERTa's `<mask>`. `BpeTokenizer` keeps that space, so the IDs differ next to those tokens.
- Use `BpeTokenizer` rather than `EnglishRobertaTokenizer` for these models. `EnglishRobertaTokenizer` doesn't always produce the same IDs. For example, it gives different IDs for accented characters and drops CJK characters and emoji.

#### T5 and XLM-RoBERTa

```csharp
using Microsoft.ML.Tokenizers;

// google-t5/t5-small or FacebookAI/xlm-roberta-base
using Stream json = File.OpenRead("tokenizer.json");

// Add the tokens from the "post_processor" section, as Hugging Face does by default:
// </s> after the text for T5, and <s> before and </s> after the text for XLM-RoBERTa.
SentencePieceTokenizer tokenizer = SentencePieceTokenizer.CreateFromTokenizerJson(json, addBeginningOfSentence: true, addEndOfSentence: true);

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // T5: [8774, 6, 296, 55, 1]
```

- Load `tokenizer.json` rather than the SentencePiece model file. XLM-RoBERTa's `sentencepiece.bpe.model` gives different IDs, and T5's `spiece.model` can't be loaded ([#7756](https://github.com/dotnet/machinelearning/issues/7756)).
- A run of characters that aren't in the vocabulary gives one `<unk>` for each character, while Hugging Face gives a single `<unk>` ([#7859](https://github.com/dotnet/machinelearning/issues/7859)).
- `CreateFromTokenizerJson` doesn't read the `decoder` section ([#7660](https://github.com/dotnet/machinelearning/issues/7660)) and doesn't apply the templates for pairs of texts. Its API documentation lists the settings that it reads.

#### Llama 2

```csharp
using Microsoft.ML.Tokenizers;

// TinyLlama/TinyLlama-1.1B-Chat-v1.0, which uses the Llama 2 tokenizer.
using Stream model = File.OpenRead("tokenizer.model");

// LlamaTokenizer adds <s>, like Hugging Face. Pass the special tokens so that they're found in the text.
LlamaTokenizer tokenizer = LlamaTokenizer.Create(model, specialTokens: new Dictionary<string, int> { ["<unk>"] = 0, ["<s>"] = 1, ["</s>"] = 2 });

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // [1, 15043, 29892, 3186, 29991]
```

For text that starts with spaces or comes right after a special token, `transformers` and `tokenizers` don't always give the same IDs for Llama 2 tokenizers. Compare such text with the package that you use in Python.

#### CodeGen and Phi-2

```csharp
using Microsoft.ML.Tokenizers;

// Salesforce/codegen-350M-mono. For microsoft/phi-2, call Phi2Tokenizer.Create instead.
using Stream vocab = File.OpenRead("vocab.json");
using Stream merges = File.OpenRead("merges.txt");

CodeGenTokenizer tokenizer = CodeGenTokenizer.Create(vocab, merges);

IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!"); // [15496, 11, 995, 0]
```

### Common operations

| `transformers` | `Microsoft.ML.Tokenizers` |
| --- | --- |
| `tokenizer(text)["input_ids"]` | `EncodeToIds(text)` |
| `add_special_tokens=False` | `BertTokenizer`: `EncodeToIds(text, addSpecialTokens: false)`<br>`SentencePieceTokenizer` and `LlamaTokenizer`: `EncodeToIds(text, addBeginningOfSentence: false, addEndOfSentence: false)`<br>`BpeTokenizer`: don't set `BeginningOfSentenceToken` and `EndOfSentenceToken` |
| `tokenize(text)`, `return_offsets_mapping=True` | `EncodeToTokens(text, out string? normalizedText)` returns each token's `Value` and `Offset`. When `normalizedText` isn't `null`, the offsets refer to it rather than to `text`. |
| `decode(ids, skip_special_tokens=True)` | `SentencePieceTokenizer` and `LlamaTokenizer`: `Decode(ids)`<br>`BpeTokenizer`: `Decode(ids, considerSpecialTokens: false)`<br>`BertTokenizer`: `Decode(ids, skipSpecialTokens: true)`<br>Spaces around punctuation can differ from Hugging Face. |
| `truncation=True, max_length=n` | `BertTokenizer`: `EncodeToIds(text, n, out _, out _)`<br>Other tokenizers drop the end token, such as `</s>`, when they cut the text. See the example after this table. |
| `padding`, `attention_mask`, `return_tensors` | No equivalent. Create these arrays from the IDs. |

For a `SentencePieceTokenizer` that adds `</s>`, this function keeps room for it, like `truncation=True`:

```csharp
static List<int> EncodeWithTruncation(SentencePieceTokenizer tokenizer, string text, int maxLength) =>
    [.. tokenizer.EncodeToIds(text, addBeginningOfSentence: true, addEndOfSentence: false, maxLength - 1, out _, out _), tokenizer.EndOfSentenceId];
```

### Models with only tokenizer.json

Some models only include `tokenizer.json`. `SentencePieceTokenizer.CreateFromTokenizerJson` reads Unigram models, as shown earlier. For WordPiece and BPE models, read the settings from `tokenizer.json` yourself.

For WordPiece, write `model.vocab` in the `vocab.txt` format, and take the `BertOptions` settings from the `normalizer` section:

```csharp
using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

// google-bert/bert-base-uncased, google-bert/bert-base-cased, or sentence-transformers/all-MiniLM-L6-v2
using JsonDocument json = JsonDocument.Parse(File.ReadAllText("tokenizer.json"));
JsonElement vocab = json.RootElement.GetProperty("model").GetProperty("vocab");
JsonElement normalizer = json.RootElement.GetProperty("normalizer");

// vocab.txt has one token per line, and the line number is the token ID.
string[] lines = new string[vocab.EnumerateObject().Max(token => token.Value.GetInt32()) + 1];
foreach (JsonProperty token in vocab.EnumerateObject())
{
    lines[token.Value.GetInt32()] = token.Name;
}

bool lowercase = normalizer.GetProperty("lowercase").GetBoolean();
JsonElement stripAccents = normalizer.GetProperty("strip_accents");

using MemoryStream vocabStream = new(Encoding.UTF8.GetBytes(string.Join("\n", lines)));
BertTokenizer tokenizer = BertTokenizer.Create(vocabStream, new BertOptions
{
    LowerCaseBeforeTokenization = lowercase,
    RemoveNonSpacingMarks = stripAccents.ValueKind == JsonValueKind.Null ? lowercase : stripAccents.GetBoolean(),
    IndividuallyTokenizeCjk = normalizer.GetProperty("handle_chinese_chars").GetBoolean(),
});
```

For byte-level BPE, create `BpeOptions` from `model.vocab`, `model.merges`, and `added_tokens`:

```csharp
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

// openai-community/gpt2. For FacebookAI/roberta-base, also pass "<s>" and "</s>".
BpeTokenizer tokenizer = CreateByteLevelBpeTokenizer("tokenizer.json");

static BpeTokenizer CreateByteLevelBpeTokenizer(string path, string? beginningOfSentenceToken = null, string? endOfSentenceToken = null)
{
    using JsonDocument json = JsonDocument.Parse(File.ReadAllText(path));
    JsonElement root = json.RootElement;
    JsonElement model = root.GetProperty("model");
    JsonElement preTokenizer = root.GetProperty("pre_tokenizer");

    // Fail rather than return a tokenizer that gives different IDs.
    if (root.GetProperty("normalizer").ValueKind != JsonValueKind.Null ||
        preTokenizer.ValueKind != JsonValueKind.Object ||
        preTokenizer.GetProperty("type").GetString() != "ByteLevel" ||
        preTokenizer.GetProperty("add_prefix_space").GetBoolean())
    {
        throw new NotSupportedException("Only GPT-2 style byte-level BPE files are supported.");
    }

    Dictionary<string, int> vocab = model.GetProperty("vocab").EnumerateObject()
        .ToDictionary(token => token.Name, token => token.Value.GetInt32());

    // Older files store each merge as "left right", and newer files as ["left", "right"].
    List<string> merges = model.GetProperty("merges").EnumerateArray()
        .Select(merge => merge.ValueKind == JsonValueKind.String ? merge.GetString()! : $"{merge[0].GetString()} {merge[1].GetString()}")
        .ToList();

    // Hugging Face finds the added tokens in the text before it splits the text.
    Dictionary<string, int> specialTokens = root.GetProperty("added_tokens").EnumerateArray()
        .ToDictionary(token => token.GetProperty("content").GetString()!, token => token.GetProperty("id").GetInt32());

    Regex byteLevelPattern = new(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");

    return BpeTokenizer.Create(new BpeOptions(vocab)
    {
        Merges = merges,
        SpecialTokens = specialTokens,
        ByteLevel = true,
        PreTokenizer = new RegexPreTokenizer(byteLevelPattern, specialTokens),
        BeginningOfSentenceToken = beginningOfSentenceToken,
        EndOfSentenceToken = endOfSentenceToken,
    });
}
```

This function only reads files like GPT-2's, which have a `ByteLevel` pre-tokenizer and no normalizer. If `pre_tokenizer` is a `Sequence` of `Split` and `ByteLevel` steps, as in DeepSeek-R1, split the text with the `Split` patterns instead of the GPT-2 pattern and combine them with `CompositePreTokenizer`. `CreateBpeTokenizerFromJson` in [BpeTests.cs](../../test/Microsoft.ML.Tokenizers.Tests/BpeTests.cs) shows how. `BpeTokenizer` has no options for `byte_fallback`, `ignore_merges`, or the `lstrip` and `rstrip` settings of added tokens, so compare the IDs with Hugging Face when a file uses them.

### Check the results

Compare the IDs with Hugging Face for text that's typical for your app, including line breaks, non-ASCII characters, and any special tokens that it can contain. For example:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
print(tokenizer("Hello, world!")["input_ids"])  # [101, 7592, 1010, 2088, 999, 102]
```

Known differences are tracked in the [open Tokenizers issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aissue%20is%3Aopen%20label%3Aarea-Tokenizers).
