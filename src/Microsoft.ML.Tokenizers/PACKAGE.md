## About

Microsoft.ML.Tokenizers supports various the implmentation of the tokenization used in the NLP transforms.

## Key Features

* Extensisble tokenizer architecture that allows for specialization of Normalizer, PreTokenizer, Model/Encoder, Decoder
* BPE - Byte pair encoding model
* English Roberta model
* Tiktoken model

## How to Use

```c#
using Microsoft.ML.Tokenizers;

// initialize the tokenizer for `gpt-4` model, downloading data files
Tokenizer tokenizer = await Tiktoken.CreateByModelNameAsync("gpt-4");

string source = "Text tokenization is the process of splitting a string into a list of tokens.";

Console.WriteLine($"Tokens: {tokenizer.CountTokens(source)}");
// print: Tokens: 16

var trimIndex = tokenizer.LastIndexOfTokenCount(source, 5, out string processedText, out _);
Console.WriteLine($"5 tokens from end: {processedText.Substring(trimIndex)}");
// 5 tokens from end:  a list of tokens.

trimIndex = tokenizer.IndexOfTokenCount(source, 5, out processedText, out _);
Console.WriteLine($"5 tokens from start: {processedText.Substring(0, trimIndex)}");
// 5 tokens from start: Text tokenization is the
```

## Main Types

The main types provided by this library are:

* `Microsoft.ML.Tokenizers.Tokenizer`
* `Microsoft.ML.Tokenizers.Bpe`
* `Microsoft.ML.Tokenizers.EnglishRoberta`
* `Microsoft.ML.Tokenizers.TikToken`
* `Microsoft.ML.Tokenizers.TokenizerDecoder`
* `Microsoft.ML.Tokenizers.Normalizer`
* `Microsoft.ML.Tokenizers.PreTokenizer`

## Additional Documentation

* [Conceptual documentation](TODO)
* [API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers)

## Related Packages

<!-- The related packages associated with this package -->

## Feedback & Contributing

Microsoft.ML.Tokenizers is released as open source under the [MIT license](https://licenses.nuget.org/MIT). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/dotnet/machinelearning).