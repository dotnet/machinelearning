## About

Microsoft.ML.Tokenizers supports various the implementation of the tokenization used in the NLP transforms.

## Key Features

* Extensible tokenizer architecture that allows for specialization of Normalizer, PreTokenizer, Model/Encoder, Decoder
* BPE - Byte pair encoding model
* English Roberta model
* Tiktoken model
* Llama model
* Phi2 model

## How to Use

```c#
using Microsoft.ML.Tokenizers;
using System.Net.Http;
using System.IO;

//
// Using Tiktoken Tokenizer
//

// initialize the tokenizer for `gpt-4` model
Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");

string source = "Text tokenization is the process of splitting a string into a list of tokens.";

Console.WriteLine($"Tokens: {tokenizer.CountTokens(source)}");
// print: Tokens: 16

var trimIndex = tokenizer.GetIndexByTokenCountFromEnd(source, 5, out string processedText, out _);
Console.WriteLine($"5 tokens from end: {processedText.Substring(trimIndex)}");
// 5 tokens from end:  a list of tokens.

trimIndex = tokenizer.GetIndexByTokenCount(source, 5, out processedText, out _);
Console.WriteLine($"5 tokens from start: {processedText.Substring(0, trimIndex)}");
// 5 tokens from start: Text tokenization is the

IReadOnlyList<int> ids = tokenizer.EncodeToIds(source);
Console.WriteLine(string.Join(", ", ids));
// prints: 1199, 4037, 2065, 374, 279, 1920, 315, 45473, 264, 925, 1139, 264, 1160, 315, 11460, 13

//
// Using Llama Tokenizer
//

// Open stream of remote Llama tokenizer model data file
using HttpClient httpClient = new();
const string modelUrl = @"https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model";
using Stream remoteStream = await httpClient.GetStreamAsync(modelUrl);

// Create the Llama tokenizer using the remote stream
Tokenizer llamaTokenizer = LlamaTokenizer.Create(remoteStream);
string input = "Hello, world!";
ids = llamaTokenizer.EncodeToIds(input);
Console.WriteLine(string.Join(", ", ids));
// prints: 1, 15043, 29892, 3186, 29991

Console.WriteLine($"Tokens: {llamaTokenizer.CountTokens(input)}");
// print: Tokens: 5
```

## Main Types

The main types provided by this library are:

* `Microsoft.ML.Tokenizers.Tokenizer`
* `Microsoft.ML.Tokenizers.BpeTokenizer`
* `Microsoft.ML.Tokenizers.EnglishRobertaTokenizer`
* `Microsoft.ML.Tokenizers.TiktokenTokenizer`
* `Microsoft.ML.Tokenizers.Normalizer`
* `Microsoft.ML.Tokenizers.PreTokenizer`

## Additional Documentation

* [Conceptual documentation](TODO)
* [API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers)

## Related Packages

<!-- The related packages associated with this package -->

## Feedback & Contributing

Microsoft.ML.Tokenizers is released as open source under the [MIT license](https://licenses.nuget.org/MIT). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/dotnet/machinelearning).
