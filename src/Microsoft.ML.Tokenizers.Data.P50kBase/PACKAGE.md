## About

The `Microsoft.ML.Tokenizers.Data.P50kBase` includes the Tiktoken tokenizer data file `p50k_base.tiktoken`, which is utilized by models such as `text-davinci-002`.

## Key Features

* This package mainly contains the `p50k_base.tiktoken` file, which is used by the Tiktoken tokenizer. This data file is used by the following models:
      1. text-davinci-002
      2. text-davinci-003
      3. code-davinci-001
      4. code-davinci-002
      5. code-cushman-001
      6. code-cushman-002
      7. davinci-codex
      8. cushman-codex

## How to Use

Reference this package in your project to use the Tiktoken tokenizer with the specified models.

```csharp

// Create a tokenizer for the specified model or any other listed model name
Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("text-davinci-002");

// Create a tokenizer for the specified encoding
Tokenizer tokenizer = TiktokenTokenizer.CreateForEncoding("p50k_base");

```

## Main Types

Users shouldn't use any types exposed by this package directly. This package is intended to provide tokenizer data files.

## Additional Documentation

* [API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers)

## Related Packages

<!-- The related packages associated with this package -->
Microsoft.ML.Tokenizers

## Feedback & Contributing

Microsoft.ML.Tokenizers.Data.P50kBase is released as open source under the [MIT license](https://licenses.nuget.org/MIT). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/dotnet/machinelearning).
