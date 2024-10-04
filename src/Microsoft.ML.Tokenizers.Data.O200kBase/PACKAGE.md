## About

The `Microsoft.ML.Tokenizers.Data.O200kBase` includes the Tiktoken tokenizer data file o200k_base.tiktoken, which is utilized by models such as `Gpt-4o`.

## Key Features

* This package mainly contains the o200k_base.tiktoken file, which is used by the Tiktoken tokenizer. This data file is used by the Gpt-4o model.

## How to Use

Reference this package in your project to use the Tiktoken tokenizer with the specified model.

```csharp

// Create a tokenizer for the specified model
Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("Gpt-4o");

// Create a tokenizer for the specified encoding
Tokenizer tokenizer = TiktokenTokenizer.CreateForEncoding("o200k_base");

```

## Main Types

Users shouldn't use any types exposed by this package directly. This package is intended to provide tokenizer data files.

## Additional Documentation

* [API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers)

## Related Packages

<!-- The related packages associated with this package -->
Microsoft.ML.Tokenizers

## Feedback & Contributing

Microsoft.ML.Tokenizers.Data.O200kBase is released as open source under the [MIT license](https://licenses.nuget.org/MIT). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/dotnet/machinelearning).
