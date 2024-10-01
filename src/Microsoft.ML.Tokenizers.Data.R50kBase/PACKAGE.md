## About

The `Microsoft.ML.Tokenizers.Data.R50kBase` includes the Tiktoken tokenizer data file `r50k_base.tiktoken`, which is utilized by models such as `text-davinci-001`.

## Key Features

* This package mainly contains the `r50k_base.tiktoken` file, which is used by the Tiktoken tokenizer. This data file is used by the following models:
      1. text-davinci-001
      2. text-curie-001
      3. text-babbage-001
      4. text-ada-001
      5. davinci
      6. curie
      7. babbage
      8. ada
      9. text-similarity-davinci-001
     10. text-similarity-curie-001
     11. text-similarity-babbage-001
     12. text-similarity-ada-001
     13. text-search-davinci-doc-001
     14. text-search-curie-doc-001
     15. text-search-babbage-doc-001
     16. text-search-ada-doc-001
     17. code-search-babbage-code-001
     18. code-search-ada-code-001

## How to Use

Reference this package in your project to use the Tiktoken tokenizer with the specified models.

```csharp

// Create a tokenizer for the specified model or any other listed model name
Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("text-davinci-001");

// Create a tokenizer for the specified encoding
Tokenizer tokenizer = TiktokenTokenizer.CreateForEncoding("r50k_base");

```

## Main Types

Users shouldn't use any types exposed by this package directly. This package is intended to provide tokenizer data files.

## Additional Documentation

* [API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers)

## Related Packages

<!-- The related packages associated with this package -->
Microsoft.ML.Tokenizers

## Feedback & Contributing

Microsoft.ML.Tokenizers.Data.R50kBase is released as open source under the [MIT license](https://licenses.nuget.org/MIT). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/dotnet/machinelearning).
