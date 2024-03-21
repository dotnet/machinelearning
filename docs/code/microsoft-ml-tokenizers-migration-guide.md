# Porting to Microsoft.ML.Tokenizers

This guide provides general guidance on how to migrate from various tokenizer libraries to `Microsoft.ML.Tokenizers` for Tiktoken.

## Microsoft.DeepDev.TokenizerLib

### API Guidance

| Microsoft.DeepDev.TokenizerLib | Microsoft.ML.Tokenizers
| --- | --- |
| [TikTokenizer](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/TikTokenizer.cs#L20) | [Tokenizer](https://github.com/dotnet/machinelearning/blob/acced974bea6ed484503a595d87a3e7016c8a558/src/Microsoft.ML.Tokenizers/Tokenizer.cs#L28) |
| [ITokenizer](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/ITokenizer.cs#L7) | [Tokenizer](https://github.com/dotnet/machinelearning/blob/acced974bea6ed484503a595d87a3e7016c8a558/src/Microsoft.ML.Tokenizers/Tokenizer.cs#L28) |
| [TokenizerBuilder](https://github.com/microsoft/Tokenizer/blob/2c9ba5d343de52eb27521afef7c0c2f0f76c9c52/Tokenizer_C%23/TokenizerLib/TokenizerBuilder.cs#L14) | [Tokenizer.CreateTiktokenForModel/Async](https://github.com/dotnet/machinelearning/blob/70e191b3fae444f6625fdc001071de1e2bd1080b/src/Microsoft.ML.Tokenizers/Tokenizer.cs#L298-L330) downloads<br> [Tokenizer.CreateTiktokenForModel(Async/Stream)](https://github.com/dotnet/machinelearning/blob/70e191b3fae444f6625fdc001071de1e2bd1080b/src/Microsoft.ML.Tokenizers/Tokenizer.cs#L222-L296) user provided file stream |

### General Guidance

- Avoid hard-coding tiktoken regexes and special tokens.  Instead use the appropriate Tiktoken.`CreateByModelNameAsync` method to create the tokenizer from either a downloaded file, or a provided stream.
- Avoid doing encoding if you need the token count or encoded Ids. Instead use `Tokenizer.CountTokens` for getting the token count and `Tokenizer.EncodeToIds` for getting the encode ids.
- Avoid doing encoding if all you need is to truncate to a token budget.  Instead use `Tokenizer.IndexOfCount` or `LastIndexOfCount` to find the index to truncate from the start or end of a string, respectively.
