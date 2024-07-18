# Porting to Microsoft.ML.Tokenizers

This guide provides general guidance on how to migrate from various tokenizer libraries to `Microsoft.ML.Tokenizers` for Tiktoken.

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
