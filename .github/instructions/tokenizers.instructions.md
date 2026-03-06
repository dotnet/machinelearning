---
applyTo:
  - "src/Microsoft.ML.Tokenizers*/**"
  - "test/Microsoft.ML.Tokenizers*/**"
---

# Tokenizer Guidelines

## Overview

`Microsoft.ML.Tokenizers` provides BPE, WordPiece, and SentencePiece tokenizer implementations. Pre-built tokenizer data packages exist for common vocabularies: `Cl100kBase`, `Gpt2`, `O200kBase`, `P50kBase`, `R50kBase`.

## Structure

- `src/Microsoft.ML.Tokenizers/` — Core tokenizer engine
- `src/Microsoft.ML.Tokenizers.Data.*/` — Pre-built vocabulary data packages
- `test/Microsoft.ML.Tokenizers.Tests/` — Unit tests
- `test/Microsoft.ML.Tokenizers.Data.Tests/` — Data package tests

## Conventions

- Tokenizer implementations should be stateless where possible
- Vocabulary data is embedded as assembly resources in the `.Data.*` packages
- Test tokenizer output against known expected token sequences
