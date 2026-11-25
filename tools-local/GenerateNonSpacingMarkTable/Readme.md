# GenerateNonSpacingMarkTable Tool

## Overview

This tool generates [NonSpacingMarkRemover.Data.cs](../../src/Microsoft.ML.Tokenizers/Normalizer/NonSpacingMarkRemover.Data.cs) from the Unicode Character Database. For each character whose canonical decomposition contains non-spacing marks, that table holds the character without them (for example, é becomes e). `BertTokenizer` uses it to remove accents when `string.Normalize` doesn't work, which is the case in globalization-invariant mode.

The table isn't generated during the build, and this tool isn't part of `Microsoft.ML.sln`. Run it by hand when updating the Unicode version, and commit the new table.

The current table was generated from Unicode 18.0.0.

## Updating the version of Unicode used

1. Download `UnicodeData.txt` and `DerivedNormalizationProps.txt` of the new version from unicode.org. For example, for version 18.0.0 they're at https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt and https://www.unicode.org/Public/18.0.0/ucd/DerivedNormalizationProps.txt.
2. From this folder, run `dotnet run -- <path to UnicodeData.txt> <path to DerivedNormalizationProps.txt>`.
3. The tool writes `NonSpacingMarkRemover.Data.cs` to this folder. Move it over the one in `src/Microsoft.ML.Tokenizers/Normalizer`. Don't commit it here.
4. Update the Unicode version in this file.
5. Run the BertTokenizer tests. `TestRemoveNonSpacingMarksForAllCharactersInInvariantGlobalizationMode` compares the table with `string.Normalize` for every character that the .NET runtime running the test can decompose.
6. Update the "License notice for Unicode data" section in [THIRD-PARTY-NOTICES.TXT](../../THIRD-PARTY-NOTICES.TXT) with the current text of https://www.unicode.org/license.html.

These commands do steps 1, 2, 3 and 5 from the repo root. Replace 18.0.0 with the new version. The tests need a build of the repo first, see the [developer guide](../../docs/project-docs/developer-guide.md).

Windows (cmd.exe):

```cmd
curl -o "%TEMP%\UnicodeData.txt" https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt
curl -o "%TEMP%\DerivedNormalizationProps.txt" https://www.unicode.org/Public/18.0.0/ucd/DerivedNormalizationProps.txt
cd tools-local\GenerateNonSpacingMarkTable
dotnet run -- "%TEMP%\UnicodeData.txt" "%TEMP%\DerivedNormalizationProps.txt"
move /y NonSpacingMarkRemover.Data.cs ..\..\src\Microsoft.ML.Tokenizers\Normalizer\
cd ..\..
.dotnet\dotnet.exe test test\Microsoft.ML.Tokenizers.Tests\Microsoft.ML.Tokenizers.Tests.csproj -f net10.0 --filter "FullyQualifiedName~BertTokenizerTests"
```

Linux/macOS (bash):

```bash
ucd=$(mktemp -d)
curl -o "$ucd/UnicodeData.txt" https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt
curl -o "$ucd/DerivedNormalizationProps.txt" https://www.unicode.org/Public/18.0.0/ucd/DerivedNormalizationProps.txt
cd tools-local/GenerateNonSpacingMarkTable
dotnet run -- "$ucd/UnicodeData.txt" "$ucd/DerivedNormalizationProps.txt"
mv NonSpacingMarkRemover.Data.cs ../../src/Microsoft.ML.Tokenizers/Normalizer/
cd ../..
./.dotnet/dotnet test test/Microsoft.ML.Tokenizers.Tests/Microsoft.ML.Tokenizers.Tests.csproj -f net10.0 --filter "FullyQualifiedName~BertTokenizerTests"
```
