// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers;

/// <summary>
/// CompositePreTokenizer is a pre-tokenizer that applies multiple pre-tokenizers in sequence.
/// </summary>
public class CompositePreTokenizer : PreTokenizer
{
    private const int MaxPreTokenizersCount = 10;
    private readonly IReadOnlyList<PreTokenizer> _preTokenizers;

    /// <summary>
    /// Initializes a new instance of the <see cref="CompositePreTokenizer"/> class.
    /// </summary>
    /// <param name="preTokenizers">The list of pre-tokenizers to apply.</param>
    /// <param name="specialTokens">The special tokens to apply.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="preTokenizers"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="preTokenizers"/> contains null elements.</exception>
    /// <remarks>
    /// The <see cref="CompositePreTokenizer"/> can accept a list of pre-tokenizers with a maximum of 10 items.
    /// </remarks>
    public CompositePreTokenizer(IReadOnlyList<PreTokenizer> preTokenizers, IReadOnlyDictionary<string, int>? specialTokens = null)
    {
        if (preTokenizers is null)
        {
            throw new ArgumentNullException(nameof(preTokenizers));
        }

        // Limit the number of pre-tokenizers to a reasonable amount as we do a recursive calls depending on the number of pre-tokenizers
        if (preTokenizers.Count > MaxPreTokenizersCount)
        {
            throw new ArgumentException($"Too many pre-tokenizers provided. Maximum is {MaxPreTokenizersCount}.", nameof(preTokenizers));
        }

        foreach (var preTokenizer in preTokenizers)
        {
            if (preTokenizer is null)
            {
                throw new ArgumentException("Pre-tokenizer cannot be null.", nameof(preTokenizers));
            }
        }

        if (specialTokens is { Count: > 0 })
        {
            var list = new List<PreTokenizer>(specialTokens.Count + 1);

            list.Add(new RegexPreTokenizer(new Regex(string.Join("|", specialTokens.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled), null));

            foreach (var preTokenizer in preTokenizers)
            {
                list.Add(preTokenizer);
            }

            _preTokenizers = list.AsReadOnly();
        }
        else
        {
            _preTokenizers = preTokenizers;
        }
    }

    /// <summary>
    /// Gets the list of pre-tokenizers.
    /// </summary>
    public IReadOnlyList<PreTokenizer> PreTokenizers => _preTokenizers;

    /// <summary>
    /// Pre-tokenizes the input text using the specified pre-tokenizers.
    /// </summary>
    /// <param name="text">The input text to pre-tokenize.</param>
    /// <returns>The list of pre-tokenized ranges.</returns>
    public override IEnumerable<(int Offset, int Length)> PreTokenize(string text)
    {
        int beginning = 0;
        foreach ((int Offset, int Length) range in SplitText(text, _preTokenizers, preTokenizerIndex: 0, beginning, text.Length - beginning))
        {
            yield return (range.Offset, range.Length);
            beginning += range.Length;
        }

        static IEnumerable<(int Offset, int Length)> SplitText(string text, IReadOnlyList<PreTokenizer> preTokenizers, int preTokenizerIndex, int offset, int length)
        {
            Debug.Assert(preTokenizerIndex < preTokenizers.Count, "Index out of range for pre-tokenizers.");
            var preTokenizer = preTokenizers[preTokenizerIndex];

            int beginning = 0; // relative to the offset
            foreach ((int Offset, int Length) range in preTokenizer.PreTokenize(text.AsSpan(offset, length)))
            {
                if (range.Offset > beginning)
                {
                    // Recurse for subsequent tokenizers
                    if (preTokenizerIndex + 1 < preTokenizers.Count)
                    {
                        foreach ((int Offset, int Length) subRange in SplitText(text, preTokenizers, preTokenizerIndex + 1, offset + beginning, range.Offset - beginning))
                        {
                            yield return subRange;
                        }
                    }
                    else
                    {
                        yield return (offset + beginning, range.Offset);
                    }
                }

                beginning = range.Offset + range.Length;

                yield return (offset + range.Offset, range.Length);
            }

            if (beginning < length)
            {
                // Handle the remaining of the text
                if (preTokenizerIndex + 1 < preTokenizers.Count)
                {
                    foreach ((int Offset, int Length) subRange in SplitText(text, preTokenizers, preTokenizerIndex + 1, offset + beginning, length - beginning))
                    {
                        yield return subRange;
                    }
                }
                else
                {
                    yield return (offset + beginning, length);
                }
            }
        }
    }

    /// <summary>
    /// Pre-tokenizes the input text span using the specified pre-tokenizers.
    /// </summary>
    /// <param name="text">The input text span to pre-tokenize.</param>
    /// <returns>The list of pre-tokenized ranges.</returns>
    public override IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text)
    {
        if (text.IsEmpty)
        {
            return [];
        }

        char[] buffer = ArrayPool<char>.Shared.Rent(text.Length);
        text.CopyTo(buffer);

        IEnumerable<(int Offset, int Length)> result = PreTokenize(buffer, text.Length);

        ArrayPool<char>.Shared.Return(buffer);
        return result;
    }

    private IEnumerable<(int Offset, int Length)> PreTokenize(char[] text, int length)
    {
        int beginning = 0;

        foreach ((int Offset, int Length) range in SplitText(text, _preTokenizers, preTokenizerIndex: 0, beginning, length - beginning))
        {
            yield return (range.Offset, range.Length);
            beginning += range.Length;
        }

        static IEnumerable<(int Offset, int Length)> SplitText(char[] text, IReadOnlyList<PreTokenizer> preTokenizers, int preTokenizerIndex, int offset, int length)
        {
            Debug.Assert(preTokenizerIndex < preTokenizers.Count, "Index out of range for pre-tokenizers.");
            var preTokenizer = preTokenizers[preTokenizerIndex];

            int beginning = 0; // relative to the offset
            foreach ((int Offset, int Length) range in preTokenizer.PreTokenize(text.AsSpan(offset, length)))
            {
                if (range.Offset > beginning)
                {
                    // Recurse for subsequent tokenizers
                    if (preTokenizerIndex + 1 < preTokenizers.Count)
                    {
                        foreach ((int Offset, int Length) subRange in SplitText(text, preTokenizers, preTokenizerIndex + 1, offset + beginning, range.Offset - beginning))
                        {
                            yield return subRange;
                        }
                    }
                    else
                    {
                        yield return (offset + beginning, range.Offset);
                    }
                }

                beginning = range.Offset + range.Length;

                yield return (offset + range.Offset, range.Length);
            }

            if (beginning < length)
            {
                // Handle the remaining of the text
                if (preTokenizerIndex + 1 < preTokenizers.Count)
                {
                    foreach ((int Offset, int Length) subRange in SplitText(text, preTokenizers, preTokenizerIndex + 1, offset + beginning, length - beginning))
                    {
                        yield return subRange;
                    }
                }
                else
                {
                    yield return (offset + beginning, length);
                }
            }
        }
    }
}
