// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The Encoding represents the output of a Tokenizer.
    /// </summary>
    public sealed class TokenizerResult
    {
        /// <summary>
        /// Create a new object of the TokenizerResult object.
        /// </summary>
        /// <param name="originalString">The list of tokens to merge.</param>
        /// <param name="normalizedString">The list of tokens to merge.</param>
        /// <param name="splits">The list of tokens to merge.</param>
        /// <param name="offsetsMappedToOriginalString">Indicate whether the offsets is mapped to the original string or the normalized string.</param>
        public TokenizerResult(string originalString, string normalizedString, IReadOnlyList<Split> splits, bool offsetsMappedToOriginalString)
        {
            OriginalString = originalString;
            NormalizedString = normalizedString;
            Splits = splits;
            OffsetsMappedToOriginalString = offsetsMappedToOriginalString;
        }

        /// <summary>
        /// Gets the original tokenized string.
        /// </summary>
        public string? OriginalString { get; }

        /// <summary>
        /// Gets the normalized form of the original string.
        /// </summary>
        public string? NormalizedString { get; }

        /// <summary>
        /// Gets the normalized form of the original string.
        /// </summary>
        public bool OffsetsMappedToOriginalString { get; }

        internal IReadOnlyList<Split> Splits { get; }
        private List<Token>? _tokens;
        private List<string>? _tokensWords;
        private List<int>? _ids;
        private List<(int Index, int End)>? _offsets;

        internal void AddTokens(IReadOnlyList<Token> addedTokens)
        {
            if (_tokens is null)
            {
                _tokens = new(addedTokens);
                return;
            }

            foreach (var token in addedTokens)
            {
                _tokens.Add(token);
            }
        }

        private static readonly IReadOnlyList<int> _emptyIds = new List<int>();

        /// <summary>
        /// Gets list of the tokens Ids.
        /// The Ids are the main input to a Language Model. They are the token indices, the numerical representations that a LM understands.
        /// </summary>
        public IReadOnlyList<int> Ids
        {
            get
            {
                if (_ids is not null)
                {
                    return _ids;
                }

                if (_tokens is null)
                {
                    return _emptyIds;
                }

                _ids = new List<int>(_tokens.Count);

                foreach (var token in _tokens)
                {
                    _ids.Add(token.Id);
                }

                return _ids;
            }
        }

        private static readonly IReadOnlyList<string> _emptyTokens = new List<string>();

        /// <summary>
        /// Gets the generated tokens. They are the string representation of the Ids.
        /// </summary>
        public IReadOnlyList<string> Tokens
        {
            get
            {
                if (_tokensWords is not null)
                {
                    return _tokensWords;
                }

                if (_tokens is null)
                {
                    return _emptyTokens;
                }

                _tokensWords = new List<string>(_tokens.Count);

                foreach (var token in _tokens)
                {
                    _tokensWords.Add(token.Value);
                }

                return _tokensWords;
            }
        }

        private static readonly IReadOnlyList<(int, int)> _emptyOffsets = new List<(int, int)>();

        /// <summary>
        /// Gets The list of offsets. These offsets let’s you slice the input string, and thus retrieve
        /// the original part that led to producing the corresponding token.
        /// </summary>
        public IReadOnlyList<(int Index, int End)> Offsets
        {
            get
            {
                if (_offsets is not null)
                {
                    return _offsets;
                }

                if (_tokens is null)
                {
                    return _emptyOffsets;
                }

                _offsets = new List<(int Index, int End)>(_tokens.Count);

                foreach (var token in _tokens)
                {
                    _offsets.Add(token.Offset);
                }

                return _offsets;
            }
        }
    }
}
