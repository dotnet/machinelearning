// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// This Split contains the underlying split token as well as its offsets
    /// in the original string. These offsets are in the `original` referential.
    /// It also contains any `Token` associated to the current split.
    /// </summary>
    public sealed class Split : IEquatable<Split>
    {
        /// <summary>
        /// Gets the underlying split token. Each SubString is represented by a token
        /// and in the end we might be carrying a lot of SubString representing various parts of the
        /// original input string.
        /// </summary>
        public string TokenString { get; }

        /// <summary>
        /// Returns the offset mapping to the original string
        /// </summary>
        public (int Index, int End) Offset { get; }

        /// <summary>
        /// create a Split object using the token and the offset
        /// </summary>
        public Split(string token, (int Index, int End) offset)
        {
            TokenString = token;
            Offset = offset;
        }

        /// <summary>
        /// Indicates whether the current Split object is equal to another Split object.
        /// </summary>
        /// <param name="other">The Split object to compare with the current object.</param>
        public bool Equals(Split? other) =>
            other is not null &&
            TokenString == other.TokenString &&
            Offset.Index == other.Offset.Index &&
            Offset.End == other.Offset.End;
    }


    /// <summary>
    /// Base class for all pre-tokenizers classes.
    /// The PreTokenizer is in charge of doing the pre-segmentation step.
    /// </summary>
    public abstract class PreTokenizer
    {
        /// <summary>
        /// Splits the given string in multiple substrings at the word boundary, keeping track of the offsets of said substrings from the original string.
        /// </summary>
        /// <param name="sentence">The string to split into tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public abstract IReadOnlyList<Split> PreTokenize(string sentence);
    }
}
