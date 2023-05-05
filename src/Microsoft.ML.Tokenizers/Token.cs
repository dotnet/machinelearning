// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the token produced from the tokenization process containing the token substring,
    /// the id associated to the token substring, and the offset mapping to the original string.
    /// </summary>
    public sealed class Token
    {
        /// <summary>
        /// Gets or sets the Id value associated to the token.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the token string value.
        /// </summary>
        public string Value { get; set; }

        /// <summary>
        /// Gets or sets the offset mapping to the original string.
        /// </summary>
        public (int Index, int End) Offset { get; set; }

        /// <summary>
        /// Construct a new Token object using the token value, Id, and the offset mapping to the original string.
        /// </summary>
        /// <param name="id">The Id value associated to the token.</param>
        /// <param name="value">The token string value.</param>
        /// <param name="offset">The offset mapping to the original string.</param>
        public Token(int id, string value, (int, int) offset)
        {
            Id = id;
            Offset = offset;
            Value = value;
        }
    }
}
