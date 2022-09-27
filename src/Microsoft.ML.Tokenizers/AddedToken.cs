// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent a token added by the user on top of the existing Model vocabulary.
    /// AddedToken can be configured to specify the behavior they should have in various situations
    /// like:
    ///   - Whether they should only match single words
    ///   - Whether to include any WhiteSpace on its left or right
    /// </summary>
    public struct AddedToken : IEquatable<AddedToken>
    {
        /// <summary>
        /// Gets or sets the content of the added token
        /// </summary>
        public string Content { get; set; }

        /// <summary>
        /// Gets or sets whether this token must be a single word or can break words
        /// </summary>
        internal bool SingleWord { get; set; }

        /// <summary>
        /// Gets or sets whether this token should strip WhiteSpaces on its left
        /// </summary>
        internal bool LeftStrip { get; set; }

        /// <summary>
        /// Gets or sets whether this token should strip WhiteSpaces on its right
        /// </summary>
        internal bool RightStrip { get; set; }

        /// <summary>
        /// Gets or sets whether this token should be normalized
        /// </summary>
        internal bool Normalized { get; set; }

        /// <summary>
        /// Gets or sets whether this token is special
        /// </summary>
        internal bool Special { get; set; }

        /// <summary>
        /// Create a new AddedToken object.
        /// </summary>
        public AddedToken()
        {
            Content = "";
            SingleWord = LeftStrip = RightStrip = Special = false;
            Normalized = true;
        }

        /// <summary>
        /// Create a new AddedToken object from the given content, specifying if it is intended to be a
        /// special token. Special tokens are not normalized by default.
        /// </summary>
        /// <param name="content">The content of the added token.</param>
        /// <param name="special">Indicate whether this token is special.</param>
        public AddedToken(string content, bool special = false) : this()
        {
            Content = content ?? "";
            Special = special;
            Normalized = !special;
        }

        /// <summary>
        /// Determines whether two token instances are equal.
        /// </summary>
        /// <param name="other">The token to compare with the current token.</param>
        public bool Equals(AddedToken other) => Content == other.Content;

        // We only want to hash on the content. AddedToken cannot be added multiple times with different options
        /// <summary>
        /// Returns the hash code for the current token.
        /// </summary>
        public override int GetHashCode() => Content.GetHashCode();


        /// <summary>
        /// Defines an implicit conversion of a string object to AddedToken.
        /// </summary>
        public static implicit operator AddedToken(string token) => new AddedToken(token);
    }
}
