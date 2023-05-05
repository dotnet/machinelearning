// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal struct Symbol
    {
        internal int C { get; set; }
        internal int Prev { get; set; }
        internal int Next { get; set; }
        internal int Len { get; set; } // number of characters

        public Symbol(int c, int prev, int next, int len)
        {
            C = c;
            Prev = prev;
            Next = next;
            Len = len;
        }

        /// Merges the current Symbol with the other one.
        /// In order to update prev/next, we consider Self to be the Symbol on the left,
        /// and other to be the next one on the right.
        internal void MergeWith(ref Symbol other, int c)
        {
            C = c;
            Len += other.Len;
            Next = other.Next;
        }
    }
}
