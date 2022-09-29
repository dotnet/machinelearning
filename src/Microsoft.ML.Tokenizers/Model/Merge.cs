// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    // Used inside the Word class for merging the tokenized parts.
    internal struct Merge : IEquatable<Merge>, IComparable<Merge>
    {
        public Merge(int pos, int rank, int newId)
        {
            Pos = pos;
            Rank = rank;
            NewId = newId;
        }

        public int Pos { get; set; }
        public int Rank { get; set; }
        public int NewId { get; set; }

        public int CompareTo(Merge other)
        {
            if (Rank != other.Rank)
            {
                return Rank.CompareTo(other.Rank);
            }

            return Pos.CompareTo(other.Pos);
        }

        public override int GetHashCode()
        {
            int hashcode = 23;
            hashcode = (hashcode * 37) + Rank.GetHashCode();
            hashcode = (hashcode * 37) + Pos.GetHashCode();
            return hashcode;
        }

        public bool Equals(Merge other) => Pos == other.Pos && Rank == other.Rank;
    }
}
