// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal struct Pair<T> : IEquatable<Pair<T>>, IComparable<Pair<T>> where T : struct, IEquatable<T>, IComparable<T>
    {
        public T First { get; set; }
        public T Second { get; set; }

        public static Pair<T> Create(T first, T second) => new Pair<T>(first, second);

        public Pair(T first, T second)
        {
            First = first;
            Second = second;
        }

        public bool Equals(Pair<T> other) => First.Equals(other.First) && Second.Equals(other.Second);

        public override int GetHashCode()
        {
            int hashcode = 23;
            hashcode = (hashcode * 37) + First.GetHashCode();
            hashcode = (hashcode * 37) + Second.GetHashCode();
            return hashcode;

        }

        public int CompareTo(Pair<T> other)
        {
            int compareFirst = First.CompareTo(other.First);
            return compareFirst == 0 ? Second.CompareTo(other.Second) : compareFirst;
        }
    }
}
