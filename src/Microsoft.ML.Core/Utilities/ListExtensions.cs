// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public static class ListExtensions
    {
        public static void Push<T>(this List<T> list, T item)
        {
            Contracts.AssertValue(list);
            list.Add(item);
        }

        public static T Pop<T>(this List<T> list)
        {
            Contracts.AssertValue(list);
            Contracts.Assert(list.Count > 0);
            int index = list.Count - 1;
            T item = list[index];
            list.RemoveAt(index);
            return item;
        }

        public static void PopTo<T>(this List<T> list, int depth)
        {
            Contracts.AssertValue(list);
            Contracts.Assert(0 <= depth && depth <= list.Count);
            list.RemoveRange(depth, list.Count - depth);
        }

        public static T Peek<T>(this List<T> list)
        {
            Contracts.AssertValue(list);
            Contracts.Assert(list.Count > 0);
            return list[list.Count - 1];
        }
    }
}
