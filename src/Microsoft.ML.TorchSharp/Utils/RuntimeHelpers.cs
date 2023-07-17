// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace System.Runtime.CompilerServices
{
    internal class RuntimeHelpers
    {
        public static T[] GetSubArray<T>(T[] array, Range range)
        {
            Type elementType = array.GetType().GetElementType();
            var o = range.GetOffsetAndLength(array.Length);
            Span<T> source = new Span<T>(array, o.Offset, o.Length);

            if (elementType.IsValueType)
            {
                return source.ToArray();
            }
            else
            {
                T[] newArray = (T[])Array.CreateInstance(elementType, source.Length);
                source.CopyTo(newArray);
                return newArray;
            }
        }
    }
}
