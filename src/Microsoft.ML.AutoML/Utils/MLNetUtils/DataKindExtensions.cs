// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class DataKindExtensions
    {
        /// <summary>
		/// Try to map a System.Type to a corresponding DataKind value.
		/// </summary>
		public static bool TryGetDataKind(this Type type, out DataKind kind)
        {
            if (type == typeof(sbyte))
            {
                kind = DataKind.SByte;
            }
            else if (type == typeof(byte))
            {
                kind = DataKind.Byte;
            }
            else if (type == typeof(short))
            {
                kind = DataKind.Int16;
            }
            else if (type == typeof(ushort))
            {
                kind = DataKind.UInt16;
            }
            else if (type == typeof(int))
            {
                kind = DataKind.Int32;
            }
            else if (type == typeof(uint))
            {
                kind = DataKind.UInt32;
            }
            else if (type == typeof(long))
            {
                kind = DataKind.Int64;
            }
            else if (type == typeof(ulong))
            {
                kind = DataKind.UInt64;
            }
            else if (type == typeof(float))
            {
                kind = DataKind.Single;
            }
            else if (type == typeof(double))
            {
                kind = DataKind.Double;
            }
            else
            {
                if (!(type == typeof(ReadOnlyMemory<char>)) && !(type == typeof(string)))
                {
                    if (type == typeof(bool))
                    {
                        kind = DataKind.Boolean;
                        goto IL_01ad;
                    }
                    if (type == typeof(TimeSpan))
                    {
                        kind = DataKind.TimeSpan;
                        goto IL_01ad;
                    }
                    if (type == typeof(DateTime))
                    {
                        kind = DataKind.DateTime;
                        goto IL_01ad;
                    }
                    if (type == typeof(DateTimeOffset))
                    {
                        kind = DataKind.DateTimeOffset;
                        goto IL_01ad;
                    }
                    if (type == typeof(DataViewRowId))
                    {
                        kind = DataKind.UInt16;
                        goto IL_01ad;
                    }
                    kind = (DataKind)0;
                    return false;
                }
                kind = DataKind.String;
            }
            goto IL_01ad;
        IL_01ad:
            return true;
        }
    }
}
