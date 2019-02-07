// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
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
                kind = DataKind.I1;
            }
            else if (type == typeof(byte))
            {
                kind = DataKind.U1;
            }
            else if (type == typeof(short))
            {
                kind = DataKind.I2;
            }
            else if (type == typeof(ushort))
            {
                kind = DataKind.U2;
            }
            else if (type == typeof(int))
            {
                kind = DataKind.I4;
            }
            else if (type == typeof(uint))
            {
                kind = DataKind.U4;
            }
            else if (type == typeof(long))
            {
                kind = DataKind.I8;
            }
            else if (type == typeof(ulong))
            {
                kind = DataKind.U8;
            }
            else if (type == typeof(float))
            {
                kind = DataKind.R4;
            }
            else if (type == typeof(double))
            {
                kind = DataKind.R8;
            }
            else
            {
                if (!(type == typeof(ReadOnlyMemory<char>)) && !(type == typeof(string)))
                {
                    if (type == typeof(bool))
                    {
                        kind = DataKind.BL;
                        goto IL_01ad;
                    }
                    if (type == typeof(TimeSpan))
                    {
                        kind = DataKind.TS;
                        goto IL_01ad;
                    }
                    if (type == typeof(DateTime))
                    {
                        kind = DataKind.DT;
                        goto IL_01ad;
                    }
                    if (type == typeof(DateTimeOffset))
                    {
                        kind = DataKind.DZ;
                        goto IL_01ad;
                    }
                    if (type == typeof(RowId))
                    {
                        kind = DataKind.UG;
                        goto IL_01ad;
                    }
                    kind = (DataKind)0;
                    return false;
                }
                kind = DataKind.TX;
            }
            goto IL_01ad;
        IL_01ad:
            return true;
        }
    }
}
