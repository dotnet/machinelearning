// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Data type specifier.
    /// </summary>
    public enum DataKind : byte
    {
        // Notes:
        // * These values are serialized, so changing them breaks binary formats.
        // * We intentionally skip zero.
        // * Some code depends on sizeof(DataKind) == sizeof(byte).

        I1 = 1,
        U1 = 2,
        I2 = 3,
        U2 = 4,
        I4 = 5,
        U4 = 6,
        I8 = 7,
        U8 = 8,
        R4 = 9,
        R8 = 10,
        Num = R4,

        TX = 11,
#pragma warning disable MSML_GeneralName // The data kind enum has its own logic, independent of C# naming conventions.
        TXT = TX,
        Text = TX,

        BL = 12,
        Bool = BL,

        TS = 13,
        TimeSpan = TS,
        DT = 14,
        DateTime = DT,
        DZ = 15,
        DateTimeZone = DZ,

        UG = 16, // Unsigned 16-byte integer.
        U16 = UG,
#pragma warning restore MSML_GeneralName
    }

    /// <summary>
    /// Extension methods related to the DataKind enum.
    /// </summary>
    public static class DataKindExtensions
    {
        public const DataKind KindMin = DataKind.I1;
        public const DataKind KindLim = DataKind.U16 + 1;
        public const int KindCount = KindLim - KindMin;

        /// <summary>
        /// Maps a DataKind to a value suitable for indexing into an array of size KindCount.
        /// </summary>
        public static int ToIndex(this DataKind kind)
        {
            return kind - KindMin;
        }

        /// <summary>
        /// Maps from an index into an array of size KindCount to the corresponding DataKind
        /// </summary>
        public static DataKind FromIndex(int index)
        {
            Contracts.Check(0 <= index && index < KindCount);
            return (DataKind)(index + (int)KindMin);
        }

        /// <summary>
        /// For integer DataKinds, this returns the maximum legal value. For un-supported kinds,
        /// it returns zero.
        /// </summary>
        public static ulong ToMaxInt(this DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return (ulong)sbyte.MaxValue;
                case DataKind.U1:
                    return byte.MaxValue;
                case DataKind.I2:
                    return (ulong)short.MaxValue;
                case DataKind.U2:
                    return ushort.MaxValue;
                case DataKind.I4:
                    return int.MaxValue;
                case DataKind.U4:
                    return uint.MaxValue;
                case DataKind.I8:
                    return long.MaxValue;
                case DataKind.U8:
                    return ulong.MaxValue;
            }

            return 0;
        }

        /// <summary>
        /// For integer DataKinds, this returns the minimum legal value. For un-supported kinds,
        /// it returns one.
        /// </summary>
        public static long ToMinInt(this DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return sbyte.MinValue;
                case DataKind.U1:
                    return byte.MinValue;
                case DataKind.I2:
                    return short.MinValue;
                case DataKind.U2:
                    return ushort.MinValue;
                case DataKind.I4:
                    return int.MinValue;
                case DataKind.U4:
                    return uint.MinValue;
                case DataKind.I8:
                    return long.MinValue;
                case DataKind.U8:
                    return 0;
            }

            return 1;
        }

        /// <summary>
        /// Maps a DataKind to the associated .Net representation type.
        /// </summary>
        public static Type ToType(this DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return typeof(sbyte);
                case DataKind.U1:
                    return typeof(byte);
                case DataKind.I2:
                    return typeof(short);
                case DataKind.U2:
                    return typeof(ushort);
                case DataKind.I4:
                    return typeof(int);
                case DataKind.U4:
                    return typeof(uint);
                case DataKind.I8:
                    return typeof(long);
                case DataKind.U8:
                    return typeof(ulong);
                case DataKind.R4:
                    return typeof(Single);
                case DataKind.R8:
                    return typeof(Double);
                case DataKind.TX:
                    return typeof(ReadOnlyMemory<char>);
                case DataKind.BL:
                    return typeof(bool);
                case DataKind.TS:
                    return typeof(TimeSpan);
                case DataKind.DT:
                    return typeof(DateTime);
                case DataKind.DZ:
                    return typeof(DateTimeOffset);
                case DataKind.UG:
                    return typeof(RowId);
            }

            return null;
        }

        /// <summary>
        /// Try to map a System.Type to a corresponding DataKind value.
        /// </summary>
        public static bool TryGetDataKind(this Type type, out DataKind kind)
        {
            Contracts.CheckValueOrNull(type);

            // REVIEW: Make this more efficient. Should we have a global dictionary?
            if (type == typeof(sbyte))
                kind = DataKind.I1;
            else if (type == typeof(byte))
                kind = DataKind.U1;
            else if (type == typeof(short))
                kind = DataKind.I2;
            else if (type == typeof(ushort))
                kind = DataKind.U2;
            else if (type == typeof(int))
                kind = DataKind.I4;
            else if (type == typeof(uint))
                kind = DataKind.U4;
            else if (type == typeof(long))
                kind = DataKind.I8;
            else if (type == typeof(ulong))
                kind = DataKind.U8;
            else if (type == typeof(Single))
                kind = DataKind.R4;
            else if (type == typeof(Double))
                kind = DataKind.R8;
            else if (type == typeof(ReadOnlyMemory<char>) || type == typeof(string))
                kind = DataKind.TX;
            else if (type == typeof(bool))
                kind = DataKind.BL;
            else if (type == typeof(TimeSpan))
                kind = DataKind.TS;
            else if (type == typeof(DateTime))
                kind = DataKind.DT;
            else if (type == typeof(DateTimeOffset))
                kind = DataKind.DZ;
            else if (type == typeof(RowId))
                kind = DataKind.UG;
            else
            {
                kind = default(DataKind);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Get the canonical string for a DataKind. Note that using DataKind.ToString() is not stable
        /// and is also slow, so use this instead.
        /// </summary>
        public static string GetString(this DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return "I1";
                case DataKind.I2:
                    return "I2";
                case DataKind.I4:
                    return "I4";
                case DataKind.I8:
                    return "I8";
                case DataKind.U1:
                    return "U1";
                case DataKind.U2:
                    return "U2";
                case DataKind.U4:
                    return "U4";
                case DataKind.U8:
                    return "U8";
                case DataKind.R4:
                    return "R4";
                case DataKind.R8:
                    return "R8";
                case DataKind.BL:
                    return "BL";
                case DataKind.TX:
                    return "TX";
                case DataKind.TS:
                    return "TS";
                case DataKind.DT:
                    return "DT";
                case DataKind.DZ:
                    return "DZ";
                case DataKind.UG:
                    return "UG";
            }
            return "";
        }
    }
}