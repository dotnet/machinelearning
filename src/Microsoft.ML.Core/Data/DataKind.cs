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
#pragma warning disable TLC_GeneralName // The data kind enum has its own logic, independnet of C# naming conventions.
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
#pragma warning restore TLC_GeneralName
    }

    /// <summary>
    /// Extension methods related to the DataKind enum.
    /// </summary>
    public static class DataKindExtensions
    {
        public const DataKind KindMin = DataKind.I1;
        public const DataKind KindLim = DataKind.UG + 1;
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
                    return typeof(DvInt1);
                case DataKind.U1:
                    return typeof(byte);
                case DataKind.I2:
                    return typeof(DvInt2);
                case DataKind.U2:
                    return typeof(ushort);
                case DataKind.I4:
                    return typeof(DvInt4);
                case DataKind.U4:
                    return typeof(uint);
                case DataKind.I8:
                    return typeof(DvInt8);
                case DataKind.U8:
                    return typeof(ulong);
                case DataKind.R4:
                    return typeof(Single);
                case DataKind.R8:
                    return typeof(Double);
                case DataKind.TX:
                    return typeof(DvText);
                case DataKind.BL:
                    return typeof(DvBool);
                case DataKind.TS:
                    return typeof(DvTimeSpan);
                case DataKind.DT:
                    return typeof(DvDateTime);
                case DataKind.DZ:
                    return typeof(DvDateTimeZone);
                case DataKind.UG:
                    return typeof(UInt128);
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
            if (type == typeof(DvInt1) || type == typeof(sbyte) || type == typeof(sbyte?))
                kind = DataKind.I1;
            else if (type == typeof(byte) || type == typeof(byte?))
                kind = DataKind.U1;
            else if (type == typeof(DvInt2)|| type== typeof(short) || type == typeof(short?))
                kind = DataKind.I2;
            else if (type == typeof(ushort)|| type == typeof(ushort?))
                kind = DataKind.U2;
            else if (type == typeof(DvInt4) || type == typeof(int)|| type == typeof(int?))
                kind = DataKind.I4;
            else if (type == typeof(uint)|| type == typeof(uint?))
                kind = DataKind.U4;
            else if (type == typeof(DvInt8) || type==typeof(long)|| type == typeof(long?))
                kind = DataKind.I8;
            else if (type == typeof(ulong)|| type == typeof(ulong?))
                kind = DataKind.U8;
            else if (type == typeof(Single)|| type == typeof(Single?))
                kind = DataKind.R4;
            else if (type == typeof(Double)|| type == typeof(Double?))
                kind = DataKind.R8;
            else if (type == typeof(DvText))
                kind = DataKind.TX;
            else if (type == typeof(DvBool) || type == typeof(bool) || type == typeof(bool?))
                kind = DataKind.BL;
            else if (type == typeof(DvTimeSpan))
                kind = DataKind.TS;
            else if (type == typeof(DvDateTime))
                kind = DataKind.DT;
            else if (type == typeof(DvDateTimeZone))
                kind = DataKind.DZ;
            else if (type == typeof(UInt128))
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