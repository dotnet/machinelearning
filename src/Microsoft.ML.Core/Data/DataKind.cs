// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.Data
{
    /// <summary>
    /// Specifies a simple data type.
    /// </summary>
    // Data type specifiers mainly used in creating text loader and type converter.
    public enum DataKind : byte
    {
        /// <summary>1-byte integer, type of <see cref="System.SByte"/>.</summary>
        SByte = 1,
        /// <summary>1-byte unsigned integer, type of <see cref="System.Byte"/>.</summary>
        Byte = 2,
        /// <summary>2-byte integer, type of <see cref="System.Int16"/>.</summary>
        Int16 = 3,
        /// <summary>2-byte usigned integer, type of <see cref="System.UInt16"/>.</summary>
        UInt16 = 4,
        /// <summary>4-byte integer, type of <see cref="System.Int32"/>.</summary>
        Int32 = 5,
        /// <summary>4-byte usigned integer, type of <see cref="System.UInt32"/>.</summary>
        UInt32 = 6,
        /// <summary>8-byte integer, type of <see cref="System.Int64"/>.</summary>
        Int64 = 7,
        /// <summary>8-byte usigned integer, type of <see cref="System.UInt64"/>.</summary>
        UInt64 = 8,
        /// <summary>4-byte floating-point number, type of <see cref="System.Single"/>.</summary>
        Single = 9,
        /// <summary>8-byte floating-point number, type of <see cref="System.Double"/>.</summary>
        Double = 10,
        /// <summary>string, type of <see cref="System.String"/>.</summary>
        String = 11,
        /// <summary>boolean variable type, type of <see cref="System.Boolean"/>.</summary>
        Boolean = 12,
        /// <summary>type of <see cref="System.TimeSpan"/>.</summary>
        TimeSpan = 13,
        /// <summary>type of <see cref="System.DateTime"/>.</summary>
        DateTime = 14,
        /// <summary>type of <see cref="System.DateTimeOffset"/>.</summary>
        DateTimeOffset = 15,
    }

    /// <summary>
    /// Data type specifier used in command line. <see cref="InternalDataKind"/> is the underlying version of <see cref="DataKind"/>
    /// used for command line and entry point BC.
    /// </summary>
    [BestFriend]
    internal enum InternalDataKind : byte
    {
        // Notes:
        // * These values are serialized, so changing them breaks binary formats.
        // * We intentionally skip zero.
        // * Some code depends on sizeof(DataKind) == sizeof(byte).

        I1 = DataKind.SByte,
        U1 = DataKind.Byte,
        I2 = DataKind.Int16,
        U2 = DataKind.UInt16,
        I4 = DataKind.Int32,
        U4 = DataKind.UInt32,
        I8 = DataKind.Int64,
        U8 = DataKind.UInt64,
        R4 = DataKind.Single,
        R8 = DataKind.Double,
        Num = R4,

        TX = DataKind.String,
#pragma warning disable MSML_GeneralName // The data kind enum has its own logic, independent of C# naming conventions.
        TXT = TX,
        Text = TX,

        BL = DataKind.Boolean,
        Bool = BL,

        TS = DataKind.TimeSpan,
        TimeSpan = TS,
        DT = DataKind.DateTime,
        DateTime = DT,
        DZ = DataKind.DateTimeOffset,
        DateTimeZone = DZ,

        UG = 16, // Unsigned 16-byte integer.
        U16 = UG,
#pragma warning restore MSML_GeneralName
    }

    /// <summary>
    /// Extension methods related to the DataKind enum.
    /// </summary>
    [BestFriend]
    internal static class InternalDataKindExtensions
    {
        public const InternalDataKind KindMin = InternalDataKind.I1;
        public const InternalDataKind KindLim = InternalDataKind.U16 + 1;
        public const int KindCount = KindLim - KindMin;

        /// <summary>
        /// Maps a DataKind to a value suitable for indexing into an array of size KindCount.
        /// </summary>
        public static int ToIndex(this InternalDataKind kind)
        {
            return kind - KindMin;
        }

        /// <summary>
        /// Maps from an index into an array of size KindCount to the corresponding DataKind
        /// </summary>
        public static InternalDataKind FromIndex(int index)
        {
            Contracts.Check(0 <= index && index < KindCount);
            return (InternalDataKind)(index + (int)KindMin);
        }

        /// <summary>
        /// This function converts <paramref name="dataKind"/> to <see cref="InternalDataKind"/>.
        /// Because <see cref="DataKind"/> is a subset of <see cref="InternalDataKind"/>, the conversion is straightforward.
        /// </summary>
        public static InternalDataKind ToInternalDataKind(this DataKind dataKind) => (InternalDataKind)dataKind;

        /// <summary>
        /// This function converts <paramref name="kind"/> to <see cref="DataKind"/>.
        /// Because <see cref="DataKind"/> is a subset of <see cref="InternalDataKind"/>, we should check if <paramref name="kind"/>
        /// can be found in <see cref="DataKind"/>.
        /// </summary>
        public static DataKind ToDataKind(this InternalDataKind kind)
        {
            Contracts.Check(kind != InternalDataKind.UG);
            return (DataKind)kind;
        }

        /// <summary>
        /// For integer DataKinds, this returns the maximum legal value. For un-supported kinds,
        /// it returns zero.
        /// </summary>
        public static ulong ToMaxInt(this InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.I1:
                    return (ulong)sbyte.MaxValue;
                case InternalDataKind.U1:
                    return byte.MaxValue;
                case InternalDataKind.I2:
                    return (ulong)short.MaxValue;
                case InternalDataKind.U2:
                    return ushort.MaxValue;
                case InternalDataKind.I4:
                    return int.MaxValue;
                case InternalDataKind.U4:
                    return uint.MaxValue;
                case InternalDataKind.I8:
                    return long.MaxValue;
                case InternalDataKind.U8:
                    return ulong.MaxValue;
            }

            return 0;
        }

        /// <summary>
        /// For integer Types, this returns the maximum legal value. For un-supported Types,
        /// it returns zero.
        /// </summary>
        public static ulong ToMaxInt(this Type type)
        {
            if (type == typeof(sbyte))
                return (ulong)sbyte.MaxValue;
            else if (type == typeof(byte))
                return byte.MaxValue;
            else if (type == typeof(short))
                return (ulong)short.MaxValue;
            else if (type == typeof(ushort))
                return ushort.MaxValue;
            else if (type == typeof(int))
                return int.MaxValue;
            else if (type == typeof(uint))
                return uint.MaxValue;
            else if (type == typeof(long))
                return long.MaxValue;
            else if (type == typeof(ulong))
                return ulong.MaxValue;

            return 0;
        }

        /// <summary>
        /// For integer DataKinds, this returns the minimum legal value. For un-supported kinds,
        /// it returns one.
        /// </summary>
        public static long ToMinInt(this InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.I1:
                    return sbyte.MinValue;
                case InternalDataKind.U1:
                    return byte.MinValue;
                case InternalDataKind.I2:
                    return short.MinValue;
                case InternalDataKind.U2:
                    return ushort.MinValue;
                case InternalDataKind.I4:
                    return int.MinValue;
                case InternalDataKind.U4:
                    return uint.MinValue;
                case InternalDataKind.I8:
                    return long.MinValue;
                case InternalDataKind.U8:
                    return 0;
            }

            return 1;
        }

        /// <summary>
        /// Maps a DataKind to the associated .Net representation type.
        /// </summary>
        public static Type ToType(this InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.I1:
                    return typeof(sbyte);
                case InternalDataKind.U1:
                    return typeof(byte);
                case InternalDataKind.I2:
                    return typeof(short);
                case InternalDataKind.U2:
                    return typeof(ushort);
                case InternalDataKind.I4:
                    return typeof(int);
                case InternalDataKind.U4:
                    return typeof(uint);
                case InternalDataKind.I8:
                    return typeof(long);
                case InternalDataKind.U8:
                    return typeof(ulong);
                case InternalDataKind.R4:
                    return typeof(Single);
                case InternalDataKind.R8:
                    return typeof(Double);
                case InternalDataKind.TX:
                    return typeof(ReadOnlyMemory<char>);
                case InternalDataKind.BL:
                    return typeof(bool);
                case InternalDataKind.TS:
                    return typeof(TimeSpan);
                case InternalDataKind.DT:
                    return typeof(DateTime);
                case InternalDataKind.DZ:
                    return typeof(DateTimeOffset);
                case InternalDataKind.UG:
                    return typeof(DataViewRowId);
            }

            return null;
        }

        /// <summary>
        /// Try to map a System.Type to a corresponding DataKind value.
        /// </summary>
        public static bool TryGetDataKind(this Type type, out InternalDataKind kind)
        {
            Contracts.CheckValueOrNull(type);

            // REVIEW: Make this more efficient. Should we have a global dictionary?
            if (type == typeof(sbyte))
                kind = InternalDataKind.I1;
            else if (type == typeof(byte))
                kind = InternalDataKind.U1;
            else if (type == typeof(short))
                kind = InternalDataKind.I2;
            else if (type == typeof(ushort))
                kind = InternalDataKind.U2;
            else if (type == typeof(int))
                kind = InternalDataKind.I4;
            else if (type == typeof(uint))
                kind = InternalDataKind.U4;
            else if (type == typeof(long))
                kind = InternalDataKind.I8;
            else if (type == typeof(ulong))
                kind = InternalDataKind.U8;
            else if (type == typeof(Single))
                kind = InternalDataKind.R4;
            else if (type == typeof(Double))
                kind = InternalDataKind.R8;
            else if (type == typeof(ReadOnlyMemory<char>) || type == typeof(string))
                kind = InternalDataKind.TX;
            else if (type == typeof(bool))
                kind = InternalDataKind.BL;
            else if (type == typeof(TimeSpan))
                kind = InternalDataKind.TS;
            else if (type == typeof(DateTime))
                kind = InternalDataKind.DT;
            else if (type == typeof(DateTimeOffset))
                kind = InternalDataKind.DZ;
            else if (type == typeof(DataViewRowId))
                kind = InternalDataKind.UG;
            else
            {
                kind = default(InternalDataKind);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Get the canonical string for a DataKind. Note that using DataKind.ToString() is not stable
        /// and is also slow, so use this instead.
        /// </summary>
        public static string GetString(this InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.I1:
                    return "I1";
                case InternalDataKind.I2:
                    return "I2";
                case InternalDataKind.I4:
                    return "I4";
                case InternalDataKind.I8:
                    return "I8";
                case InternalDataKind.U1:
                    return "U1";
                case InternalDataKind.U2:
                    return "U2";
                case InternalDataKind.U4:
                    return "U4";
                case InternalDataKind.U8:
                    return "U8";
                case InternalDataKind.R4:
                    return "R4";
                case InternalDataKind.R8:
                    return "R8";
                case InternalDataKind.BL:
                    return "BL";
                case InternalDataKind.TX:
                    return "TX";
                case InternalDataKind.TS:
                    return "TS";
                case InternalDataKind.DT:
                    return "DT";
                case InternalDataKind.DZ:
                    return "DZ";
                case InternalDataKind.UG:
                    return "UG";
            }
            return "";
        }
    }
}