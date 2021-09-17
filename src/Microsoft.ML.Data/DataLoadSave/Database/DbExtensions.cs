// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data;

namespace Microsoft.ML.Data
{
    internal static class DbExtensions
    {
        /// <summary>
        /// Maps a <see cref="DbType"/> to the associated .Net representation type.
        /// </summary>
        public static Type ToType(this DbType dbType)
        {
            switch (dbType)
            {
                case DbType.AnsiString:
                case DbType.AnsiStringFixedLength:
                case DbType.String:
                case DbType.StringFixedLength:
                    return typeof(string);
                case DbType.Binary:
                    return typeof(byte[]);
                case DbType.Boolean:
                    return typeof(bool);
                case DbType.Byte:
                    return typeof(byte);
                case DbType.Date:
                case DbType.DateTime:
                case DbType.DateTime2:
                    return typeof(DateTime);
                case DbType.DateTimeOffset:
                    return typeof(DateTimeOffset);
                case DbType.Decimal:
                    return typeof(decimal);
                case DbType.Double:
                    return typeof(double);
                case DbType.Guid:
                    return typeof(Guid);
                case DbType.Int16:
                    return typeof(short);
                case DbType.Int32:
                    return typeof(int);
                case DbType.Int64:
                    return typeof(long);
                case DbType.Object:
                    return typeof(object);
                case DbType.SByte:
                    return typeof(sbyte);
                case DbType.Single:
                    return typeof(float);
                case DbType.Time:
                    return typeof(TimeSpan);
                case DbType.UInt16:
                    return typeof(ushort);
                case DbType.UInt32:
                    return typeof(uint);
                case DbType.UInt64:
                    return typeof(ulong);
                case DbType.Currency:
                case DbType.VarNumeric:
                case DbType.Xml:
                default:
                    return null;
            }
        }

        /// <summary>Maps a <see cref="InternalDataKind"/> to the associated <see cref="DbType"/>.</summary>
        public static DbType ToDbType(this InternalDataKind dataKind)
        {
            switch (dataKind)
            {
                case InternalDataKind.I1:
                    {
                        return DbType.SByte;
                    }

                case InternalDataKind.U1:
                    {
                        return DbType.Byte;
                    }

                case InternalDataKind.I2:
                    {
                        return DbType.Int16;
                    }

                case InternalDataKind.U2:
                    {
                        return DbType.UInt16;
                    }

                case InternalDataKind.I4:
                    {
                        return DbType.Int32;
                    }

                case InternalDataKind.U4:
                    {
                        return DbType.UInt32;
                    }

                case InternalDataKind.I8:
                    {
                        return DbType.Int64;
                    }

                case InternalDataKind.U8:
                    {
                        return DbType.UInt64;
                    }

                case InternalDataKind.R4:
                    {
                        return DbType.Single;
                    }

                case InternalDataKind.R8:
                    {
                        return DbType.Double;
                    }

                case InternalDataKind.TX:
                    {
                        return DbType.String;
                    }

                case InternalDataKind.BL:
                    {
                        return DbType.Boolean;
                    }

                case InternalDataKind.DT:
                    {
                        return DbType.DateTime;
                    }

                default:
                    {
                        throw new NotSupportedException();
                    }
            }
        }
    }
}
