// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(DateTimeTransformer), null, typeof(SignatureLoadModel),
    DateTimeTransformer.UserName, DateTimeTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DateTimeTransformer), null, typeof(SignatureLoadRowMapper),
   DateTimeTransformer.UserName, DateTimeTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(DateTimeTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{

    public static class DateTimeTransformerExtensionClass
    {
        /// <summary>
        /// Create a <see cref="DateTimeEstimator"/>, which splits up the input column specified by <paramref name="inputColumnName"/>
        /// into all its individual datetime components. Input column must be of type Int64 representing the number of seconds since the unix epoch.
        /// This transformer will append the <paramref name="columnPrefix"/> to all the output columns. If you specify a country,
        /// Holiday details will be looked up for that country as well.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="inputColumnName">Input column name</param>
        /// <param name="columnPrefix">Prefix to add to the generated columns</param>
        /// <param name="country">Country name to get holiday details for</param>
        /// <returns><see cref="DateTimeEstimator"/></returns>
        public static DateTimeEstimator FeaturizeDateTime(this TransformsCatalog catalog, string inputColumnName, string columnPrefix, DateTimeEstimator.HolidayList country = DateTimeEstimator.HolidayList.None)
            => new DateTimeEstimator(CatalogUtils.GetEnvironment(catalog), inputColumnName, columnPrefix, country);

        #region ColumnsProduced static extentions

        internal static Type GetRawColumnType(this DateTimeEstimator.ColumnsProduced column)
        {
            switch (column)
            {
                case DateTimeEstimator.ColumnsProduced.Year:
                case DateTimeEstimator.ColumnsProduced.YearIso:
                    return typeof(int);
                case DateTimeEstimator.ColumnsProduced.DayOfYear:
                case DateTimeEstimator.ColumnsProduced.WeekOfMonth:
                    return typeof(ushort);
                case DateTimeEstimator.ColumnsProduced.MonthLabel:
                case DateTimeEstimator.ColumnsProduced.AmPmLabel:
                case DateTimeEstimator.ColumnsProduced.DayOfWeekLabel:
                case DateTimeEstimator.ColumnsProduced.HolidayName:
                    return typeof(ReadOnlyMemory<char>);
                default:
                    return typeof(byte);
            }
        }

        #endregion
    }

    /// <summary>
    /// The DateTimeTransformerEstimator splits up a date into all of its sub parts as individual columns. It generates these fields with a user specified prefix:
    /// int Year, byte Month, byte Day, byte Hour, byte Minute, byte Second, byte AmPm, byte Hour12, byte DayOfWeek, byte DayOfQuarter,
    /// ushort DayOfYear, ushort WeekOfMonth, byte QuarterOfYear, byte HalfOfYear, byte WeekIso, int YearIso, string MonthLabel, string AmPmLabel,
    /// string DayOfWeekLabel, string HolidayName, byte IsPaidTimeOff
    ///
    /// You can optionally specify a country and it will pull holiday information about the country as well
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Int64 |
    /// | Output column data type | Columns and types listed in the summary |
    ///
    /// The <xref:Microsoft.ML.Transforms.DateTimeEstimator> is a trivial estimator and does not need training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="DateTimeTransformerExtensionClass.FeaturizeDateTime(TransformsCatalog, string, string, DateTimeEstimator.HolidayList)"/>
    public sealed class DateTimeEstimator : IEstimator<DateTimeTransformer>
    {
        private readonly Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Input column", Name = "Source", ShortName = "src", SortOrder = 1)]
            public string Source;

            // This transformer adds columns
            [Argument(ArgumentType.Required, HelpText = "Output column prefix", Name = "Prefix", ShortName = "pre", SortOrder = 2)]
            public string Prefix;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Country to get holidays for. Defaults to none if not passed", Name = "Country", ShortName = "ctry", SortOrder = 4)]
            public HolidayList Country = HolidayList.None;
        }

        #endregion

        internal DateTimeEstimator(IHostEnvironment env, string inputColumnName, string columnPrefix, HolidayList country = HolidayList.None)
        {

            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("DateTimeTransformerEstimator");
            _host.CheckValue(inputColumnName, nameof(inputColumnName), "Input column should not be null.");

            _options = new Options
            {
                Source = inputColumnName,
                Prefix = columnPrefix,
                Country = country
            };
        }

        internal DateTimeEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("DateTimeTransformerEstimator");

            _options = options;
        }

        public DateTimeTransformer Fit(IDataView input)
        {
            return new DateTimeTransformer(_host, _options.Source, _options.Prefix, _options.Country, input.Schema);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (ColumnsProduced column in Enum.GetValues(typeof(ColumnsProduced)))
            {
                columns[_options.Prefix + column.ToString()] = new SchemaShape.Column(_options.Prefix + column.ToString(), SchemaShape.Column.VectorKind.Scalar,
                ColumnTypeExtensions.PrimitiveTypeFromType(column.GetRawColumnType()), false, null);
            }

            return new SchemaShape(columns.Values);
        }

        #region Enums
        public enum ColumnsProduced : byte
        {
            Year = 1,
            Month = 2,
            Day = 3,
            Hour = 4,
            Minute = 5,
            Second = 6,
            AmPm = 7,
            Hour12 = 8,
            DayOfWeek = 9,
            DayOfQuarter = 10,
            DayOfYear = 11,
            WeekOfMonth = 12,
            QuarterOfYear = 13,
            HalfOfYear = 14,
            WeekIso = 15,
            YearIso = 16,
            MonthLabel = 17,
            AmPmLabel = 18,
            DayOfWeekLabel = 19,
            HolidayName = 20,
            IsPaidTimeOff = 21
        };

        public enum HolidayList : uint
        {
            None = 1,
            Argentina = 2,
            Australia = 3,
            Austria = 4,
            Belarus = 5,
            Belgium = 6,
            Brazil = 7,
            Canada = 8,
            Colombia = 9,
            Croatia = 10,
            Czech = 11,
            Denmark = 12,
            England = 13,
            Finland = 14,
            France = 15,
            Germany = 16,
            Hungary = 17,
            India = 18,
            Ireland = 19,
            IsleofMan = 20,
            Italy = 21,
            Japan = 22,
            Mexico = 23,
            Netherlands = 24,
            NewZealand = 25,
            NorthernIreland = 26,
            Norway = 27,
            Poland = 28,
            Portugal = 29,
            Scotland = 30,
            Slovenia = 31,
            SouthAfrica = 32,
            Spain = 33,
            Sweden = 34,
            Switzerland = 35,
            Ukraine = 36,
            UnitedKingdom = 37,
            UnitedStates = 38,
            Wales = 39
        }

        #endregion
    }

    public sealed class DateTimeTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Splits a date time value into each individual component";
        internal const string UserName = "DateTime Transform";
        internal const string ShortName = "DateTimeTransform";
        internal const string LoadName = "DateTimeTransform";
        internal const string LoaderSignature = "DateTimeTransform";
        private readonly LongTypedColumn _column;
        private readonly DataViewSchema _schema;

        #endregion

        internal DateTimeTransformer(IHostEnvironment host, string inputColumnName, string columnPrefix, DateTimeEstimator.HolidayList country, DataViewSchema schema) :
            base(host.Register(nameof(DateTimeTransformer)))
        {
            _schema = schema;
            if (_schema[inputColumnName].Type.RawType != typeof(long) &&
                _schema[inputColumnName].Type.RawType != typeof(DateTime))
            {
                throw new Exception($"Unsupported type {_schema[inputColumnName].Type.RawType} for input column ${inputColumnName}. Only long and System.DateTime are supported");
            }

            _column = new LongTypedColumn(inputColumnName, columnPrefix);
            _column.CreateTransformerFromEstimator(country);
        }

        // Factory method for SignatureLoadModel.
        internal DateTimeTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(DateTimeTransformer)))
        {

            Host.CheckValue(ctx, nameof(ctx));
            host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // name of input column
            // column prefix
            // length of C++ state array
            // C++ byte state array

            _column = new LongTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString());

            var dataLength = ctx.Reader.ReadInt32();
            var data = ctx.Reader.ReadByteArray(dataLength);
            _column.CreateTransformerFromSavedData(data);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new DateTimeTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DATETI T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DateTimeTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {

            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // name of input column
            // column prefix
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_column.Source);
            ctx.Writer.Write(_column.Prefix);

            var data = _column.CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        public void Dispose()
        {
            _column.Dispose();
        }

        #region Native Safe handle classes

        internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private readonly DestroyTransformedDataNative _destroyTransformedDataHandler;

            public TransformedDataSafeHandle(IntPtr handle, DestroyTransformedDataNative destroyTransformedDataHandler) : base(true)
            {
                SetHandle(handle);
                _destroyTransformedDataHandler = destroyTransformedDataHandler;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shouldn't ever be one though.
                var success = _destroyTransformedDataHandler(handle, out IntPtr errorHandle);
                Marshal.FreeHGlobal(handle); // Free the memory allocated in C#.
                return success;
            }
        }

        #endregion

        #region TimePoint

        // Exact native representation to get the correct struct size.
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct NativeTimePoint
        {
            public int Year;
            public byte Month;
            public byte Day;
            public byte Hour;
            public byte Minute;
            public byte Second;
            public byte AmPm;
            public byte Hour12;
            public byte DayOfWeek;
            public byte DayOfQuarter;
            public ushort DayOfYear;
            public ushort WeekOfMonth;
            public byte QuarterOfYear;
            public byte HalfOfYear;
            public byte WeekIso;
            public int YearIso;
            public IntPtr MonthLabelPointer;
            public IntPtr AmPmLabelPointer;
            public IntPtr DayOfWeekLabelPointer;
            public IntPtr HolidayNamePointer;
            public byte IsPaidTimeOff;
        }

        #endregion TimePoint

        #region Structs

        [StructLayout(LayoutKind.Sequential)]
        internal struct TimePoint
        {
            public int Year;
            public byte Month;
            public byte Day;
            public byte Hour;
            public byte Minute;
            public byte Second;
            public byte AmPm;
            public byte Hour12;
            public byte DayOfWeek;
            public byte DayOfQuarter;
            public ushort DayOfYear;
            public ushort WeekOfMonth;
            public byte QuarterOfYear;
            public byte HalfOfYear;
            public byte WeekIso;
            public int YearIso;
            public string MonthLabel;
            public string AmPmLabel;
            public string DayOfWeekLabel;
            public string HolidayName;
            public byte IsPaidTimeOff;

            internal TimePoint(ReadOnlySpan<byte> rawData, int intPtrSize)
            {

                int index = 0;

                Year = MemoryMarshal.Read<int>(rawData);
                index += 4;

                Month = rawData[index++];
                Day = rawData[index++];
                Hour = rawData[index++];
                Minute = rawData[index++];
                Second = rawData[index++];
                AmPm = rawData[index++];
                Hour12 = rawData[index++];
                DayOfWeek = rawData[index++];
                DayOfQuarter = rawData[index++];
                DayOfYear = MemoryMarshal.Read<ushort>(rawData.Slice(index));
                index += 2;

                WeekOfMonth = MemoryMarshal.Read<ushort>(rawData.Slice(index));
                index += 2;

                QuarterOfYear = rawData[index++];
                HalfOfYear = rawData[index++];
                WeekIso = rawData[index++];
                YearIso = MemoryMarshal.Read<int>(rawData.Slice(index));
                index += 4;

                // Convert char * to string
                MonthLabel = GetStringFromPointer(ref rawData, ref index, intPtrSize);
                AmPmLabel = GetStringFromPointer(ref rawData, ref index, intPtrSize);
                DayOfWeekLabel = GetStringFromPointer(ref rawData, ref index, intPtrSize);
                HolidayName = GetStringFromPointer(ref rawData, ref index, intPtrSize);
                IsPaidTimeOff = rawData[index];
            }

            // Converts a pointer to a native char* to a string and increments pointer by to the next value.
            // The length of the string is stored at byte* + sizeof(IntPtr).
            private static unsafe string GetStringFromPointer(ref ReadOnlySpan<byte> rawData, ref int index, int intPtrSize)
            {
                string result;

                if (intPtrSize == 4)  // 32 bit machine
                {
                    IntPtr stringData = new IntPtr(MemoryMarshal.Read<int>(rawData.Slice(index)));
                    result = PointerToString(stringData);
                }
                else // 64 bit machine
                {
                    IntPtr stringData = new IntPtr(MemoryMarshal.Read<long>(rawData.Slice(index)));
                    result = PointerToString(stringData);
                }

                index += intPtrSize;

                return result;
            }

        };

        #region Union
        // The folowing structs/enums are to enable us to simulate Unions on the Native side

        // The Type of data in the union
        private enum DateTimeTypeValue : byte
        {
            DateTimeInt64 = 1,  // Posix time
            DateTimeString = 2     // ISO 8601
        };

        // Struct to be able to send string data in the union
        [StructLayout(LayoutKind.Sequential)]
        private struct StringDataType
        {
            internal IntPtr Buffer;
            internal IntPtr BufferSize;
        }

        // Struct to simulate the Union for 32 bit machines. Need to have it defined since the size of the
        // struct will change based on 32/64 bit.
        [StructLayout(LayoutKind.Explicit, Size = 8)]
        private struct DataTypeX32
        {
            [FieldOffset(0)]
            internal long PosixData;

            [FieldOffset(0)]
            internal StringDataType StringData;

        }

        // Struct to simulate the Union for 64 bit machines. Need to have it defined since the size of the
        // struct will change based on 32/64 bit.
        [StructLayout(LayoutKind.Explicit, Size = 16)]
        private struct DataTypeX64
        {
            [FieldOffset(0)]
            internal long PosixData;

            [FieldOffset(0)]
            internal StringDataType StringData;
        }

        // Final struct to send to native code, 32 bit
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct NativeDateTimeParameterX32
        {
            internal DateTimeTypeValue DataType;
            internal DataTypeX32 Data;
        }

        // Final struct to send to native code, 64 bit
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct NativeDateTimeParameterX64
        {
            internal DateTimeTypeValue DataType;
            internal DataTypeX64 Data;
        }

        #endregion Union

        #endregion Structs

        #region ColumnInfo

        #region BaseClass

        internal delegate bool DestroyCppTransformerEstimator(IntPtr estimator, out IntPtr errorHandle);
        internal delegate bool DestroyTransformerSaveData(IntPtr buffer, IntPtr bufferSize, out IntPtr errorHandle);
        internal delegate bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Source;
            internal readonly string Prefix;

            internal unsafe TypedColumn(string source, string prefix)
            {
                Source = source;
                Prefix = prefix;
            }

            internal abstract void CreateTransformerFromEstimator(DateTimeEstimator.HolidayList country);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract unsafe bool CreateEstimatorHelper(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

            public abstract void Dispose();

            private protected unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(DateTimeEstimator.HolidayList country)
            {
                bool success;
                IntPtr errorHandle;
                IntPtr estimator;
                if (country == DateTimeEstimator.HolidayList.None)
                {
                    success = CreateEstimatorHelper(null, null, out estimator, out errorHandle);
                }
                else
                {
                    byte[] dataRoot;

                    if (Directory.Exists(AppDomain.CurrentDomain.BaseDirectory + "/Data/DateTimeFeaturizer"))
                    {
                        dataRoot = Encoding.UTF8.GetBytes(AppDomain.CurrentDomain.BaseDirectory + char.MinValue);
                    }
                    else
                    {
                        dataRoot = Encoding.UTF8.GetBytes(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + char.MinValue);
                    }

                    fixed (byte* dataRootDir = dataRoot)
                    fixed (byte* countryPointer = Encoding.UTF8.GetBytes(Enum.GetName(typeof(DateTimeEstimator.HolidayList), country) + char.MinValue))
                    {
                        success = CreateEstimatorHelper(countryPointer, dataRootDir, out estimator, out errorHandle);
                    }
                }
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }

                using (var estimatorHandler = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    success = CompleteTrainingHelper(estimatorHandler, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    success = CreateTransformerFromEstimatorHelper(estimatorHandler, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }
        }

        internal abstract class TypedColumn<T> : TypedColumn
        {
            internal TypedColumn(string source, string prefix) :
                base(source, prefix)
            {
            }

            internal abstract TimePoint Transform(T input);

        }

        #endregion BaseClass

        #region LongTypedColumn

        internal sealed class LongTypedColumn : TypedColumn<long>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            private readonly int _intPtrSize;
            private readonly int _structSize;
            internal LongTypedColumn(string source, string prefix) :
                base(source, prefix)
            {
                _intPtrSize = IntPtr.Size;

                // The native struct is 25 bytes + 4 size_t.
                unsafe
                {
                    _structSize = sizeof(NativeTimePoint);
                }
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override unsafe void CreateTransformerFromEstimator(DateTimeEstimator.HolidayList country)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(country);
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateTransformerFromSavedDataWithDataRoot"), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, byte* dataRootDir, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                byte[] dataRoot;

                if (Directory.Exists(AppDomain.CurrentDomain.BaseDirectory + "/Data/DateTimeFeaturizer"))
                {
                    dataRoot = Encoding.UTF8.GetBytes(AppDomain.CurrentDomain.BaseDirectory + char.MinValue);
                }
                else
                {
                    dataRoot = Encoding.UTF8.GetBytes(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + char.MinValue);
                }

                fixed (byte* dataRootDir = dataRoot)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, dataRootDir, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNativeX32(TransformerEstimatorSafeHandle transformer, NativeDateTimeParameterX32 input, IntPtr output, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNativeX64(TransformerEstimatorSafeHandle transformer, NativeDateTimeParameterX64 input, IntPtr output, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);
            internal override unsafe TimePoint Transform(long input)
            {
                bool success;
                IntPtr errorHandle;
                IntPtr output = Marshal.AllocHGlobal(_structSize);

                if (IntPtr.Size == 4)
                    success = TransformDataNativeX32(_transformerHandler, new NativeDateTimeParameterX32() { DataType = DateTimeTypeValue.DateTimeInt64, Data = new DataTypeX32() { PosixData = input } }, output, out errorHandle);
                else
                    success = TransformDataNativeX64(_transformerHandler, new NativeDateTimeParameterX64() { DataType = DateTimeTypeValue.DateTimeInt64, Data = new DataTypeX64() { PosixData = input } }, output, out errorHandle);

                if (!success)
                {
                    // If we error on the native side, make sure to free allocated memory.
                    Marshal.FreeHGlobal(output);
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }

                using (var handler = new TransformedDataSafeHandle(output, DestroyTransformedDataNative))
                {
                    unsafe
                    {
                        return new TimePoint(new ReadOnlySpan<byte>(output.ToPointer(), _structSize), _intPtrSize);
                    }
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(countryName, dataRootDir, out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out errorHandle);
        }

        #endregion LongTypedColumn

        #endregion ColumnInfo

        private sealed class Mapper : MapperBase
        {
            private static readonly FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate> _makeGetterMethodInfo
                = FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate>.Create(target => target.MakeGetter<int, int>);

            #region Class data members
            private static readonly DateTime _unixEpoch = new DateTime(1970, 1, 1);

            private readonly DateTimeTransformer _parent;

            #endregion

            public Mapper(DateTimeTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var columns = new List<DataViewSchema.DetachedColumn>();

                foreach (DateTimeEstimator.ColumnsProduced column in Enum.GetValues(typeof(DateTimeEstimator.ColumnsProduced)))
                {
                    columns.Add(new DataViewSchema.DetachedColumn(_parent._column.Prefix + column.ToString(),
                        ColumnTypeExtensions.PrimitiveTypeFromType(column.GetRawColumnType())));
                }

                return columns.ToArray();
            }

            private Delegate MakeGetter<TInput, TTransformed>(DataViewRow input, int iinfo)
            {
                // If already in posix time.
                if (typeof(TInput) == typeof(long))
                    return MakeLongGetter<TTransformed>(input, iinfo);
                // System.DateTime
                else
                    return MakeDateTimeGetter<TTransformed>(input, iinfo);
            }

            private Delegate MakeLongGetter<TTransformed>(DataViewRow input, int iinfo)
            {
                var getter = input.GetGetter<long>(input.Schema[_parent._column.Source]);
                ValueGetter<TTransformed> result = (ref TTransformed dst) =>
                {
                    long dateTime = default;
                    getter(ref dateTime);

                    var timePoint = _parent._column.Transform(dateTime);

                    dst = GetColumnFromStruct<TTransformed>(ref timePoint, iinfo);
                };

                return result;
            }

            private Delegate MakeDateTimeGetter<TTransformed>(DataViewRow input, int iinfo)
            {
                var getter = input.GetGetter<DateTime>(input.Schema[_parent._column.Source]);
                ValueGetter<TTransformed> result = (ref TTransformed dst) =>
                {
                    DateTime dateTime = default;
                    getter(ref dateTime);

                    var timePoint = _parent._column.Transform(dateTime.Subtract(_unixEpoch).Ticks / TimeSpan.TicksPerSecond);

                    dst = GetColumnFromStruct<TTransformed>(ref timePoint, iinfo);
                };

                return result;
            }

            private TTransformed GetColumnFromStruct<TTransformed>(ref TimePoint timePoint, int iinfo)
            {
                if (iinfo == 0)
                    return (TTransformed)Convert.ChangeType(timePoint.Year, typeof(TTransformed));
                else if (iinfo == 1)
                    return (TTransformed)Convert.ChangeType(timePoint.Month, typeof(TTransformed));
                else if (iinfo == 2)
                    return (TTransformed)Convert.ChangeType(timePoint.Day, typeof(TTransformed));
                else if (iinfo == 3)
                    return (TTransformed)Convert.ChangeType(timePoint.Hour, typeof(TTransformed));
                else if (iinfo == 4)
                    return (TTransformed)Convert.ChangeType(timePoint.Minute, typeof(TTransformed));
                else if (iinfo == 5)
                    return (TTransformed)Convert.ChangeType(timePoint.Second, typeof(TTransformed));
                else if (iinfo == 6)
                    return (TTransformed)Convert.ChangeType(timePoint.AmPm, typeof(TTransformed));
                else if (iinfo == 7)
                    return (TTransformed)Convert.ChangeType(timePoint.Hour12, typeof(TTransformed));
                else if (iinfo == 8)
                    return (TTransformed)Convert.ChangeType(timePoint.DayOfWeek, typeof(TTransformed));
                else if (iinfo == 9)
                    return (TTransformed)Convert.ChangeType(timePoint.DayOfQuarter, typeof(TTransformed));
                else if (iinfo == 10)
                    return (TTransformed)Convert.ChangeType(timePoint.DayOfYear, typeof(TTransformed));
                else if (iinfo == 11)
                    return (TTransformed)Convert.ChangeType(timePoint.WeekOfMonth, typeof(TTransformed));
                else if (iinfo == 12)
                    return (TTransformed)Convert.ChangeType(timePoint.QuarterOfYear, typeof(TTransformed));
                else if (iinfo == 13)
                    return (TTransformed)Convert.ChangeType(timePoint.HalfOfYear, typeof(TTransformed));
                else if (iinfo == 14)
                    return (TTransformed)Convert.ChangeType(timePoint.WeekIso, typeof(TTransformed));
                else if (iinfo == 15)
                    return (TTransformed)Convert.ChangeType(timePoint.YearIso, typeof(TTransformed));
                else if (iinfo == 16)
                    return (TTransformed)Convert.ChangeType(timePoint.MonthLabel.AsMemory(), typeof(TTransformed));
                else if (iinfo == 17)
                    return (TTransformed)Convert.ChangeType(timePoint.AmPmLabel.AsMemory(), typeof(TTransformed));
                else if (iinfo == 18)
                    return (TTransformed)Convert.ChangeType(timePoint.DayOfWeekLabel.AsMemory(), typeof(TTransformed));
                else if (iinfo == 19)
                    return (TTransformed)Convert.ChangeType(timePoint.HolidayName.AsMemory(), typeof(TTransformed));
                else
                    return (TTransformed)Convert.ChangeType(timePoint.IsPaidTimeOff, typeof(TTransformed));
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;

                // Have to add 1 to iinfo since the enum starts at 1
                return Utils.MarshalInvoke(_makeGetterMethodInfo, this, input.Schema[_parent._column.Source].Type.RawType, ((DateTimeEstimator.ColumnsProduced)iinfo + 1).GetRawColumnType(), input, iinfo);

            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (InputSchema[i].Name.Equals(_parent._column.Source))
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class DateTimeTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.DateTimeSplitter",
            Desc = DateTimeTransformer.Summary,
            UserName = DateTimeTransformer.UserName,
            ShortName = DateTimeTransformer.ShortName)]
        public static CommonOutputs.TransformOutput DateTimeSplit(IHostEnvironment env, DateTimeEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, DateTimeTransformer.ShortName, input);
            var xf = new DateTimeEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
