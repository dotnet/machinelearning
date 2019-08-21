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
        /// Create a <see cref="DateTimeTransformerEstimator"/>, which splits up the input column specified by <paramref name="inputColumnName"/>
        /// into all its individual datetime components. Input column must be of type Int64 representing the number of seconds since the unix epoc.
        /// This transformer will append the <paramref name="columnPrefix"/> to all the output columns. If <paramref name="columnsToDrop"/> is empty,
        /// then all the columns are returned. Otherwise, the columns listed in the array will be dropped from the return value.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="inputColumnName">Input column name</param>
        /// <param name="columnPrefix">Prefix to add to the generated columns</param>
        /// <param name="columnsToDrop">List of columns to drop, if any</param>
        /// <returns><see cref="DateTimeTransformerEstimator"/></returns>
        public static DateTimeTransformerEstimator DateTimeTransformer(this TransformsCatalog catalog, string inputColumnName, string columnPrefix, params DateTimeTransformerEstimator.ColumnsProduced[] columnsToDrop)
            => new DateTimeTransformerEstimator(CatalogUtils.GetEnvironment(catalog), inputColumnName, columnPrefix, columnsToDrop);

        /// <summary>
        /// Create a <see cref="DateTimeTransformerEstimator"/>, which splits up the input column specified by <paramref name="inputColumnName"/>
        /// into all its individual datetime components. Input column must be of type Int64 representing the number of seconds since the unix epoc.
        /// This transformer will append the <paramref name="columnPrefix"/> to all the output columns. If <paramref name="columnsToDrop"/> is empty,
        /// then all the columns are returned. Otherwise, the columns listed in the array will be dropped from the return value. If you specify a country,
        /// Holiday details will be looked up for that country as well.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="inputColumnName">Input column name</param>
        /// <param name="columnPrefix">Prefix to add to the generated columns</param>
        /// <param name="columnsToDrop">List of columns to drop, if any</param>
        /// <param name="country">Country name to get holiday details for</param>
        /// <returns><see cref="DateTimeTransformerEstimator"/></returns>
        public static DateTimeTransformerEstimator DateTimeTransformer(this TransformsCatalog catalog, string inputColumnName, string columnPrefix, DateTimeTransformerEstimator.ColumnsProduced[] columnsToDrop = null, DateTimeTransformerEstimator.Countries country = DateTimeTransformerEstimator.Countries.None)
            => new DateTimeTransformerEstimator(CatalogUtils.GetEnvironment(catalog), inputColumnName, columnPrefix, columnsToDrop, country);

        #region ColumnsProduced static extentions

        internal static Type GetRawColumnType(this DateTimeTransformerEstimator.ColumnsProduced column)
        {
            switch (column)
            {
                case DateTimeTransformerEstimator.ColumnsProduced.Year:
                case DateTimeTransformerEstimator.ColumnsProduced.YearIso:
                    return typeof(int);
                case DateTimeTransformerEstimator.ColumnsProduced.DayOfYear:
                case DateTimeTransformerEstimator.ColumnsProduced.WeekOfMonth:
                    return typeof(ushort);
                case DateTimeTransformerEstimator.ColumnsProduced.MonthLabel:
                case DateTimeTransformerEstimator.ColumnsProduced.AmPmLabel:
                case DateTimeTransformerEstimator.ColumnsProduced.DayOfWeekLabel:
                case DateTimeTransformerEstimator.ColumnsProduced.HolidayName:
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
    /// The <xref:Microsoft.ML.Transforms.CategoryImputerEstimator> is a trivial estimator and does not need training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="DateTimeTransformerExtensionClass.DateTimeTransformer(TransformsCatalog, string, string, DateTimeTransformerEstimator.ColumnsProduced[])"/>
    /// <seealso cref="DateTimeTransformerExtensionClass.DateTimeTransformer(TransformsCatalog, string, string, DateTimeTransformerEstimator.ColumnsProduced[], DateTimeTransformerEstimator.Countries)"/>
    public sealed class DateTimeTransformerEstimator : IEstimator<DateTimeTransformer>
    {
        private readonly Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options: TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Input column", Name = "Source", ShortName = "src", SortOrder = 1)]
            public string Source;

            // This transformer adds columns
            [Argument(ArgumentType.Required, HelpText = "Output column prefix", Name = "Prefix", ShortName = "pre", SortOrder = 2)]
            public string Prefix;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to drop after the DateTime Expansion", Name = "ColumnsToDrop", ShortName = "drop", SortOrder = 3)]
            public ColumnsProduced[] ColumnsToDrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Country to get holidays for. Defaults to none if not passed", Name = "Country", ShortName = "ctry", SortOrder = 4)]
            public Countries Country = Countries.None;
        }

        #endregion

        public DateTimeTransformerEstimator(IHostEnvironment env, string inputColumnName, string columnPrefix, ColumnsProduced[] columnsToDrop, Countries country = Countries.None)
        {

            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("DateTimeTransformerEstimator");
            _host.CheckValue(inputColumnName, nameof(inputColumnName), "Input column should not be null.");

            _options = new Options
            {
                Source = inputColumnName,
                Prefix = columnPrefix,
                ColumnsToDrop = columnsToDrop == null ? Array.Empty<ColumnsProduced>() : columnsToDrop,
                Country = country
            };
        }

        internal DateTimeTransformerEstimator(IHostEnvironment env, Options options)
        {

            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("DateTimeTransformerEstimator");

            _options = options;
            _options.ColumnsToDrop = _options.ColumnsToDrop == null ? Array.Empty<ColumnsProduced>() : _options.ColumnsToDrop;
        }

        public DateTimeTransformer Fit(IDataView input)
        {
            return new DateTimeTransformer(_host, _options.Source, _options.Prefix, _options.ColumnsToDrop, _options.Country);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (ColumnsProduced column in Enum.GetValues(typeof(ColumnsProduced)))
                if (_options.ColumnsToDrop == null || !_options.ColumnsToDrop.Contains(column))
            {
                    columns[_options.Prefix + column.ToString()] = new SchemaShape.Column(_options.Prefix + column.ToString(), SchemaShape.Column.VectorKind.Scalar,
                    ColumnTypeExtensions.PrimitiveTypeFromType(column.GetRawColumnType()), false, null);
            }

            return new SchemaShape(columns.Values);
        }

        #region Enums
        public enum ColumnsProduced : byte
        {
            Year = 1, Month, Day, Hour, Minute, Second, AmPm, Hour12, DayOfWeek, DayOfQuarter, DayOfYear,
            WeekOfMonth, QuarterOfYear, HalfOfYear, WeekIso, YearIso, MonthLabel, AmPmLabel, DayOfWeekLabel,
            HolidayName, IsPaidTimeOff
        };

        public enum Countries : byte
        {
            None = 1,
            Argentina, Australia, Austria, Belarus, Belgium, Brazil, Canada, Colombia, Croatia, Czech, Denmark,
            England, Finland, France, Germany, Hungary, India, Ireland, IsleofMan, Italy, Japan, Mexico, Netherlands,
            NewZealand, NorthernIreland, Norway, Poland, Portugal, Scotland, Slovenia, SouthAfrica, Spain, Sweden, Switzerland,
            Ukraine, UnitedKingdom, UnitedStates, Wales
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
        private DateTimeTypedColumn _column;

        private DateTimeTransformerEstimator.ColumnsProduced[] _columnsToDrop;
        private byte[] _activeColumnMapping;

        #endregion

        public DateTimeTransformer(IHostEnvironment env, string inputColumnName, string columnPrefix, DateTimeTransformerEstimator.ColumnsProduced[] columnsToDrop, DateTimeTransformerEstimator.Countries country ) :
            base(env.Register(nameof(DateTimeTransformer)))
        {

            _columnsToDrop = columnsToDrop;
            var activeColumnLength = Enum.GetValues(typeof(DateTimeTransformerEstimator.ColumnsProduced)).Length - (_columnsToDrop == null ? 0 : _columnsToDrop.Length);
            _activeColumnMapping = new byte[activeColumnLength];
            var index = 0;
            foreach(DateTimeTransformerEstimator.ColumnsProduced column in Enum.GetValues(typeof(DateTimeTransformerEstimator.ColumnsProduced)))
            {
                if (_columnsToDrop == null || !_columnsToDrop.Contains(column))
                {
                    _activeColumnMapping[index++] = (byte)column;
                }
            }

            _column = new DateTimeTypedColumn(inputColumnName, columnPrefix);
            _column.CreateTransformerFromEstimator(country);
        }

        // Factory method for SignatureLoadModel.
        internal DateTimeTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(DateTimeTransformer)))
        {

            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // name of input column
            // column prefix
            // byte length of columns to drop array
            // byte array of columns to drop
            // length of C++ state array
            // C++ byte state array

            _column = new DateTimeTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString());

            var dropColumnsLength = ctx.Reader.ReadInt32();
            if (dropColumnsLength > 0)
            {
                _columnsToDrop = new DateTimeTransformerEstimator.ColumnsProduced[dropColumnsLength];
                //read in enum bytes
                for (int i = 0; i < dropColumnsLength; i++)
                    _columnsToDrop[i] = (DateTimeTransformerEstimator.ColumnsProduced)ctx.Reader.ReadByte();
            }

            _activeColumnMapping = new byte[Enum.GetValues(typeof(DateTimeTransformerEstimator.ColumnsProduced)).Length - dropColumnsLength];
            var index = 0;
            foreach (DateTimeTransformerEstimator.ColumnsProduced column in Enum.GetValues(typeof(DateTimeTransformerEstimator.ColumnsProduced)))
            {
                if (_columnsToDrop == null || !_columnsToDrop.Contains(column))
                {
                    _activeColumnMapping[index++] = (byte)column;
                }
            }

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
            // byte length of columns to drop array
            // byte array of columns to drop
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_column.Source);
            ctx.Writer.Write(_column.Prefix);

            ctx.Writer.Write(_columnsToDrop == null ? 0 : _columnsToDrop.Length);
            if (_columnsToDrop != null)
            {
                foreach (var toDrop in _columnsToDrop)
                    ctx.Writer.Write((byte)toDrop);
            }

            var data = _column.CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        public void Dispose()
        {
            _column.Dispose();
        }

        #region C++ Safe handle classes

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
                // Not sure what to do with error stuff here.  There shoudln't ever be one though.
                return _destroyTransformedDataHandler(handle, out IntPtr errorHandle);
            }
        }

        #endregion

        #region TimePoint

        [StructLayoutAttribute(LayoutKind.Sequential)]
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

            internal unsafe TimePoint(byte* rawData)
            {
                int intPtrSize = sizeof(IntPtr);

                Year = *(int*)rawData;
                rawData += 4;

                Month = *rawData++;
                Day = *rawData++;
                Hour = *rawData++;
                Minute = *rawData++;
                Second = *rawData++;
                AmPm = *rawData++;
                Hour12 = *rawData++;
                DayOfWeek = *rawData++;
                DayOfQuarter = *rawData++;
                DayOfYear = *(ushort*)rawData;
                rawData += 2;

                WeekOfMonth = *(ushort*)rawData;
                rawData += 2;

                QuarterOfYear = *rawData++;
                HalfOfYear = *rawData++;
                WeekIso = *rawData++;
                YearIso = *(int*)rawData;
                rawData += 4;

                // Convert char * to string
                MonthLabel = GetStringFromPointer(ref rawData, intPtrSize);
                AmPmLabel = GetStringFromPointer(ref rawData, intPtrSize);
                DayOfWeekLabel = GetStringFromPointer(ref rawData, intPtrSize);
                HolidayName = GetStringFromPointer(ref rawData, intPtrSize);
                IsPaidTimeOff = *rawData;
            }

            // Converts a pointer to a native char* to a string and increments pointer by to the next value.
            // The length of the string is stored at byte* + sizeof(IntPtr).
            private static unsafe string GetStringFromPointer(ref byte* rawData, int intPtrSize)
            {
                byte[] buffer;
                if (intPtrSize == 4) // 32 bit machine
                    buffer = new byte[*(uint*)(rawData + intPtrSize)];
                else // 64 bit machine
                    buffer = new byte[*(ulong*)(rawData + intPtrSize)];

                if (buffer.Length == 0)
                {
                    rawData += intPtrSize * 2;
                    return string.Empty;
                }

                Marshal.Copy(new IntPtr(*(int**)rawData), buffer, 0, buffer.Length);
                rawData += intPtrSize * 2;

                return Encoding.UTF8.GetString(buffer);
            }

        };

        #endregion

        #region BaseClass

        internal delegate bool DestroyCppTransformerEstimator(IntPtr estimator, out IntPtr errorHandle);
        internal delegate bool DestroyTransformerSaveData(IntPtr buffer, IntPtr bufferSize, out IntPtr errorHandle);
        internal delegate bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Source;
            internal readonly string Prefix;

            internal TypedColumn(string source, string prefix)
            {
                Source = source;
                Prefix = prefix;
            }

            internal abstract void CreateTransformerFromEstimator(DateTimeTransformerEstimator.Countries country);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected unsafe abstract bool CreateEstimatorHelper(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            public abstract void Dispose();

            private protected unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(DateTimeTransformerEstimator.Countries country)
            {
                bool success;
                IntPtr errorHandle;
                IntPtr estimator;
                if (country == DateTimeTransformerEstimator.Countries.None)
                {
                    success = CreateEstimatorHelper(null, null, out estimator, out errorHandle);
                }
                else
                {
                    fixed (byte* dataRootDir = Encoding.UTF8.GetBytes(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + char.MinValue))
                    fixed (byte* countryPointer = Encoding.UTF8.GetBytes(Enum.GetName(typeof(DateTimeTransformerEstimator.Countries), country) + char.MinValue))
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

                    success = CreateTransformerFromEstimatorHelper(estimatorHandler, out IntPtr transformer, out errorHandle);
                    if (!success)
                    {
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }

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

        #endregion

        #region DateTimeTypedColumn

        internal sealed class DateTimeTypedColumn : TypedColumn<long>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal DateTimeTypedColumn(string source, string prefix) :
                base(source, prefix)
            {
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateEstimatorNative(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override unsafe void CreateTransformerFromEstimator(DateTimeTransformerEstimator.Countries country)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(country);
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_CreateTransformerFromSavedDataWithDataRoot"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, byte* dataRootDir, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                fixed (byte* dataRootDir = Encoding.UTF8.GetBytes(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + char.MinValue))
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, dataRootDir, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }
            }

            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long input, out IntPtr output, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "DateTimeFeaturizer_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);
            internal override TimePoint Transform(long input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var handler = new TransformedDataSafeHandle(output, DestroyTransformedDataNative))
                {
                    unsafe
                    {
                        return new TimePoint((byte*)output.ToPointer());
                    }
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected unsafe override bool CreateEstimatorHelper(byte* countryName, byte* dataRootDir, out IntPtr estimator, out IntPtr errorHandle) =>
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
        }

        #endregion

        private sealed class Mapper : MapperBase
        {

            #region Class data members

            private readonly DateTimeTransformer _parent;
            private ConcurrentDictionary<long, TimePoint> _cache;
            private ConcurrentQueue<long> _oldestKeys;

            #endregion

            public Mapper(DateTimeTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _cache = new ConcurrentDictionary<long, TimePoint>();
                _oldestKeys = new ConcurrentQueue<long>();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var columns = new List<DataViewSchema.DetachedColumn>();

                foreach (DateTimeTransformerEstimator.ColumnsProduced column in Enum.GetValues(typeof(DateTimeTransformerEstimator.ColumnsProduced)))
                    if (_parent._columnsToDrop == null || !_parent._columnsToDrop.Contains(column))
                {
                    columns.Add(new DataViewSchema.DetachedColumn(_parent._column.Prefix + column.ToString(),
                        ColumnTypeExtensions.PrimitiveTypeFromType(column.GetRawColumnType())));
                }

                return columns.ToArray();
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo)
            {
                ValueGetter<T> result = (ref T dst) =>
                {
                    long dateTime = default;
                    var getter = input.GetGetter<long>(input.Schema[_parent._column.Source]);
                    getter(ref dateTime);

                    if (!_cache.TryGetValue(dateTime, out TimePoint timePoint)){
                        _cache[dateTime] = _parent._column.Transform(dateTime);
                        _oldestKeys.Enqueue(dateTime);
                        timePoint = _cache[dateTime];

                        // If more than 100 cached items, remove 20
                        if(_cache.Count > 100)
                        {
                            for(int i = 0; i < 20; i++)
                            {
                                long key;
                                while (!_oldestKeys.TryDequeue(out key)) { }
                                while (!_cache.TryRemove(key, out TimePoint removedValue)) { }
                            }
                        }
                    }

                    if (iinfo == 0)
                        dst = (T)Convert.ChangeType(timePoint.Year, typeof(T));
                    else if (iinfo == 1)
                        dst = (T)Convert.ChangeType(timePoint.Month, typeof(T));
                    else if (iinfo == 2)
                        dst = (T)Convert.ChangeType(timePoint.Day, typeof(T));
                    else if (iinfo == 3)
                        dst = (T)Convert.ChangeType(timePoint.Hour, typeof(T));
                    else if (iinfo == 4)
                        dst = (T)Convert.ChangeType(timePoint.Minute, typeof(T));
                    else if (iinfo == 5)
                        dst = (T)Convert.ChangeType(timePoint.Second, typeof(T));
                    else if (iinfo == 6)
                        dst = (T)Convert.ChangeType(timePoint.AmPm, typeof(T));
                    else if (iinfo == 7)
                        dst = (T)Convert.ChangeType(timePoint.Hour12, typeof(T));
                    else if (iinfo == 8)
                        dst = (T)Convert.ChangeType(timePoint.DayOfWeek, typeof(T));
                    else if (iinfo == 9)
                        dst = (T)Convert.ChangeType(timePoint.DayOfQuarter, typeof(T));
                    else if (iinfo == 10)
                        dst = (T)Convert.ChangeType(timePoint.DayOfYear, typeof(T));
                    else if (iinfo == 11)
                        dst = (T)Convert.ChangeType(timePoint.WeekOfMonth, typeof(T));
                    else if (iinfo == 12)
                        dst = (T)Convert.ChangeType(timePoint.QuarterOfYear, typeof(T));
                    else if (iinfo == 13)
                        dst = (T)Convert.ChangeType(timePoint.HalfOfYear, typeof(T));
                    else if (iinfo == 14)
                        dst = (T)Convert.ChangeType(timePoint.WeekIso, typeof(T));
                    else if (iinfo == 15)
                        dst = (T)Convert.ChangeType(timePoint.YearIso, typeof(T));
                    else if (iinfo == 16)
                        dst = (T)Convert.ChangeType(timePoint.MonthLabel.AsMemory(), typeof(T));
                    else if (iinfo == 17)
                        dst = (T)Convert.ChangeType(timePoint.AmPmLabel.AsMemory(), typeof(T));
                    else if (iinfo == 18)
                        dst = (T)Convert.ChangeType(timePoint.DayOfWeekLabel.AsMemory(), typeof(T));
                    else if (iinfo == 19)
                        dst = (T)Convert.ChangeType(timePoint.HolidayName.AsMemory(), typeof(T));
                    else
                        dst = (T)Convert.ChangeType(timePoint.IsPaidTimeOff, typeof(T));
                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;

                var outputColumn = (int)_parent._activeColumnMapping[iinfo];

                // Have to subtract 1 from the output column since the enum starts and 1 and not 0.
                return Utils.MarshalInvoke(MakeGetter<int>, ((DateTimeTransformerEstimator.ColumnsProduced)outputColumn).GetRawColumnType(), input, outputColumn - 1);
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
        public static CommonOutputs.TransformOutput DateTimeSplit(IHostEnvironment env, DateTimeTransformerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, DateTimeTransformer.ShortName, input);
            var xf = new DateTimeTransformerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
