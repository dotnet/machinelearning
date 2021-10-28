// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(TimeSeriesImputerTransformer), null, typeof(SignatureLoadModel),
    TimeSeriesImputerTransformer.UserName, TimeSeriesImputerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(TimeSeriesImputerTransformer), null, typeof(SignatureLoadDataTransform),
   TimeSeriesImputerTransformer.UserName, TimeSeriesImputerTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TimeSeriesTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class TimeSeriesImputerExtensionClass
    {
        /// <summary>
        /// Create a <see cref="TimeSeriesImputerEstimator"/>, Imputes missing rows and column data per grain. Operates on all columns in the IDataView.
        /// Currently only float/double/string columns are supported for imputation strategies, and an empty string is considered "missing" for the
        /// purpose of this estimator. Other column types will have the default value placed if a row is imputed.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="timeSeriesColumn">Column representing the time series. Should be of type <see cref="long"/> or <see cref="System.DateTime"/></param>
        /// <param name="grainColumns">List of columns to use as grains</param>
        /// <param name="imputeMode">Mode of imputation for missing values in column. If not passed defaults to forward fill</param>
        public static TimeSeriesImputerEstimator ReplaceMissingTimeSeriesValues(this TransformsCatalog catalog, string timeSeriesColumn, string[] grainColumns,
            TimeSeriesImputerEstimator.ImputationStrategy imputeMode = TimeSeriesImputerEstimator.ImputationStrategy.ForwardFill)
            => new TimeSeriesImputerEstimator(CatalogUtils.GetEnvironment(catalog), timeSeriesColumn, grainColumns, null, TimeSeriesImputerEstimator.FilterMode.NoFilter, imputeMode, true);

        /// <summary>
        /// Create a <see cref="TimeSeriesImputerEstimator"/>, Imputes missing rows and column data per grain. Applies the imputation strategy on
        /// a filtered list of columns in the IDataView. Columns that are excluded will have the default value for that data type used when a row
        /// is imputed. Currently only float/double/string columns are supported for imputation strategies, and an empty string is considered "missing" for the
        /// purpose of this estimator.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="timeSeriesColumn">Column representing the time series. Should be of type <see cref="long"/> or <see cref="System.DateTime"/></param>
        /// <param name="grainColumns">List of columns to use as grains</param>
        /// <param name="filterColumns">List of columns to filter. If <paramref name="filterMode"/> is <see cref="TimeSeriesImputerEstimator.FilterMode.Exclude"/> than columns in the list will be ignored.
        /// If <paramref name="filterMode"/> is <see cref="TimeSeriesImputerEstimator.FilterMode.Include"/> than values in the list are the only columns imputed.</param>
        /// <param name="filterMode">Whether the list <paramref name="filterColumns"/> should include or exclude those columns.</param>
        /// <param name="imputeMode">Mode of imputation for missing values in column. If not passed defaults to forward fill</param>
        /// <param name="suppressTypeErrors">Suppress the errors that would occur if a column and impute mode are incompatible. If true, will skip the column and use the default value. If false, will stop and throw an error.</param>
        public static TimeSeriesImputerEstimator ReplaceMissingTimeSeriesValues(this TransformsCatalog catalog, string timeSeriesColumn,
            string[] grainColumns, string[] filterColumns, TimeSeriesImputerEstimator.FilterMode filterMode = TimeSeriesImputerEstimator.FilterMode.Exclude,
            TimeSeriesImputerEstimator.ImputationStrategy imputeMode = TimeSeriesImputerEstimator.ImputationStrategy.ForwardFill,
            bool suppressTypeErrors = false)
            => new TimeSeriesImputerEstimator(CatalogUtils.GetEnvironment(catalog), timeSeriesColumn, grainColumns, filterColumns, filterMode, imputeMode, suppressTypeErrors);
    }

    /// <summary>
    /// Imputes missing rows and column data per grain, based on the dates in the date column.
    ///
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | All Types |
    /// | Output column data type | All Types |
    /// | Exportable to ONNX | No |
    ///
    /// The TimeSeriesImputer imputes missing rows and column data per grain (category), based on the dates in the date column. This operation needs to happen to every column in the IDataView,
    /// If you "filter" a column using the filterColumns and filterMode parameters, if a row is imputed the default value for that type will be used.
    /// Currently only float/double/string columns are supported for imputation strategies, and an empty string is considered "missing" for the
    /// purpose of this estimator. A new column is added to the schema after this operation is run. The column is called "IsRowImputed" and is a
    /// boolean value representing if the row was created as a result of this operation or not.
    ///
    /// The imputation strategies that are currently supported are ForwardFill, where the last good value is propagated forward, Backfill, where the next good value is propagated backwards,
    /// and Median, where the mathematical median is used to fill in missing values.
    ///
    /// NOTE: It is not recommended to chain this multiple times. If a column is filtered, the default value is placed when a row is imputed, and the
    /// default value is not null. Thus any other TimeSeriesImputers will not be able to replace those values anymore causing essentially a very
    /// computationally expensive NO-OP.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TimeSeriesImputerExtensionClass.ReplaceMissingTimeSeriesValues(TransformsCatalog, string, string[], ImputationStrategy)"/>
    /// <seealso cref="TimeSeriesImputerExtensionClass.ReplaceMissingTimeSeriesValues(TransformsCatalog, string, string[], string[], FilterMode, ImputationStrategy, bool)"/>
    public sealed class TimeSeriesImputerEstimator : IEstimator<TimeSeriesImputerTransformer>
    {
        private readonly Options _options;
        internal const string IsRowImputedColumnName = "IsRowImputed";

        private readonly IHost _host;
        private static readonly List<Type> _currentSupportedTypes = new List<Type> { typeof(sbyte), typeof(byte), typeof(short), typeof(ushort), typeof(int), typeof(uint),
            typeof(long), typeof(ulong), typeof(float), typeof(double), typeof(string), typeof(ReadOnlyMemory<char>), typeof(DateTime)};

        #region Options
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Column representing the time", Name = "TimeSeriesColumn", ShortName = "time", SortOrder = 1)]
            public string TimeSeriesColumn;

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns", Name = "GrainColumns", ShortName = "grains", SortOrder = 2)]
            public string[] GrainColumns;

            // This transformer adds columns
            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to filter", Name = "FilterColumns", ShortName = "filters", SortOrder = 2)]
            public string[] FilterColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Filter mode. Either include or exclude", Name = "FilterMode", ShortName = "fmode", SortOrder = 3)]
            public FilterMode FilterMode = FilterMode.Exclude;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Mode for imputing, defaults to ForwardFill if not provided", Name = "ImputeMode", ShortName = "mode", SortOrder = 3)]
            public ImputationStrategy ImputeMode = ImputationStrategy.ForwardFill;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Suppress the errors that would occur if a column and impute mode are incompatible. If true, will skip the column. If false, will stop and throw an error.", Name = "SupressTypeErrors", ShortName = "error", SortOrder = 3)]
            public bool SupressTypeErrors = false;
        }

        #endregion

        #region Class Enums

        /// <summary>
        /// This is the representation of which Imputation Strategy to use.
        /// ForwardFill takes the value from the last good row and propagates it forward anytime a row is imputed or a missing value is found.
        /// BackFill is the same as ForwardFill, except it takes from the next good row and propagates backwards.
        /// Median only supports float/double, takes the median value found during training and uses that to replace missing values
        /// </summary>
        public enum ImputationStrategy : byte
        {
            /// <summary>
            /// Takes the value from the last good row and propagates it forward anytime a row is imputed or a missing value is found.
            /// </summary>
            ForwardFill = 1,

            /// <summary>
            /// Takes the value from the next good row and propagates it backwards anytime a row is imputed or a missing value is found.
            /// </summary>
            BackFill = 2,

            /// <summary>
            /// Takes the median found during training and propagates that anytime a row is imputed or a missing value is found.
            /// </summary>
            Median = 3,
            // Interpolate = 4, interpolate not currently supported in the native code.
        };

        /// <summary>
        /// Method by which columns are selected for imputing values.
        /// NoFilter takes all of the columns so you don't have to specify anything.
        /// Include only does the specified ImputationStrategy on the columns you specify. The other columns will get a default value.
        /// Exclude is the exact opposite of Include, and does the ImputationStrategy on all columns but the ones you specify, which will get the default value.
        /// </summary>
        public enum FilterMode : byte
        {
            /// <summary>
            /// Takes all of the columns so you don't have to specify anything.
            /// </summary>
            NoFilter = 1,

            /// <summary>
            /// Only does the specified ImputationStrategy on the columns you specify. The other columns will get a default value.
            /// </summary>
            Include = 2,

            /// <summary>
            /// Does the ImputationStrategy on all columns but the ones you specify, which will get the default value.
            /// </summary>
            Exclude = 3
        };

        #endregion

        internal TimeSeriesImputerEstimator(IHostEnvironment env, string timeSeriesColumn, string[] grainColumns, string[] filterColumns, FilterMode filterMode, ImputationStrategy imputeMode, bool supressTypeErrors)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("TimeSeriesImputerEstimator");
            _host.CheckValue(timeSeriesColumn, nameof(timeSeriesColumn), "TimePoint column should not be null.");
            _host.CheckNonEmpty(grainColumns, nameof(grainColumns), "Need at least one grain column.");
            if (filterMode == FilterMode.Include)
                _host.CheckNonEmpty(filterColumns, nameof(filterColumns), "Need at least 1 filter column if a FilterMode is specified");

            _options = new Options
            {
                TimeSeriesColumn = timeSeriesColumn,
                GrainColumns = grainColumns,
                FilterColumns = filterColumns == null ? new string[] { } : filterColumns,
                FilterMode = filterMode,
                ImputeMode = imputeMode,
                SupressTypeErrors = supressTypeErrors
            };
        }

        internal TimeSeriesImputerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("TimeSeriesImputerEstimator");
            _host.CheckValue(options.TimeSeriesColumn, nameof(options.TimeSeriesColumn), "TimePoint column should not be null.");
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");
            if (options.FilterMode != FilterMode.NoFilter)
                _host.CheckNonEmpty(options.FilterColumns, nameof(options.FilterColumns), "Need at least 1 filter column if a FilterMode is specified");

            _options = options;
        }

        public TimeSeriesImputerTransformer Fit(IDataView input)
        {
            // If we are not suppressing type errors make sure columns to impute only contain supported types.
            if (!_options.SupressTypeErrors)
            {
                var columns = input.Schema.Where(x => !_options.GrainColumns.Contains(x.Name));
                if (_options.FilterMode == FilterMode.Exclude)
                    columns = columns.Where(x => !_options.FilterColumns.Contains(x.Name));
                else if (_options.FilterMode == FilterMode.Include)
                    columns = columns.Where(x => _options.FilterColumns.Contains(x.Name));

                foreach (var column in columns)
                {
                    if (!_currentSupportedTypes.Contains(column.Type.RawType))
                        throw new InvalidOperationException($"Type {column.Type.RawType.ToString()} for column {column.Name} not a supported type.");
                }
            }

            return new TimeSeriesImputerTransformer(_host, _options, input);
        }

        // Add one column called WasColumnImputed, otherwise everything stays the same.
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);
            columns[IsRowImputedColumnName] = new SchemaShape.Column(IsRowImputedColumnName, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false);
            return new SchemaShape(columns.Values);
        }
    }

    public sealed class TimeSeriesImputerTransformer : ITransformer, IDisposable
    {
        #region Class data members

        internal const string Summary = "Fills in missing row and values";
        internal const string UserName = "TimeSeriesImputer";
        internal const string ShortName = "tsi";
        internal const string LoadName = "TimeSeriesImputer";
        internal const string LoaderSignature = "TimeSeriesImputer";

        private readonly IHost _host;
        private readonly string _timeSeriesColumn;
        private readonly string[] _grainColumns;
        private readonly string[] _dataColumns;
        private readonly string[] _allColumnNames;
        private readonly bool _suppressTypeErrors;
        private readonly TimeSeriesImputerEstimator.ImputationStrategy _imputeMode;
        internal TransformerEstimatorSafeHandle TransformerHandle;

        #endregion

        // Normal constructor.
        internal TimeSeriesImputerTransformer(IHostEnvironment host, TimeSeriesImputerEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(TimeSeriesImputerTransformer));
            _timeSeriesColumn = options.TimeSeriesColumn;
            _grainColumns = options.GrainColumns;
            _imputeMode = options.ImputeMode;
            _suppressTypeErrors = options.SupressTypeErrors;

            IEnumerable<string> tempDataColumns;

            if (options.FilterMode == TimeSeriesImputerEstimator.FilterMode.Exclude)
                tempDataColumns = input.Schema.Where(x => !options.FilterColumns.Contains(x.Name)).Select(x => x.Name);
            else if (options.FilterMode == TimeSeriesImputerEstimator.FilterMode.Include)
                tempDataColumns = input.Schema.Where(x => options.FilterColumns.Contains(x.Name)).Select(x => x.Name);
            else
                tempDataColumns = input.Schema.Select(x => x.Name);

            // Time series and Grain columns should never be included in the data columns
            _dataColumns = tempDataColumns.Where(x => x != _timeSeriesColumn && !_grainColumns.Contains(x)).ToArray();

            // 1 is for the time series column. Make one array in the correct order of all the columns.
            // Order is Timeseries column, All grain columns, All data columns.
            _allColumnNames = new string[1 + _grainColumns.Length + _dataColumns.Length];
            _allColumnNames[0] = _timeSeriesColumn;
            Array.Copy(_grainColumns, 0, _allColumnNames, 1, _grainColumns.Length);
            Array.Copy(_dataColumns, 0, _allColumnNames, 1 + _grainColumns.Length, _dataColumns.Length);

            TransformerHandle = CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal TimeSeriesImputerTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(TimeSeriesImputerTransformer));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");

            // *** Binary format ***
            // name of time series column
            // length of grain column array
            // all column names in grain column array
            // length of filter column array
            // all column names in filter column array
            // byte value of filter mode
            // byte value of impute mode
            // length of C++ state array
            // C++ byte state array

            _timeSeriesColumn = ctx.Reader.ReadString();

            _grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < _grainColumns.Length; i++)
                _grainColumns[i] = ctx.Reader.ReadString();

            _dataColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < _dataColumns.Length; i++)
                _dataColumns[i] = ctx.Reader.ReadString();

            _imputeMode = (TimeSeriesImputerEstimator.ImputationStrategy)ctx.Reader.ReadByte();

            _allColumnNames = new string[1 + _grainColumns.Length + _dataColumns.Length];
            _allColumnNames[0] = _timeSeriesColumn;
            Array.Copy(_grainColumns, 0, _allColumnNames, 1, _grainColumns.Length);
            Array.Copy(_dataColumns, 0, _allColumnNames, 1 + _grainColumns.Length, _dataColumns.Length);

            var nativeState = ctx.Reader.ReadByteArray();
            TransformerHandle = CreateTransformerFromSavedData(nativeState);
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedData(byte[] nativeState)
        {
            fixed (byte* rawStatePointer = nativeState)
            {
                IntPtr dataSize = new IntPtr(nativeState.Count());
                var result = CreateTransformerFromSavedDataNative(rawStatePointer, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new TimeSeriesImputerTransformer(env, ctx).Transform(input));
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimator(IDataView input)
        {
            IntPtr estimator;
            IntPtr errorHandle;
            bool success;

            var allColumns = input.Schema.Where(x => _allColumnNames.Contains(x.Name)).Select(x => TypedColumn.CreateTypedColumn(x, _dataColumns)).ToDictionary(x => x.Column.Name);

            // Create TypeId[] for types of grain and data columns;
            var dataColumnTypes = new TypeId[_dataColumns.Length];
            var grainColumnTypes = new TypeId[_grainColumns.Length];

            foreach (var column in _grainColumns.Select((value, index) => new { index, value }))
                grainColumnTypes[column.index] = allColumns[column.value].GetTypeId();

            foreach (var column in _dataColumns.Select((value, index) => new { index, value }))
                dataColumnTypes[column.index] = allColumns[column.value].GetTypeId();

            fixed (bool* suppressErrors = &_suppressTypeErrors)
            fixed (TypeId* rawDataColumnTypes = dataColumnTypes)
            fixed (TypeId* rawGrainColumnTypes = grainColumnTypes)
            {
                success = CreateEstimatorNative(rawGrainColumnTypes, new IntPtr(grainColumnTypes.Length), rawDataColumnTypes, new IntPtr(dataColumnTypes.Length), _imputeMode, suppressErrors, out estimator, out errorHandle);
            }
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorNative))
            {
                TrainingState trainingState;
                FitResult fitResult;

                // Create buffer to hold binary data
                var memoryStream = new MemoryStream(4096);
                var binaryWriter = new BinaryWriter(memoryStream, Encoding.UTF8);

                // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                var cursor = input.GetRowCursorForAllColumns();
                // Initialize getters
                foreach (var column in allColumns.Values)
                    column.InitializeGetter(cursor);

                // Start the loop with the cursor in a valid state already.
                var valid = cursor.MoveNext();

                // Make sure its not an empty data frame
                Debug.Assert(valid);
                while (true)
                {
                    // Get the state of the native estimator.
                    success = GetStateNative(estimatorHandle, out trainingState, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we are no longer training then exit loop.
                    if (trainingState != TrainingState.Training)
                        break;

                    // Build byte array to send column data to native featurizer
                    BuildColumnByteArray(allColumns, ref binaryWriter);

                    // Fit the estimator
                    fixed (byte* bufferPointer = memoryStream.GetBuffer())
                    {
                        var binaryArchiveData = new NativeBinaryArchiveData() { Data = bufferPointer, DataSize = new IntPtr(memoryStream.Position) };
                        success = FitNative(estimatorHandle, binaryArchiveData, out fitResult, out errorHandle);
                    }

                    // Reset memory stream to 0
                    memoryStream.Position = 0;

                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we need to reset the data to the beginning.
                    if (fitResult == FitResult.ResetAndContinue)
                        ResetCursor(input, ref cursor, allColumns);

                    // If we are at the end of the data.
                    if (!cursor.MoveNext())
                    {
                        // If we get here fitResult should never be ResetAndContinue
                        Debug.Assert(fitResult != FitResult.ResetAndContinue);

                        OnDataCompletedNative(estimatorHandle, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        ResetCursor(input, ref cursor, allColumns);
                    }
                }

                // When done training complete the estimator.
                success = CompleteTrainingNative(estimatorHandle, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Create the native transformer from the estimator;
                success = CreateTransformerFromEstimatorNative(estimatorHandle, out IntPtr transformer, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Manually dispose of the IEnumerator since we don't have a using statement;
                cursor.Dispose();

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        private void ResetCursor(IDataView input, ref DataViewRowCursor cursor, Dictionary<string, TypedColumn> allColumns)
        {
            cursor.Dispose();
            cursor = input.GetRowCursorForAllColumns();

            // Initialize getters
            foreach (var column in allColumns.Values)
                column.InitializeGetter(cursor);

            // Move cursor to valid position
            var valid = cursor.MoveNext();
            Debug.Assert(valid);
        }

        private void BuildColumnByteArray(Dictionary<string, TypedColumn> allColumns, ref BinaryWriter binaryWriter)
        {
            foreach (var column in _allColumnNames)
            {
                allColumns[column].SerializeValue(ref binaryWriter);
            }
        }

        public bool IsRowToRowMapper => false;

        // Schema not changed
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumns(inputSchema.AsEnumerable());
            schemaBuilder.AddColumn(TimeSeriesImputerEstimator.IsRowImputedColumnName, BooleanDataViewType.Instance);

            return schemaBuilder.ToSchema();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TimeIm T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TimeSeriesImputerTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // name of time series column
            // length of grain column array
            // all column names in grain column array
            // length of data column array
            // all column names in data column array
            // byte value of impute mode
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_timeSeriesColumn);
            ctx.Writer.Write(_grainColumns.Length);
            foreach (var column in _grainColumns)
                ctx.Writer.Write(column);
            ctx.Writer.Write(_dataColumns.Length);
            foreach (var column in _dataColumns)
                ctx.Writer.Write(column);
            ctx.Writer.Write((byte)_imputeMode);
            var data = CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        private byte[] CreateTransformerSaveData()
        {
            var success = CreateTransformerSaveDataNative(TransformerHandle, out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
            {
                byte[] savedData = new byte[bufferSize.ToInt32()];
                Marshal.Copy(buffer, savedData, 0, savedData.Length);
                return savedData;
            }
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal TimeSeriesImputerDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new TimeSeriesImputerDataView(_host, input, _timeSeriesColumn, _grainColumns, _dataColumns, _allColumnNames, this);
        }

        internal TransformerEstimatorSafeHandle CloneTransformer() => CreateTransformerFromSavedData(CreateTransformerSaveData());

        public void Dispose()
        {
            if (!TransformerHandle.IsClosed)
                TransformerHandle.Close();
        }

        #region C++ function declarations
        // TODO: Update entry points

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool CreateEstimatorNative(TypeId* grainTypes, IntPtr grainTypesSize, TypeId* dataTypes, IntPtr dataTypesSize, TimeSeriesImputerEstimator.ImputationStrategy strategy, bool* suppressTypeErrors, out IntPtr estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool FitNative(TransformerEstimatorSafeHandle estimator, NativeBinaryArchiveData data, out FitResult fitResult, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CreateONNXSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateOnnxSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_OnDataCompleted"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_GetState"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);

        #endregion

        #region Typed Columns

        private abstract class TypedColumn
        {
            internal readonly DataViewSchema.Column Column;
            internal TypedColumn(DataViewSchema.Column column)
            {
                Column = column;
            }

            internal abstract void InitializeGetter(DataViewRowCursor cursor);
            internal abstract void SerializeValue(ref BinaryWriter binaryWriter);
            internal abstract TypeId GetTypeId();

            internal static TypedColumn CreateTypedColumn(DataViewSchema.Column column, string[] optionalColumns)
            {
                var type = column.Type.RawType.ToString();
                if (type == typeof(sbyte).ToString())
                    return new NumericTypedColumn<sbyte>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(short).ToString())
                    return new NumericTypedColumn<short>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(int).ToString())
                    return new NumericTypedColumn<int>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(long).ToString())
                    return new NumericTypedColumn<long>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(byte).ToString())
                    return new NumericTypedColumn<byte>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(ushort).ToString())
                    return new NumericTypedColumn<ushort>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(uint).ToString())
                    return new NumericTypedColumn<uint>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(ulong).ToString())
                    return new NumericTypedColumn<ulong>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(float).ToString())
                    return new NumericTypedColumn<float>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(double).ToString())
                    return new NumericTypedColumn<double>(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(ReadOnlyMemory<char>).ToString())
                    return new StringTypedColumn(column, optionalColumns.Contains(column.Name));
                else if (type == typeof(DateTime).ToString())
                    return new DateTimeTypedColumn(column, optionalColumns.Contains(column.Name));

                throw new InvalidOperationException($"Unsupported type {type}");
            }
        }

        private abstract class TypedColumn<T> : TypedColumn
        {
            private ValueGetter<T> _getter;
            private T _value;

            internal TypedColumn(DataViewSchema.Column column) :
                base(column)
            {
                _value = default;
            }

            internal override void InitializeGetter(DataViewRowCursor cursor)
            {
                _getter = cursor.GetGetter<T>(Column);
            }

            internal T GetValue()
            {
                _getter(ref _value);
                return _value;
            }

            internal override TypeId GetTypeId()
            {
                return typeof(T).GetNativeTypeIdFromType();
            }
        }

        private class NumericTypedColumn<T> : TypedColumn<T>
        {
            private readonly bool _isNullable;

            internal NumericTypedColumn(DataViewSchema.Column column, bool isNullable = false) :
                base(column)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(ref BinaryWriter binaryWriter)
            {
                dynamic value = GetValue();

                if (_isNullable && value.GetType() != typeof(float) && value.GetType() != typeof(double))
                    binaryWriter.Write(true);

                binaryWriter.Write(value);
            }
        }

        private class StringTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {
            private readonly bool _isNullable;

            internal StringTypedColumn(DataViewSchema.Column column, bool isNullable = false) :
                base(column)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(ref BinaryWriter binaryWriter)
            {
                var value = GetValue().ToString();
                var stringBytes = Encoding.UTF8.GetBytes(value);

                if (_isNullable)
                    binaryWriter.Write(true);

                binaryWriter.Write(stringBytes.Length);

                binaryWriter.Write(stringBytes);
            }
        }

        private class DateTimeTypedColumn : TypedColumn<DateTime>
        {
            private static readonly DateTime _unixEpoch = new DateTime(1970, 1, 1);
            private readonly bool _isNullable;

            internal DateTimeTypedColumn(DataViewSchema.Column column, bool isNullable = false) :
                base(column)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(ref BinaryWriter binaryWriter)
            {
                var dateTime = GetValue();

                var value = dateTime.Subtract(_unixEpoch).Ticks / TimeSpan.TicksPerSecond;

                if (_isNullable)
                    binaryWriter.Write(true);

                binaryWriter.Write(value);
            }
        }

        #endregion
    }

    internal static class TimeSeriesTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.TimeSeriesImputer",
            Desc = TimeSeriesImputerTransformer.Summary,
            UserName = TimeSeriesImputerTransformer.UserName,
            ShortName = TimeSeriesImputerTransformer.ShortName)]
        public static CommonOutputs.TransformOutput TimeSeriesImputer(IHostEnvironment env, TimeSeriesImputerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, TimeSeriesImputerTransformer.ShortName, input);
            var xf = new TimeSeriesImputerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
