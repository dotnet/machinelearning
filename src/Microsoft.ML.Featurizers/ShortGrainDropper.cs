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

[assembly: LoadableClass(typeof(ShortDropTransformer), null, typeof(SignatureLoadModel),
    ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(ShortDropTransformer), null, typeof(SignatureLoadDataTransform),
   ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ShortDropTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ShortGrainDropperExtensionClass
    {

        /// <summary>
        /// Creates a <see cref="ShortGrainDropperEstimator"/> that drops rows per grain when the number of rows for that grain is less then
        /// the <paramref name="minRows"/>
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="grainColumns">List of the grain columns. The combination of these form the "unique key" for each row.</param>
        /// <param name="minRows">The minimum number of occurances required for each "unique key". If less than this, the rows will be dropped.</param>
        /// <returns><see cref="ShortGrainDropperEstimator"/></returns>
        public static ShortGrainDropperEstimator DropShortGrains(this TransformsCatalog catalog, string[] grainColumns, UInt32 minRows)
        {
            var options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                MinRows = minRows
            };

            return new ShortGrainDropperEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// ShortGrainDroppper Featurizer determines which grains have the minimum number of rows specified, and then drops all grains
    /// that don't have that minimum number.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Takes in all columns |
    /// | Output column data type | Same as the original input type. |
    ///
    /// The <xref:Microsoft.ML.Transforms.ShortGrainDropperEstimator> is not a trivial estimator and needs training.
    /// Consider the training data:
    ///
    ///[ ["one"], ["two"], ["two"], ["three"], ["three"], ["three"] ]
    ///
    ///    and a ShortGrainDropper configured with minRows set to 2. Grains["two"] and["three"] appear
    ///  enough times in the training data to remain, while any other grain should be dropped:
    ///
    ///    [ "one" ] -> true                         # drop
    ///    [ "two" ] -> false                        # dont' drop
    ///    [ "three" ] -> false                      # don't drop
    ///    [ "never seen during training" ] -> true  # drop
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ShortGrainDropperExtensionClass.DropShortGrains(TransformsCatalog, string[], UInt32)"/>
    public sealed class ShortGrainDropperEstimator : IEstimator<ShortDropTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns", Name = "GrainColumns", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Minimum number of values required",
                Name = "MinRows", ShortName = "minr", SortOrder = 1)]
            public UInt32 MinRows;
        }

        #endregion

        internal ShortGrainDropperEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("ShortDropEstimator");
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");
            Contracts.Check(options.MinRows > 0, "Min points must be greater than 0");

            _options = options;
        }

        public ShortDropTransformer Fit(IDataView input)
        {
            if (!AllGrainColumnsAreStrings(input.Schema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            return new ShortDropTransformer(_host, _options, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // We dont change the schema, we just drop rows. Still validate grain columns are correct type
            if (!AllGrainColumnsAreStrings(inputSchema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            return inputSchema;
        }
    }

    public sealed class ShortDropTransformer : ITransformer, IDisposable
    {
        #region Class data members

        internal const string Summary = "Drops rows if there aren't enough values per grain.";
        internal const string UserName = "ShortDrop";
        internal const string ShortName = "sgd";
        internal const string LoadName = "ShortDrop";
        internal const string LoaderSignature = "ShortDrop";

        private readonly IHost _host;
        private readonly ShortGrainDropperEstimator.Options _options;
        internal TransformerEstimatorSafeHandle TransformerHandle;

        #endregion

        // Normal constructor.
        internal ShortDropTransformer(IHostEnvironment host, ShortGrainDropperEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ShortDropTransformer));
            _options = options;

            TransformerHandle = CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal ShortDropTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(ShortDropTransformer));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // minRows
            // length of C++ state array
            // C++ byte state array

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < grainColumns.Length; i++)
                grainColumns[i] = ctx.Reader.ReadString();

            var minRows = ctx.Reader.ReadUInt32();

            _options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                MinRows = minRows
            };

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
            return (IDataTransform)(new ShortDropTransformer(env, ctx).Transform(input));
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimator(IDataView input)
        {
            IntPtr estimator;
            IntPtr errorHandle;
            bool success;

            success = CreateEstimatorNative(_options.MinRows, out estimator, out errorHandle);
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorNative))
            {
                TrainingState trainingState;
                FitResult fitResult;

                // Declare these outside the loop so the size is only set once;
                GCHandle[] grainHandles = new GCHandle[_options.GrainColumns.Length];
                IntPtr[] grainArray = new IntPtr[_options.GrainColumns.Length];
                GCHandle arrayHandle = default;

                // These are initialized in InitializeGrainGetters
                ValueGetter<ReadOnlyMemory<char>>[] grainGetters = new ValueGetter<ReadOnlyMemory<char>>[_options.GrainColumns.Length];
                DataViewRowCursor cursor = null;

                // Initialize GrainGetters and put cursor in valid state.
                var valid = InitializeGrainGetters(input, ref cursor, ref grainGetters);
                Debug.Assert(valid);
                // Start the loop with the cursor in a valid state already.
                while (true)
                {
                    // Get the state of the native estimator.
                    success = GetStateNative(estimatorHandle, out trainingState, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we are no longer training then exit loop.
                    if (trainingState != TrainingState.Training)
                        break;

                    // Build the grain string array
                    try
                    {
                        CreateGrainStringArrays(grainGetters, ref grainHandles, ref arrayHandle, ref grainArray);
                        // Fit the estimator
                        success = FitNative(estimatorHandle, arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), out fitResult, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }
                    finally
                    {
                        FreeGrainStringArrays(ref grainHandles, ref arrayHandle);
                    }

                    // If we need to reset the data to the beginning.
                    if (fitResult == FitResult.ResetAndContinue)
                    {
                        valid = InitializeGrainGetters(input, ref cursor, ref grainGetters);
                        Debug.Assert(valid);
                    }

                    // If we are at the end of the data.
                    if (!cursor.MoveNext())
                    {
                        // fit result should never be reset and continue here
                        Debug.Assert(fitResult != FitResult.ResetAndContinue);

                        OnDataCompletedNative(estimatorHandle, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        valid = InitializeGrainGetters(input, ref cursor, ref grainGetters);
                        Debug.Assert(valid);
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

                // Manually dispose of the IEnumerator since we dont have a using statement;
                cursor.Dispose();

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        private bool InitializeGrainGetters(IDataView input, ref DataViewRowCursor cursor, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
        {
            // Create getters for the grain columns. Cant use using for the cursor because it may need to be reset.
            // Manually dispose of the cursor if its not null
            if (cursor != null)
                cursor.Dispose();

            cursor = input.GetRowCursor(input.Schema.Where(x => _options.GrainColumns.Contains(x.Name)));

            for (int i = 0; i < _options.GrainColumns.Length; i++)
            {
                // Inititialize the enumerator and move it to a valid position.
                grainGetters[i] = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_options.GrainColumns[i]]);
            }

            return cursor.MoveNext();
        }

        public bool IsRowToRowMapper => false;

        // Schema not changed
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            return inputSchema;
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SGDROP T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ShortDropTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // minRows
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var column in _options.GrainColumns)
                ctx.Writer.Write(column);

            ctx.Writer.Write(_options.MinRows);

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

        internal ShortGrainDropperDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new ShortGrainDropperDataView(_host, input, _options.GrainColumns, this);
        }

        internal TransformerEstimatorSafeHandle CloneTransformer() => CreateTransformerFromSavedData(CreateTransformerSaveData());

        public void Dispose()
        {
            if (!TransformerHandle.IsClosed)
                TransformerHandle.Close();
        }

        #region IDataView

        internal sealed class ShortGrainDropperDataView : IDataTransform
        {
            private ShortDropTransformer _parent;

            #region Native Exports

            [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, out bool skipRow, out IntPtr errorHandle);

            #endregion

            private readonly IHostEnvironment _host;
            private readonly IDataView _source;
            private readonly string[] _grainColumns;
            private readonly DataViewSchema _schema;

            internal ShortGrainDropperDataView(IHostEnvironment env, IDataView input, string[] grainColumns, ShortDropTransformer parent)
            {
                _host = env;
                _source = input;

                _grainColumns = grainColumns;
                _parent = parent;

                // Use existing schema since it doesn't change.
                _schema = _source.Schema;
            }

            // Conceptually this transformer can shuffle, but if time series data gets shuffled then things will break.
            // To prevent that, we dont allow shuffling with this transformer either.
            public bool CanShuffle => false;

            public DataViewSchema Schema => _schema;

            public IDataView Source => _source;

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.AssertValueOrNull(rand);

                var input = _source.GetRowCursorForAllColumns();
                return new Cursor(_host, input, _parent.CloneTransformer(), _grainColumns, _schema);
            }

            // Can't use parallel cursors so this defaults to calling non-parallel version
            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
                 new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

            // Since we may delete rows we don't know the row count
            public long? GetRowCount() => null;

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            private sealed class Cursor : DataViewRowCursor
            {
                private readonly IChannelProvider _ch;
                private DataViewRowCursor _input;
                private long _position;
                private bool _isGood;
                private readonly DataViewSchema _schema;
                private readonly TransformerEstimatorSafeHandle _transformer;
                private ValueGetter<ReadOnlyMemory<char>>[] _grainGetters;

                // These are class variables so they are only allocated once.
                private GCHandle[] _grainHandles;
                private IntPtr[] _grainArray;
                private GCHandle _arrayHandle;

                public Cursor(IChannelProvider provider, DataViewRowCursor input, TransformerEstimatorSafeHandle transformer, string[] grainColumns, DataViewSchema schema)
                {
                    _ch = provider;
                    _ch.CheckValue(input, nameof(input));

                    _input = input;
                    _position = -1;
                    _schema = schema;
                    _transformer = transformer;
                    _grainGetters = new ValueGetter<ReadOnlyMemory<char>>[grainColumns.Length];

                    _grainHandles = new GCHandle[grainColumns.Length];
                    _grainArray = new IntPtr[grainColumns.Length];

                    InitializeGrainGetters(grainColumns, ref _grainGetters);
                }

                public sealed override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                           (ref DataViewRowId val) =>
                           {
                               _ch.Check(_isGood, RowCursorUtils.FetchValueStateError);
                               val = new DataViewRowId((ulong)Position, 0);
                           };
                }

                public sealed override DataViewSchema Schema => _schema;

                /// <summary>
                /// Since rows will be dropped
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column) => true;

                protected override void Dispose(bool disposing)
                {
                    if (!_transformer.IsClosed)
                        _transformer.Close();
                }

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type.
                /// Since all we are doing is dropping rows, we can just use the source getter.
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    _ch.Check(IsColumnActive(column));

                    return _input.GetGetter<TValue>(column);
                }

                public override bool MoveNext()
                {
                    _position++;
                    while (true)
                    {
                        // If there are no more source rows exit loop and return false.
                        _isGood = _input.MoveNext();
                        if (!_isGood)
                            break;

                        try
                        {
                            CreateGrainStringArrays(_grainGetters, ref _grainHandles, ref _arrayHandle, ref _grainArray);
                            var success = TransformDataNative(_transformer, _arrayHandle.AddrOfPinnedObject(), new IntPtr(_grainArray.Length), out bool skipRow, out IntPtr errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                            // If native featurizer returns true it means to skip this row, so stay in loop.
                            // If it returns false then use it, so exit loop.
                            if (!skipRow)
                                break;
                        }
                        finally
                        {
                            FreeGrainStringArrays(ref _grainHandles, ref _arrayHandle);
                        }
                    }

                    return _isGood;
                }

                public sealed override long Position => _position;

                public sealed override long Batch => _input.Batch;

                private void InitializeGrainGetters(string[] grainColumns, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
                {
                    // Create getters for the source grain columns.

                    for (int i = 0; i < _grainGetters.Length; i++)
                    {
                        // Inititialize the getter and move it to a valid position.
                        grainGetters[i] = _input.GetGetter<ReadOnlyMemory<char>>(_input.Schema[grainColumns[i]]);
                    }
                }
            }
        }

        #endregion IDataView

        #region C++ function declarations

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateEstimatorNative(UInt32 minRows, out IntPtr estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, out FitResult fitResult, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_OnDataCompleted"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_GetState"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);

        #endregion
    }

    internal static class ShortDropTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ShortDrop",
            Desc = ShortDropTransformer.Summary,
            UserName = ShortDropTransformer.UserName,
            ShortName = ShortDropTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ShortDrop(IHostEnvironment env, ShortGrainDropperEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ShortDropTransformer.ShortName, input);
            var xf = new ShortGrainDropperEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
