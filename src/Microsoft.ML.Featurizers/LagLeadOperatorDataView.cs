using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Transactions;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

namespace Microsoft.ML.Featurizers
{
    internal sealed class LagLeadOperatorDataView : IDataTransform
    {
        private LagLeadOperatorTransformer _parent;
        private readonly IDataView _source;
        private readonly IHostEnvironment _host;
        private readonly DataViewSchema _schema;
        private readonly LagLeadOperatorEstimator.Options _options;

        internal LagLeadOperatorDataView(IHostEnvironment env, IDataView input, LagLeadOperatorEstimator.Options options, LagLeadOperatorTransformer parent)
        {
            _host = env;
            _source = input;

            _options = options;
            _parent = parent;

            // Use existing schema since it doesn't change.
            _schema = _parent.GetOutputSchema(input.Schema);
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            return new Cursor(_host, _source, _parent.CloneTransformers(), _options, _schema);
        }

        // Can't use parallel cursors so this defaults to calling non-parallel version
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
             new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

        // We aren't changing the row count, so just get the _source row count
        public long? GetRowCount() => _source.GetRowCount();

        public void Save(ModelSaveContext ctx)
        {
            _parent.Save(ctx);
        }

        private sealed class Cursor : DataViewRowCursor
        {
            private readonly IChannelProvider _ch;
            private IDataView _dataView;

            // Have 2 row cursors since one will be offset.
            private DataViewRowCursor _sourceCursor;
            private DataViewRowCursor _offsetCursor;
            private long _position;
            private bool _sourceIsGood;
            private bool _offsetIsGood;
            private readonly DataViewSchema _schema;
            private TypedColumn[] _columns;
            private readonly LagLeadOperatorEstimator.Options _options;
            private ValueGetter<ReadOnlyMemory<char>>[] _grainGetters;
            private List<string> _grainOrder;
            private bool _hasFlushed;

            // These are class variables so they are only allocated once.
            private GCHandle[] _grainHandles;
            private IntPtr[] _grainArray;
            private GCHandle _grainArrayHandle;

            public Cursor(IChannelProvider provider, IDataView input, TransformerEstimatorSafeHandle[] transformers, LagLeadOperatorEstimator.Options options, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                _dataView = input;
                _position = -1;
                _schema = schema;
                _options = options;
                _grainGetters = new ValueGetter<ReadOnlyMemory<char>>[_options.GrainColumns.Length];

                _grainHandles = new GCHandle[_options.GrainColumns.Length];
                _grainArray = new IntPtr[_options.GrainColumns.Length];
                _grainOrder = new List<string>();

                _sourceIsGood = true;
                _offsetIsGood = true;
                _hasFlushed = false;

                _sourceCursor = _dataView.GetRowCursorForAllColumns();
                _offsetCursor = _dataView.GetRowCursorForAllColumns();

                InitializeGrainGetters(_options.GrainColumns, ref _grainGetters);

                _columns = new TypedColumn[transformers.Length];
                for(int i = 0; i < transformers.Length; i++)
                {
                    _columns[i] = TypedColumn.CreateTypedColumn(options.Columns[i].Name, options.Columns[i].Source, input.Schema[options.Columns[i].Source].Type.RawType.ToString(), transformers[i], this);
                }
            }

            public sealed override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                       (ref DataViewRowId val) =>
                       {
                           _ch.Check(_sourceIsGood, RowCursorUtils.FetchValueStateError);
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
                foreach(var column in _columns)
                {
                    column.Dispose();
                }
                _sourceCursor.Dispose();
                _offsetCursor.Dispose();
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

                if (_columns.Any(x => x.Name == column.Name && x.Type == column.Type.RawType.ToString()))
                    return (ValueGetter<TValue>)_columns.Where(x => x.Name == column.Name).First().GetGetter();

                return _sourceCursor.GetGetter<TValue>(column);
            }

            public override bool MoveNext()
            {
                bool colMoveNext;

                // Move the non-offset cursor forward by one.
                _sourceIsGood = _sourceCursor.MoveNext();

                bool exitLoop = false;
                while (!exitLoop && _offsetIsGood)
                {
                    // Loop using the offset cursor until the native featurizer returns 1 result
                    _offsetIsGood = _offsetCursor.MoveNext();
                    if (!_offsetIsGood)
                        break;
                    try
                    {
                        // When the native featurizer returns the data from flush its not in the correct order.
                        // This is needed to put it back in the correct order.
                        _grainOrder.Add(GrainsToString(_grainGetters));
                        CreateGrainStringArrays(_grainGetters, ref _grainHandles, ref _grainArrayHandle, ref _grainArray);

                        // All columns have the same parameters, so if one is good then all are good and vice versa.
                        colMoveNext = _columns[0].MoveNext();

                        if (colMoveNext)
                            exitLoop = true;

                    }
                    finally
                    {
                        FreeGrainStringArrays(ref _grainHandles, ref _grainArrayHandle);
                    }
                }

                if (!_offsetIsGood)
                {
                    // If we haven't flushed the data from the native featurizer yet, do it now.
                    // Only want to do this once.
                    if(!_hasFlushed)
                    {
                        foreach(var column in _columns)
                        {
                            column.Flush();
                        }
                        _hasFlushed = true;
                    }

                    if(_sourceIsGood)
                    {
                        foreach(var column in _columns)
                        {
                            column.MoveNext(_grainOrder[0]);
                        }
                    }
                }

                // Remove the first item every time we return from the move next call if the source is still good.
                if(_sourceIsGood)
                    _grainOrder.RemoveAt(0);

                _position++;
                return _sourceIsGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _sourceCursor.Batch;

            private void InitializeGrainGetters(string[] grainColumns, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
            {
                // Create getters for the source grain columns.

                for (int i = 0; i < _grainGetters.Length; i++)
                {
                    // Inititialize the getter and move it to a valid position.
                    grainGetters[i] = _offsetCursor.GetGetter<ReadOnlyMemory<char>>(_dataView.Schema[grainColumns[i]]);
                }
            }

            #region Typed Columns

            // Safe handle that frees the memory for the transformed data.
            // Is called automatically after each call to transform.
            internal unsafe delegate bool DestroyLagLeadTransformedDataNative(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, out IntPtr errorHandle);
            internal unsafe class LagLeadTransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
            {
                private DestroyLagLeadTransformedDataNative _destroySaveDataHandler;
                private IntPtr _grainsPointer;
                private IntPtr _grainsPointerSize;
                private IntPtr _outputCols;
                private IntPtr _outputRows;
                private double** _output;
                private IntPtr _outputItems;
                public unsafe LagLeadTransformedDataSafeHandle(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, DestroyLagLeadTransformedDataNative destroyNativeData) : base(true)
                {
                    SetHandle(grainsPointer);
                    _destroySaveDataHandler = destroyNativeData;
                    _grainsPointer = grainsPointer;
                    _grainsPointerSize = grainsPointerSize;
                    _outputCols = outputCols;
                    _outputRows = outputRows;
                    _output = output;
                    _outputItems = outputItems;
                }

                protected override bool ReleaseHandle()
                {
                    // Not sure what to do with error stuff here.  There shoudln't ever be one though.
                    return _destroySaveDataHandler(_grainsPointer, _grainsPointerSize, _outputCols, _outputRows, _output, _outputItems, out IntPtr errorHandle);
                }
            }

            private abstract class TypedColumn : IDisposable
            {
                private protected readonly TransformerEstimatorSafeHandle TransformerHandle;
                private protected readonly Cursor Parent;
                private protected readonly string Source;
                internal readonly string Type;
                internal readonly string Name;

                internal TypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent, string type)
                {
                    Source = source;
                    Name = name;
                    TransformerHandle = transformer;
                    Parent = parent;
                    Type = type;
                }

                internal abstract Delegate GetGetter();
                internal abstract bool MoveNext(string grainString = "");
                public void Dispose()
                {
                    if (!TransformerHandle.IsClosed)
                        TransformerHandle.Dispose();
                }

                internal static TypedColumn CreateTypedColumn(string name, string source, string type, TransformerEstimatorSafeHandle transformer, Cursor parent)
                {
                    if (type == typeof(double).ToString())
                    {
                        return new DoubleTypedColumn(name, source, transformer, parent);
                    }

                    throw new InvalidOperationException($"Column {source} has an unsupported type {type}.");
                }

                internal abstract void Flush();
            }

            private abstract class TypedColumn<TInput, TOutput> : TypedColumn
            {
                private protected ValueGetter<TOutput> Getter;
                private protected ValueGetter<TInput> SourceGetter;
                private protected TOutput Result;
                private protected TInput InputValue;

                // When we call flush we get all the values back sorted correctly by grain, but if the original
                // wasn't sorted by grain this order is incorrect. We need to store them by grain so that we can
                // return them in the right order.
                private protected Dictionary<string, List<TOutput>> FlushedValuesCache;

                internal TypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent, string type) :
                    base(name, source, transformer, parent, type)
                {
                    SourceGetter = parent._offsetCursor.GetGetter<TInput>(parent._dataView.Schema[source]);
                    Result = default;
                    InputValue = default;
                    FlushedValuesCache = new Dictionary<string, List<TOutput>>();
                }

                internal override Delegate GetGetter()
                {
                    return Getter;
                }

                internal override bool MoveNext(string grainString = "")
                {
                    if (Parent._offsetIsGood)
                    {
                        SourceGetter(ref InputValue);
                        if (!Transform())
                            return false;
                        return true;
                    }
                    else
                    {
                        // In this case flush has already been called. We just need to match the correct
                        // grain from the internal FlushedValuesCache.
                        Debug.Assert(grainString != "", "Grain string should never be empty at this point.");
                        Result = FlushedValuesCache[grainString][0];

                        // Now that Result is set to the correct value, remove index 0
                        FlushedValuesCache[grainString].RemoveAt(0);
                    }

                    // Once Parent._offsetIsGood is false we no longer care about the return value from this.
                    // Because of that, defaulting to return true;

                    return true;
                }

                internal abstract bool Transform();
            }

            private class DoubleTypedColumn : TypedColumn<double, VBuffer<double>>
            {
                [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, double value, out IntPtr grainsPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Flush", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FlushDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr grainsPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool DestroyTransformedDataNative(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, out IntPtr errorHandle);

                internal DoubleTypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent) :
                    base(name, source, transformer, parent, typeof(VBuffer<double>).ToString())
                {
                    InitializeGetter();
                }

                private void InitializeGetter()
                {
                    double input = default;
                    Getter = (ref VBuffer<double> dst) =>
                    {
                        // If the offset is no longer good, then this is the last row.
                        if (Parent._offsetIsGood)
                            SourceGetter(ref input);
                        unsafe
                        {
                            dst = Result;
                        }
                    };
                }

                internal unsafe override void Flush()
                {
                    var success = FlushDataNative(TransformerHandle, out IntPtr grainsPointerPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    var grainsPointerSizesP = (long*)grainsPointerSize.ToPointer();
                    var allGrainsPointer = (byte***)grainsPointerPointer.ToPointer();

                    for (int items = 0; items < outputItems.ToInt32(); items++)
                    {
                        var grainsArraySize = *grainsPointerSizesP++;
                        var grainArray = new string[grainsArraySize];
                        for(int grainItems = 0; grainItems < grainsArraySize; grainItems++)
                        {
                            var grainArrayPointer = *allGrainsPointer;
                            grainArray[grainItems] = PointerToString(new IntPtr(*grainArrayPointer));
                        }

                        VBuffer<double> res = default;

                        if (IntPtr.Size == 4)
                        {
                            res = ParseTransformResultx32((int*)outputCols.ToPointer(), (int*)outputRows.ToPointer(), output++, outputItems);
                            outputCols += 4;
                            outputRows += 4;
                        }
                        else
                        {
                            res = ParseTransformResultx64((long*)outputCols.ToPointer(), (long*)outputRows.ToPointer(), output++, outputItems);
                            outputCols += 8;
                            outputRows += 8;
                        }

                        var grainString = string.Join("", grainArray);

                        if (!FlushedValuesCache.ContainsKey(grainString))
                            FlushedValuesCache[grainString] = new List<VBuffer<double>>();

                        FlushedValuesCache[grainString].Add(res);
                    }
                }

                internal override unsafe bool Transform()
                {
                    var success = TransformDataNative(TransformerHandle, Parent._grainArrayHandle.AddrOfPinnedObject(), new IntPtr(Parent._grainArray.Length), InputValue, out IntPtr grainsPointerPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    if (outputItems == IntPtr.Zero)
                        return false;

                   using var handler = new LagLeadTransformedDataSafeHandle(grainsPointerPointer, grainsPointerSize, outputCols, outputRows, output, outputItems, DestroyTransformedDataNative);

                    // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                    // These will be used when we flush all the output data, but the signature is the same for transform.
                    // Since we know there is only 1 output row, we can short circuit some of the loops.

                    // x32
                    if (IntPtr.Size == 4)
                        Result = ParseTransformResultx32((int*)outputCols.ToPointer(), (int*)outputRows.ToPointer(), output, outputItems);
                    else
                        Result = ParseTransformResultx64((long*)outputCols.ToPointer(), (long*)outputRows.ToPointer(), output, outputItems);

                    return true;
                }

                // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                // These will be used when we flush all the output data, but the signature is the same for transform.
                // Since we know there is only 1 output row, we can short circuit some of the loops.
                private unsafe VBuffer<double> ParseTransformResultx64(long* outputCols, long* outputRows, double** output, IntPtr outputItems)
                {
                    var outputArray = new double[(*outputRows) * (*outputCols)];
                    var drefOuput = *output;

                    for (int i = 0; i < outputArray.Length; i++)
                    {
                        outputArray[i] = *drefOuput++;
                    }

                    return new VBuffer<double>(outputArray.Length, outputArray);
                }

                // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                // These will be used when we flush all the output data, but the signature is the same for transform.
                // Since we know there is only 1 output row, we can short circuit some of the loops.
                private unsafe VBuffer<double> ParseTransformResultx32(int* outputCols, int* outputRows, double** output, IntPtr outputItems)
                {
                    var outputArray = new double[(*outputRows) * (*outputCols)];
                    var drefOuput = *output;

                    for (int i = 0; i < outputArray.Length; i++)
                    {
                        outputArray[i] = *drefOuput++;
                    }

                    return new VBuffer<double>(outputArray.Length, outputArray);
                }
            }

            #endregion Typed Columns
        }
    }
}
