// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(UngroupTransform.Summary, typeof(UngroupTransform), typeof(UngroupTransform.Options), typeof(SignatureDataTransform),
    UngroupTransform.UserName, UngroupTransform.ShortName)]

[assembly: LoadableClass(UngroupTransform.Summary, typeof(UngroupTransform), null, typeof(SignatureLoadDataTransform),
    UngroupTransform.UserName, UngroupTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{

    // This can be thought of as an inverse of GroupTransform. For all specified vector columns
    // ("pivot" columns), performs the "ungroup" (or "unroll") operation as outlined below.
    //
    // If the only pivot column is called P, and has size K, then for every row of the input we will produce
    // K rows, that are identical in all columns except P. The column P will become a scalar column, and this
    // column will hold all the original values of input's P, one value per row, in order. The order of columns
    // will remain the same.
    //
    // Variable-length pivot columns are supported (including zero, which will eliminate the row from the result).
    //
    // Multiple pivot columns are also supported:
    // * A number of output rows is controlled by the 'mode' parameter.
    //     - outer: it is equal to the maximum length of pivot columns,
    //     - inner: it is equal to the minimum length of pivot columns,
    //     - first: it is equal to the length of the first pivot column.
    // * If a particular pivot column has size that is different than the number of output rows, the extra slots will
    // be ignored, and the missing slots will be 'padded' with default values.
    //
    // All metadata is preserved for the retained columns. For 'unrolled' columns, all known metadata
    // except slot names is preserved.
    /// <include file='doc.xml' path='doc/members/member[@name="Ungroup"]/*' />
    internal sealed class UngroupTransform : TransformBase
    {
        public const string Summary = "Un-groups vector columns into sequences of rows, inverse of Group transform";
        public const string LoaderSignature = "UngroupTransform";
        public const string ShortName = "Ungroup";
        public const string UserName = "Un-group Transform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "UNGRP XF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(UngroupTransform).Assembly.FullName);
        }

        /// <summary>
        /// Controls the number of output rows produced by the <see cref="UngroupTransform"/> transform
        /// </summary>
        public enum UngroupMode
        {
            /// <summary>
            /// The number of output rows is equal to the minimum length of pivot columns
            /// </summary>
            Inner,

            /// <summary>
            /// The number of output rows is equal to the maximum length of pivot columns
            /// </summary>
            Outer,

            /// <summary>
            /// The number of output rows is equal to the length of the first pivot column.
            /// </summary>
            First
        }

        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to unroll, or 'pivot'", Name = "Column", ShortName = "col")]
            public string[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies how to unroll multiple pivot columns of different size.")]
            public UngroupMode Mode = UngroupMode.Inner;
        }

        private readonly UngroupBinding _ungroupBinding;

        /// <summary>
        /// Initializes a new instance of <see cref="UngroupTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="mode">Specifies how to unroll multiple pivot columns of different size.</param>
        /// <param name="columns">Columns to unroll, or 'pivot'</param>
        public UngroupTransform(IHostEnvironment env, IDataView input, UngroupMode mode, params string[] columns)
            : this(env, new Options() { Columns = columns, Mode = mode }, input)
        {
        }

        public UngroupTransform(IHostEnvironment env, Options options, IDataView input)
            : base(env, LoaderSignature, input)
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "There must be at least one pivot column");
            Host.CheckUserArg(options.Columns.Distinct().Count() == options.Columns.Length, nameof(options.Columns),
                "Duplicate pivot columns are not allowed");

            _ungroupBinding = new UngroupBinding(Host, Source.Schema, options.Mode, options.Columns);
        }

        public static UngroupTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new UngroupTransform(h, ctx, input));
        }

        private UngroupTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // (binding)
            _ungroupBinding = UngroupBinding.Create(ctx, host, input.Schema);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // (binding)
            _ungroupBinding.Save(ctx);
        }

        public override long? GetRowCount()
        {
            // Row count is known if the input's row count is known, and pivot column sizes are fixed.
            var commonSize = _ungroupBinding.GetCommonPivotColumnSize();
            if (commonSize > 0)
            {
                long? srcRowCount = Source.GetRowCount();
                if (srcRowCount.HasValue && srcRowCount.Value <= (long.MaxValue / commonSize))
                    return srcRowCount.Value * commonSize;
            }
            return null;
        }

        public override DataViewSchema OutputSchema => _ungroupBinding.OutputSchema;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            // This transform doesn't benefit from parallel cursors, but it can support them.
            return null;
        }

        // Technically, we could shuffle the ungrouped data if the source can shuffle. However, we want to maintain
        // contiguous groups. There's also a question whether we should shuffle inside groups or just shuffle groups
        // themselves. With these issues, and no anticipated use for shuffled version, it's safer to not shuffle at all.
        public override bool CanShuffle
        {
            get { return false; }
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var activeInput = _ungroupBinding.GetActiveInput(predicate);

            var inputCols = Source.Schema.Where(x => activeInput[x.Index]);
            var inputCursor = Source.GetRowCursor(inputCols, null);
            return new Cursor(Host, inputCursor, _ungroupBinding, predicate);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded,
            int n, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var activeInput = _ungroupBinding.GetActiveInput(predicate);

            var inputCols = Source.Schema.Where(x => activeInput[x.Index]);
            var inputCursors = Source.GetRowCursorSet(inputCols, n, null);
            return Utils.BuildArray<DataViewRowCursor>(inputCursors.Length,
                x => new Cursor(Host, inputCursors[x], _ungroupBinding, predicate));
        }

        private sealed class UngroupBinding
        {
            private static bool ShouldPreserveMetadata(string kind)
            {
                switch (kind)
                {
                    case AnnotationUtils.Kinds.IsNormalized:
                    case AnnotationUtils.Kinds.KeyValues:
                    case AnnotationUtils.Kinds.ScoreColumnSetId:
                    case AnnotationUtils.Kinds.ScoreColumnKind:
                    case AnnotationUtils.Kinds.ScoreValueKind:
                    case AnnotationUtils.Kinds.IsUserVisible:
                        return true;
                    default:
                        return false;
                }
            }

            public readonly struct PivotColumnOptions
            {
                public readonly string Name;
                public readonly int Index;
                public readonly int Size;
                public readonly PrimitiveDataViewType ItemType;

                public PivotColumnOptions(string name, int index, int size, PrimitiveDataViewType itemType)
                {
                    Contracts.AssertNonEmpty(name);
                    Contracts.Assert(index >= 0);
                    Contracts.Assert(size >= 0);
                    Contracts.AssertValue(itemType);
                    Name = name;
                    Index = index;
                    Size = size;
                    ItemType = itemType;
                }
            }

            private readonly DataViewSchema _inputSchema;
            private readonly IExceptionContext _ectx;

            /// <summary>
            /// Information of columns to be ungrouped in <see cref="_inputSchema"/>.
            /// </summary>
            private readonly PivotColumnOptions[] _infos;

            /// <summary>
            /// <see cref="_pivotIndex"/>[i] is -1 means that the i-th column in both of <see cref="_inputSchema"/> and <see cref="OutputSchema"/>
            /// are not produced by ungrouping; we just copy the i-th input column to the i-th output column.
            /// If <see cref="_pivotIndex"/>[i] is not -1, the i-th output column should be produced by ungrouping the i-th input column.
            /// </summary>
            private readonly int[] _pivotIndex;

            /// <summary>
            /// Columns contained in <see cref="IDataView"/> passed in <see cref="UngroupTransform"/>.
            /// Note that input data's schema is stored as <see cref="_inputSchema"/>.
            /// </summary>
            public int InputColumnCount => _inputSchema.Count;
            /// <summary>
            /// This attribute specifies how <see cref="UngroupTransform"/> expanding input columns stored in <see cref="_infos"/>.
            /// </summary>
            public readonly UngroupMode Mode;
            /// <summary>
            /// Output data's <see cref="DataViewSchema"/> produced by this <see cref="UngroupTransform"/>
            /// when input data's schema is <see cref="_inputSchema"/>.
            /// </summary>
            public DataViewSchema OutputSchema { get; }

            public UngroupBinding(IExceptionContext ectx, DataViewSchema inputSchema, UngroupMode mode, string[] pivotColumns)
            {
                Contracts.AssertValueOrNull(ectx);
                _ectx = ectx;
                _ectx.AssertValue(inputSchema);
                _ectx.AssertNonEmpty(pivotColumns);

                _inputSchema = inputSchema; // This also makes InputColumnCount valid.
                Mode = mode;

                Bind(_ectx, inputSchema, pivotColumns, out _infos);

                _pivotIndex = Utils.CreateArray(InputColumnCount, -1);
                for (int i = 0; i < _infos.Length; i++)
                {
                    var info = _infos[i];
                    _ectx.Assert(_pivotIndex[info.Index] == -1);
                    _pivotIndex[info.Index] = i;
                }

                var schemaBuilder = new DataViewSchema.Builder();
                // Iterate through input columns. Input columns which are not pivot columns will be copied to output schema with the same column index unchanged.
                // Input columns which are pivot columns would also be copied but with different data types and different metadata.
                for (int i = 0; i < InputColumnCount; ++i)
                {
                    if (_pivotIndex[i] < 0)
                    {
                        // i-th input column is not a pivot column. Let's do a naive copy.
                        schemaBuilder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
                    }
                    else
                    {
                        // i-th input column is a pivot column. Let's calculate proper type and metadata for it.
                        var metadataBuilder = new DataViewSchema.Annotations.Builder();
                        metadataBuilder.Add(inputSchema[i].Annotations, metadataName => ShouldPreserveMetadata(metadataName));
                        // To explain the output type of pivot columns, let's consider a row
                        //   Age UserID
                        //   18  {"Amy", "Willy"}
                        // where "Age" and "UserID" are column names and 18/{"Amy", "Willy"} is "Age"/"UserID" column in this example row.
                        // If the only pivot column is "UserID", the ungroup may produce
                        //   Age UserID
                        //   18  "Amy"
                        //   18  "Willy"
                        // One can see that "UserID" column (in output data) has a type identical to the element's type of the "UserID" column in input data.
                        schemaBuilder.AddColumn(inputSchema[i].Name, inputSchema[i].Type.GetItemType(), metadataBuilder.ToAnnotations());
                    }
                }
                OutputSchema = schemaBuilder.ToSchema();
            }

            private static void Bind(IExceptionContext ectx, DataViewSchema inputSchema,
                string[] pivotColumns, out PivotColumnOptions[] infos)
            {
                Contracts.AssertValueOrNull(ectx);
                ectx.AssertValue(inputSchema);
                ectx.AssertNonEmpty(pivotColumns);

                infos = new PivotColumnOptions[pivotColumns.Length];
                for (int i = 0; i < pivotColumns.Length; i++)
                {
                    var name = pivotColumns[i];
                    // REVIEW: replace Check with CheckUser, once existing CheckUser is renamed to CheckUserArg or something.
                    ectx.CheckUserArg(!string.IsNullOrEmpty(name), nameof(Options.Columns), "Column name cannot be empty");
                    int col;
                    if (!inputSchema.TryGetColumnIndex(name, out col))
                        throw ectx.ExceptUserArg(nameof(Options.Columns), "Pivot column '{0}' is not found", name);
                    if (!(inputSchema[col].Type is VectorType colType))
                        throw ectx.ExceptUserArg(nameof(Options.Columns),
                            "Pivot column '{0}' has type '{1}', but must be a vector of primitive types", name, inputSchema[col].Type);
                    infos[i] = new PivotColumnOptions(name, col, colType.Size, colType.ItemType);
                }
            }

            public static UngroupBinding Create(ModelLoadContext ctx, IExceptionContext ectx, DataViewSchema inputSchema)
            {
                Contracts.AssertValueOrNull(ectx);
                ectx.AssertValue(ctx);
                ectx.AssertValue(inputSchema);

                // *** Binary format ***
                // int: ungroup mode
                // int: K - number of pivot columns
                // int[K]: ids of pivot column names

                int modeIndex = ctx.Reader.ReadInt32();
                ectx.CheckDecode(Enum.IsDefined(typeof(UngroupMode), modeIndex));
                UngroupMode mode = (UngroupMode)modeIndex;

                int k = ctx.Reader.ReadInt32();
                ectx.CheckDecode(k > 0);
                var pivotColumns = new string[k];
                for (int i = 0; i < k; i++)
                    pivotColumns[i] = ctx.LoadNonEmptyString();

                return new UngroupBinding(ectx, inputSchema, mode, pivotColumns);
            }

            internal void Save(ModelSaveContext ctx)
            {
                _ectx.AssertValue(ctx);

                // *** Binary format ***
                // int: ungroup mode
                // int: K - number of pivot columns
                // int[K]: ids of pivot column names

                ctx.Writer.Write((int)Mode);
                ctx.Writer.Write(_infos.Length);
                foreach (var ex in _infos)
                    ctx.SaveNonEmptyString(ex.Name);
            }

            /// <summary>
            /// Return an array of active input columns given the target predicate.
            /// </summary>
            public bool[] GetActiveInput(Func<int, bool> predicate)
            {
                var activeInput = Utils.BuildArray(_inputSchema.Count, predicate);
                for (int i = 0; i < _infos.Length; i++)
                {
                    bool isNeededForSize = (_infos[i].Size == 0) && (i == 0 || Mode != UngroupMode.First);
                    activeInput[_infos[i].Index] |= isNeededForSize;
                }
                return activeInput;
            }

            public int PivotColumnCount
            {
                get { return _infos.Length; }
            }

            public PivotColumnOptions GetPivotColumnOptions(int iinfo)
            {
                _ectx.Assert(0 <= iinfo && iinfo < _infos.Length);
                return _infos[iinfo];
            }

            public PivotColumnOptions GetPivotColumnOptionsByCol(int col)
            {
                _ectx.Assert(0 <= col && col < _inputSchema.Count);
                _ectx.Assert(_pivotIndex[col] >= 0);
                return _infos[_pivotIndex[col]];
            }

            /// <summary>
            /// Determine if an output column is produced by a pivot column from input.
            /// </summary>
            /// <param name="col">Column index in <see cref="OutputSchema"/></param>
            /// <returns>True if the specified column is produced by expanding a pivot column and false otherwise.</returns>
            public bool IsPivot(int col)
            {
                _ectx.Assert(0 <= col && col < _inputSchema.Count);
                return _pivotIndex[col] >= 0;
            }

            public int GetCommonPivotColumnSize()
            {
                if (Mode == UngroupMode.First)
                    return _infos[0].Size;

                var size = 0;
                foreach (var ex in _infos)
                {
                    if (ex.Size == 0)
                        return 0;
                    if (size == 0)
                        size = ex.Size;
                    else if (size > ex.Size && Mode == UngroupMode.Inner)
                        size = ex.Size;
                    else if (size < ex.Size && Mode == UngroupMode.Outer)
                        size = ex.Size;
                }
                return size;
            }
        }

        private sealed class Cursor : LinkedRootCursorBase
        {
            private readonly UngroupBinding _ungroupBinding;

            // The size of the pivot column in the current row. If the cursor is in good state, this is positive.
            // It's calculated on every row, based on UngroupMode.
            private int _pivotColSize;
            // The current position within the pivot columns. If the cursor is in good state, this is in [0, _pivotColSize).
            private int _pivotColPosition;

            /// <summary>
            /// Total number of input columns is <see cref="UngroupBinding.InputColumnCount"/> of <see cref="_ungroupBinding"/>.
            /// Note that the number of input columns equals to the number of output columns; that is, <see cref="UngroupBinding.InputColumnCount"/>
            /// is identical to the number of columns in <see cref="UngroupBinding.OutputSchema"/>.
            /// </summary>
            private readonly bool[] _active;

            // Getters for pivot columns. Cached on first creation. Parallel to columns, and always null for non-pivot columns.
            private readonly Delegate[] _cachedGetters;

            // Size (min, max or target) of fixed columns. Zero here means that there are no fixed-size columns.
            private readonly int _fixedSize;

            // For each pivot column that we care about, these getters return the vector size.
            private readonly Func<int>[] _sizeGetters;

            // As a side effect, getters also populate these actual sizes of the necessary pivot columns on MoveNext.
            // Parallel to columns.
            private int[] _colSizes;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, UngroupBinding schema, Func<int, bool> predicate)
                : base(provider, input)
            {
                _ungroupBinding = schema;
                _active = Utils.BuildArray(_ungroupBinding.InputColumnCount, predicate);
                _cachedGetters = new Delegate[_ungroupBinding.InputColumnCount];
                _colSizes = new int[_ungroupBinding.InputColumnCount];

                int sizeColumnsLim = _ungroupBinding.Mode == UngroupMode.First ? 1 : _ungroupBinding.PivotColumnCount;
                _fixedSize = 0;
                var needed = new List<Func<int>>();
                for (int i = 0; i < sizeColumnsLim; i++)
                {
                    var info = _ungroupBinding.GetPivotColumnOptions(i);
                    if (info.Size > 0)
                    {
                        if (_fixedSize == 0)
                            _fixedSize = info.Size;
                        else if (_ungroupBinding.Mode == UngroupMode.Inner && _fixedSize > info.Size)
                            _fixedSize = info.Size;
                        else if (_ungroupBinding.Mode == UngroupMode.Outer && _fixedSize < info.Size)
                            _fixedSize = info.Size;
                    }
                    else
                    {
                        // This will also create and cache a getter for the pivot column.
                        // That's why MakeSizeGetter is an instance method.
                        var rawItemType = info.ItemType.RawType;
                        Func<int, Func<int>> del = MakeSizeGetter<int>;
                        var mi = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(rawItemType);
                        var sizeGetter = (Func<int>)mi.Invoke(this, new object[] { info.Index });
                        needed.Add(sizeGetter);
                    }
                }

                _sizeGetters = needed.ToArray();
                Ch.Assert(_fixedSize > 0 || _sizeGetters.Length > 0);

            }

            public override long Batch => Input.Batch;

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                var idGetter = Input.GetIdGetter();
                return (ref DataViewRowId val) =>
                {
                    idGetter(ref val);
                    val = val.Combine(new DataViewRowId((ulong)_pivotColPosition, 0));
                };
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(Position < 0 || (0 <= _pivotColPosition && _pivotColPosition < _pivotColSize));
                // In the very first call to MoveNext, both _pivotColPosition and _pivotColSize are equal to zero.
                // So, the below code will work seamlessly, advancing the input cursor.

                _pivotColPosition++;
                while (_pivotColPosition >= _pivotColSize)
                {
                    bool result = Input.MoveNext();
                    if (!result)
                        return false;

                    _pivotColPosition = 0;
                    _pivotColSize = CalcPivotColSize();
                    // If the current input row's pivot column size is zero, the condition in while loop will be true again,
                    // and we'll skip the row.
                }

                return true;
            }

            private int CalcPivotColSize()
            {
                var size = _fixedSize;

                foreach (var getter in _sizeGetters)
                {
                    var colSize = getter();
                    if (_ungroupBinding.Mode == UngroupMode.Inner && colSize == 0)
                        return 0;

                    if (size == 0)
                        size = colSize;
                    else if (_ungroupBinding.Mode == UngroupMode.Inner && size > colSize)
                        size = colSize;
                    else if (_ungroupBinding.Mode == UngroupMode.Outer && size < colSize)
                        size = colSize;
                }

                return size;
            }

            /// <summary>
            /// Create a getter which returns the length of a vector (aka a column's value) in the input data.
            /// </summary>
            /// <typeparam name="T">The type of the considered input vector</typeparam>
            /// <param name="col">Column index, which should point to a vector-typed column in the input data.</param>
            /// <returns>Getter of the length to the considered input vector.</returns>
            private Func<int> MakeSizeGetter<T>(int col)
            {
                Contracts.Assert(0 <= col && col < _ungroupBinding.InputColumnCount);

                var srcGetter = GetGetter<T>(Schema[col]);
                var cur = default(T);

                return
                    () =>
                    {
                        srcGetter(ref cur);
                        // We don't care about cur, we only need _colSizes to be populated.
                        return _colSizes[col];
                    };
            }

            public override DataViewSchema Schema => _ungroupBinding.OutputSchema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _ungroupBinding.InputColumnCount);
                return _active[column.Index];
            }

            /// <summary>
            /// Returns the getter of an output column.
            /// </summary>
            /// <typeparam name="TValue"> is the output column's content type, for example, <see cref="VBuffer{T}"/>.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                // Although the input argument, col, is a output index, we check its range as if it's an input column index.
                // It makes sense because the i-th output column is produced by either expanding or copying the i-th input column.
                Ch.CheckParam(column.Index < _ungroupBinding.InputColumnCount, nameof(column));

                if (!_ungroupBinding.IsPivot(column.Index))
                    return Input.GetGetter<TValue>(column);

                if (_cachedGetters[column.Index] == null)
                    _cachedGetters[column.Index] = MakeGetter<TValue>(column.Index, _ungroupBinding.GetPivotColumnOptionsByCol(column.Index).ItemType);

                var result = _cachedGetters[column.Index] as ValueGetter<TValue>;
                Ch.Check(result != null, "Unexpected getter type requested");
                return result;
            }

            private ValueGetter<T> MakeGetter<T>(int col, PrimitiveDataViewType itemType)
            {
                var srcGetter = Input.GetGetter<VBuffer<T>>(Input.Schema[col]);
                // The position of the source cursor. Used to extract the source row once.
                long cachedPosition = -1;
                // The position inside the sparse row. If the row is sparse, the invariant is
                // cachedIndex == row.Count || _pivotColPosition <= row.Indices[cachedIndex].
                int cachedIndex = 0;
                VBuffer<T> row = default(VBuffer<T>);
                T naValue = Data.Conversion.Conversions.Instance.GetNAOrDefault<T>(itemType);
                return
                    (ref T value) =>
                    {
                        // This delegate can be called from within MoveNext, so our own IsGood is not yet set.
                        Ch.Check(Input.Position >= 0, RowCursorUtils.FetchValueStateError);

                        Ch.Assert(cachedPosition <= Input.Position);
                        if (cachedPosition < Input.Position)
                        {
                            srcGetter(ref row);
                            // Side effect: populate the column size.
                            _colSizes[col] = row.Length;
                            cachedPosition = Input.Position;
                            cachedIndex = 0;
                        }

                        var rowValues = row.GetValues();
                        if (_pivotColPosition >= row.Length)
                            value = naValue;
                        else if (row.IsDense)
                            value = rowValues[_pivotColPosition];
                        else
                        {
                            // The row is sparse.
                            var rowIndices = row.GetIndices();
                            while (cachedIndex < rowIndices.Length && _pivotColPosition > rowIndices[cachedIndex])
                                cachedIndex++;

                            if (cachedIndex < rowIndices.Length && _pivotColPosition == rowIndices[cachedIndex])
                                value = rowValues[cachedIndex];
                            else
                                value = default(T);
                        }
                    };
            }
        }
    }

    internal static partial class GroupingOperations
    {
        [TlcModule.EntryPoint(Name = "Transforms.Segregator",
            Desc = UngroupTransform.Summary,
            UserName = UngroupTransform.UserName,
            ShortName = UngroupTransform.ShortName)]
        public static CommonOutputs.TransformOutput Ungroup(IHostEnvironment env, UngroupTransform.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "UngroupTransform", input);
            var view = new UngroupTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
