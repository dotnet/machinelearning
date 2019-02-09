// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(GroupTransform.Summary, typeof(GroupTransform), typeof(GroupTransform.Options), typeof(SignatureDataTransform),
    GroupTransform.UserName, GroupTransform.ShortName)]

[assembly: LoadableClass(GroupTransform.Summary, typeof(GroupTransform), null, typeof(SignatureLoadDataTransform),
    GroupTransform.UserName, GroupTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(Microsoft.ML.Transforms.GroupingOperations))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// A Trasforms that groups values of a scalar column into a vector, by a contiguous group ID.
    /// </summary>
    /// <remarks>
    /// <p>This transform essentially performs the following SQL-like operation:</p>
    /// <p>SELECT GroupKey1, GroupKey2, ... GroupKeyK, LIST(Value1), LIST(Value2), ... LIST(ValueN)
    /// FROM Data
    /// GROUP BY GroupKey1, GroupKey2, ... GroupKeyK.</p>
    ///
    /// <p>It assumes that the group keys are contiguous (if a new group key sequence is encountered, the group is over).
    /// The GroupKeyN and ValueN columns can be of any primitive types. The code requires that every raw type T of the group key column
    /// is an <see cref="IEquatable{T}"/>, which is currently true for all existing primitive types.
    /// The produced ValueN columns will be variable-length vectors of the original value column types.</p>
    ///
    /// <p>The order of ValueN entries in the lists is preserved.</p>
    ///
    /// <example><code>
    /// Example:
    /// User Item
    /// Pete Book
    /// Tom  Table
    /// Tom  Kitten
    /// Pete Chair
    /// Pete Cup
    ///
    /// Result:
    /// User Item
    /// Pete [Book]
    /// Tom  [Table, Kitten]
    /// Pete [Chair, Cup]
    /// </code></example>
    /// </remarks>
    internal sealed class GroupTransform : TransformBase
    {
        internal const string Summary = "Groups values of a scalar column into a vector, by a contiguous group ID";
        internal const string UserName = "Group Transform";
        internal const string ShortName = "Group";
        private const string RegistrationName = "GroupTransform";
        internal const string LoaderSignature = "GroupTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GRP TRNS",
                 verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(GroupTransform).Assembly.FullName);
        }

        // REVIEW: maybe we want to have an option to keep all non-group scalar columns, as opposed to
        // explicitly listing the ones to keep.

        // REVIEW: group keys and keep columns can possibly be vectors, not implemented now.

        // REVIEW: it might be feasible to have columns that are constant throughout a group, without having to list them
        // as group keys.
        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Columns to group by", Name = "GroupKey", ShortName = "g", SortOrder = 1,
                Purpose = SpecialPurpose.ColumnSelector)]
            public string[] GroupKeys;

            // The column names remain the same, there's no option to rename the column.
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to group together", Name = "Column", ShortName = "col", SortOrder = 2)]
            public string[] Columns;
        }

        private readonly GroupBinding _groupBinding;

        /// <summary>
        /// Initializes a new instance of <see cref="GroupTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="groupKey">Columns to group by</param>
        /// <param name="columns">Columns to group together</param>
        public GroupTransform(IHostEnvironment env, IDataView input, string groupKey, params string[] columns)
            : this(env, new Options() { GroupKeys = new[] { groupKey }, Columns = columns }, input)
        {
        }

        public GroupTransform(IHostEnvironment env, Options options, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckUserArg(Utils.Size(options.GroupKeys) > 0, nameof(options.GroupKeys), "There must be at least one group key");

            _groupBinding = new GroupBinding(Host, Source.Schema, options.GroupKeys, options.Columns ?? new string[0]);
        }

        public static GroupTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            var h = env.Register(RegistrationName);
            return h.Apply("Loading Model", ch => new GroupTransform(h, ctx, input));
        }

        private GroupTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // (GroupBinding)
            _groupBinding = new GroupBinding(input.Schema, host, ctx);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // (GroupBinding)
            _groupBinding.Save(ctx);
        }

        public override long? GetRowCount()
        {
            // We have no idea how many total rows we'll have.
            return null;
        }

        public override Schema OutputSchema => _groupBinding.OutputSchema;

        protected override RowCursor GetRowCursorCore(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            return new Cursor(this, predicate);
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            // There's no way to parallelize the processing: we can't ensure every group belongs to one batch.
            Host.AssertValue(predicate);
            return false;
        }

        public override bool CanShuffle { get { return false; } }

        public override RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            return new RowCursor[] { GetRowCursorCore(columnsNeeded) };
        }

        /// <summary>
        /// This class describes the relation between <see cref="GroupTransform"/>'s input <see cref="Schema"/>,
        /// <see cref="GroupBinding._inputSchema"/>, and output <see cref="Schema"/>, <see cref="GroupBinding.OutputSchema"/>.
        ///
        /// The <see cref="GroupBinding.OutputSchema"/> contains columns used to group columns and columns being aggregated from input data.
        /// In <see cref="GroupBinding.OutputSchema"/>, group columns are followed by aggregated columns. For example, if column "Age" is used to group "UserId" column,
        /// the first column and the second column in <see cref="GroupBinding.OutputSchema"/> produced by <see cref="GroupTransform"/> would be "Age" and "UserId," respectively.
        /// Note that "Age" is a group column while "UserId" is an aggregated (also call keep) column.
        ///
        /// For group columns, the schema information is intact. For aggregated columns, the type is Vector of original type and variable length.
        /// The only metadata preserved is the KeyNames and IsNormalized. All other columns are dropped. Please see
        /// <see cref="GroupBinding.BuildOutputSchema(Schema)"/> how this idea got implemented.
        /// </summary>
        private sealed class GroupBinding
        {
            private readonly IExceptionContext _ectx;
            private readonly Schema _inputSchema;

            // Column names in source schema used to group rows.
            private readonly string[] _groupColumns;
            // Column names in source schema aggregated into row's vector-typed columns.
            private readonly string[] _keepColumns;

            /// <summary>
            /// <see cref="GroupColumnIndexes"/>[i] is the i-th group(-key) column's column index in the source schema.
            /// </summary>
            public readonly int[] GroupColumnIndexes;
            /// <summary>
            /// <see cref="KeepColumnIndexes"/>[i] is the i-th aggregated column's column index in the source schema.
            /// </summary>
            public readonly int[] KeepColumnIndexes;

            /// <summary>
            /// Output <see cref="Schema"/> of <see cref="GroupTransform"/> when input schema is <see cref="_inputSchema"/>.
            /// </summary>
            public Schema OutputSchema { get; }

            public GroupBinding(IExceptionContext ectx, Schema inputSchema, string[] groupColumns, string[] keepColumns)
            {
                Contracts.AssertValue(ectx);
                _ectx = ectx;
                _ectx.AssertValue(inputSchema);
                _ectx.AssertNonEmpty(groupColumns);
                _ectx.AssertValue(keepColumns);
                _inputSchema = inputSchema;

                _groupColumns = groupColumns;
                GroupColumnIndexes = GetColumnIds(inputSchema, groupColumns, x => _ectx.ExceptUserArg(nameof(Options.GroupKeys), x));

                _keepColumns = keepColumns;
                KeepColumnIndexes = GetColumnIds(inputSchema, keepColumns, x => _ectx.ExceptUserArg(nameof(Options.Columns), x));

                // Compute output schema from the specified input schema.
                OutputSchema = BuildOutputSchema(inputSchema);
            }

            public GroupBinding(Schema inputSchema, IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.AssertValue(env);
                _ectx = env.Register(LoaderSignature);
                _ectx.AssertValue(inputSchema);
                _ectx.AssertValue(ctx);

                // *** Binary format ***
                // int: G - number of group columns
                // int[G]: ids of group column names
                // int: K: number of keep columns
                // int[K]: ids of keep column names
                _inputSchema = inputSchema;

                // Load group columns.
                int g = ctx.Reader.ReadInt32();
                _ectx.CheckDecode(g > 0);
                _groupColumns = new string[g];
                for (int i = 0; i < g; i++)
                    _groupColumns[i] = ctx.LoadNonEmptyString();

                // Load keep columns (aka columns being aggregated).
                int k = ctx.Reader.ReadInt32();
                _ectx.CheckDecode(k >= 0);
                _keepColumns = new string[k];
                for (int i = 0; i < k; i++)
                    _keepColumns[i] = ctx.LoadNonEmptyString();

                // Translate column names to column indexes in source schema.
                GroupColumnIndexes = GetColumnIds(inputSchema, _groupColumns, _ectx.Except);
                KeepColumnIndexes = GetColumnIds(inputSchema, _keepColumns, _ectx.Except);

                // Compute output schema from the specified input schema.
                OutputSchema = BuildOutputSchema(inputSchema);
            }

            /// <summary>
            /// Compute the output schema of a <see cref="GroupTransform"/> given a input schema.
            /// </summary>
            /// <param name="sourceSchema">Input schema.</param>
            /// <returns>The associated output schema produced by <see cref="GroupTransform"/>.</returns>
            private Schema BuildOutputSchema(Schema sourceSchema)
            {
                // Create schema build. We will sequentially add group columns and then aggregated columns.
                var schemaBuilder = new SchemaBuilder();

                // Handle group(-key) columns. Those columns are used as keys to partition rows in the input data; specifically,
                // rows with the same key value will be merged into one row in the output data.
                foreach (var groupKeyColumnName in _groupColumns)
                    schemaBuilder.AddColumn(groupKeyColumnName, sourceSchema[groupKeyColumnName].Type, sourceSchema[groupKeyColumnName].Metadata);

                // Handle aggregated (aka keep) columns.
                foreach (var groupValueColumnName in _keepColumns)
                {
                    // Prepare column's metadata.
                    var metadataBuilder = new MetadataBuilder();
                    metadataBuilder.Add(sourceSchema[groupValueColumnName].Metadata,
                        s => s == MetadataUtils.Kinds.IsNormalized || s == MetadataUtils.Kinds.KeyValues);

                    // Prepare column's type.
                    var aggregatedValueType = sourceSchema[groupValueColumnName].Type as PrimitiveType;
                    _ectx.CheckValue(aggregatedValueType, nameof(aggregatedValueType), "Columns being aggregated must be primitive types such as string, float, or integer");
                    var aggregatedResultType = new VectorType(aggregatedValueType);

                    // Add column into output schema.
                    schemaBuilder.AddColumn(groupValueColumnName, aggregatedResultType, metadataBuilder.GetMetadata());
                }

                return schemaBuilder.GetSchema();
            }

            internal void Save(ModelSaveContext ctx)
            {
                _ectx.AssertValue(ctx);

                // *** Binary format ***
                // int: G - number of group columns
                // int[G]: ids of group column names
                // int: K: number of keep columns
                // int[K]: ids of keep column names

                _ectx.AssertNonEmpty(_groupColumns);
                ctx.Writer.Write(_groupColumns.Length);
                foreach (var name in _groupColumns)
                {
                    _ectx.AssertNonEmpty(name);
                    ctx.SaveString(name);
                }

                _ectx.AssertValue(_keepColumns);
                ctx.Writer.Write(_keepColumns.Length);
                foreach (var name in _keepColumns)
                {
                    _ectx.AssertNonEmpty(name);
                    ctx.SaveString(name);
                }
            }

            /// <summary>
            /// Given column names, extract and return column indexes from source schema.
            /// </summary>
            /// <param name="schema">Source schema</param>
            /// <param name="names">Column names</param>
            /// <param name="except">Marked exception function</param>
            /// <returns>column indexes</returns>
            private int[] GetColumnIds(Schema schema, string[] names, Func<string, Exception> except)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertValue(names);

                var ids = new int[names.Length];

                for (int i = 0; i < names.Length; i++)
                {
                    // Find column called names[i] from input schema.
                    var retrievedColumn = schema.GetColumnOrNull(names[i]);

                    // Throw if no such a column in schema.
                    var errorMessage = string.Format("Could not find column '{0}'", names[i]);
                    _ectx.Check(retrievedColumn.HasValue, errorMessage);

                    var colType = retrievedColumn.Value.Type;
                    errorMessage = string.Format("Column '{0}' has type '{1}', but must have a primitive type", names[i], colType);
                    _ectx.Check(colType is PrimitiveType, errorMessage);

                    ids[i] = retrievedColumn.Value.Index;
                }

                return ids;
            }

            /// <summary>
            /// Determine if output column index is valid to <see cref="OutputSchema"/>. A valid output column index should be greater than or
            /// equal 0 and smaller than # of output columns.
            /// </summary>
            /// <param name="col">Column index of <see cref="OutputSchema"/></param>
            public void CheckColumnInRange(int col)
            {
                _ectx.Check(0 <= col && col < GroupColumnIndexes.Length + KeepColumnIndexes.Length);
            }
        }

        /// <summary>
        /// This cursor will create two cursors on the input data view:
        /// - The leading cursor will activate all the group columns, and will advance until it hits the end of the contiguous group.
        /// - The trailing cursor will activate all the requested columns, and will go through the group
        ///     (as identified by the leading cursor), and aggregate the keep columns.
        ///
        /// The getters are as follows:
        /// - The group column getters are taken directly from the trailing cursor.
        /// - The keep column getters are provided by the aggregators.
        /// </summary>
        private sealed class Cursor : RootCursorBase
        {
            /// <summary>
            /// This class keeps track of the previous group key and tests the current group key against the previous one.
            /// </summary>
            private sealed class GroupKeyColumnChecker
            {
                public readonly Func<bool> IsSameKey;

                private static Func<bool> MakeSameChecker<T>(Row row, int col)
                {
                    T oldValue = default(T);
                    T newValue = default(T);
                    bool first = true;
                    ValueGetter<T> getter = row.GetGetter<T>(col);
                    return
                        () =>
                        {
                            getter(ref newValue);
                            bool result;

                            if ((typeof(IEquatable<T>).IsAssignableFrom(typeof(T))))
                                result = oldValue.Equals(newValue);
                            else if ((typeof(ReadOnlyMemory<char>).IsAssignableFrom(typeof(T))))
                                result = ((ReadOnlyMemory<char>)(object)oldValue).Span.SequenceEqual(((ReadOnlyMemory<char>)(object)newValue).Span);
                            else
                                Contracts.Check(result = false, "Invalid type.");

                            result = result || first;
                            oldValue = newValue;
                            first = false;
                            return result;
                        };
                }

                public GroupKeyColumnChecker(Row row, int col)
                {
                    Contracts.AssertValue(row);
                    var type = row.Schema[col].Type;

                    Func<Row, int, Func<bool>> del = MakeSameChecker<int>;
                    var mi = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
                    IsSameKey = (Func<bool>)mi.Invoke(null, new object[] { row, col });
                }
            }

            // REVIEW: potentially, there could be other aggregators.
            // REVIEW: Currently, it always produces dense buffers. The anticipated use cases don't include many
            // default values at the moment.
            /// <summary>
            /// This class handles the aggregation of one 'keep' column into a vector. It wraps around an <see cref="Row"/>'s
            /// column, reads the data and aggregates.
            /// </summary>
            private abstract class KeepColumnAggregator
            {
                public abstract ValueGetter<T> GetGetter<T>(IExceptionContext ctx);
                public abstract void SetSize(int size);
                public abstract void ReadValue(int position);

                public static KeepColumnAggregator Create(Row row, int col)
                {
                    Contracts.AssertValue(row);
                    var colType = row.Schema[col].Type;
                    Contracts.Assert(colType is PrimitiveType);

                    var type = typeof(ListAggregator<>);

                    var cons = type.MakeGenericType(colType.RawType).GetConstructor(new[] { typeof(Row), typeof(int) });
                    return cons.Invoke(new object[] { row, col }) as KeepColumnAggregator;
                }

                private sealed class ListAggregator<TValue> : KeepColumnAggregator
                {
                    private readonly ValueGetter<TValue> _srcGetter;
                    private readonly Delegate _getter;
                    private TValue[] _buffer;
                    private int _size;

                    public ListAggregator(Row row, int col)
                    {
                        Contracts.AssertValue(row);
                        _srcGetter = row.GetGetter<TValue>(col);
                        _getter = (ValueGetter<VBuffer<TValue>>)Getter;
                    }

                    private void Getter(ref VBuffer<TValue> dst)
                    {
                        var editor = VBufferEditor.Create(ref dst, _size);
                        _buffer.AsSpan(0, _size).CopyTo(editor.Values);
                        dst = editor.Commit();
                    }

                    public override ValueGetter<T> GetGetter<T>(IExceptionContext ctx)
                    {
                        ctx.Check(typeof(T) == typeof(VBuffer<TValue>));
                        return (ValueGetter<T>)_getter;
                    }

                    public override void SetSize(int size)
                    {
                        Array.Resize(ref _buffer, size);
                        _size = size;
                    }

                    public override void ReadValue(int position)
                    {
                        Contracts.Assert(0 <= position && position < _size);
                        _srcGetter(ref _buffer[position]);
                    }
                }
            }

            private readonly GroupTransform _parent;
            private readonly bool[] _active;
            private readonly int _groupCount;

            private readonly RowCursor _leadingCursor;
            private readonly RowCursor _trailingCursor;

            private readonly GroupKeyColumnChecker[] _groupCheckers;
            private readonly KeepColumnAggregator[] _aggregators;

            public override long Batch => 0;

            public override Schema Schema => _parent.OutputSchema;

            public Cursor(GroupTransform parent, Func<int, bool> predicate)
                : base(parent.Host)
            {
                Ch.AssertValue(predicate);

                _parent = parent;
                var binding = _parent._groupBinding;
                _active = Utils.BuildArray(binding.OutputSchema.Count, predicate);
                _groupCount = binding.GroupColumnIndexes.Length;

                bool[] srcActiveLeading = new bool[_parent.Source.Schema.Count];
                foreach (var col in binding.GroupColumnIndexes)
                    srcActiveLeading[col] = true;
                var activeCols = _parent.Source.Schema.Where(x => x.Index < srcActiveLeading.Length && srcActiveLeading[x.Index]);
                _leadingCursor = parent.Source.GetRowCursor(activeCols);

                bool[] srcActiveTrailing = new bool[_parent.Source.Schema.Count];
                for (int i = 0; i < _groupCount; i++)
                {
                    if (_active[i])
                        srcActiveTrailing[binding.GroupColumnIndexes[i]] = true;
                }
                for (int i = 0; i < binding.KeepColumnIndexes.Length; i++)
                {
                    if (_active[i + _groupCount])
                        srcActiveTrailing[binding.KeepColumnIndexes[i]] = true;
                }

                activeCols = _parent.Source.Schema.Where(x => x.Index < srcActiveTrailing.Length && srcActiveTrailing[x.Index]);
                _trailingCursor = parent.Source.GetRowCursor(activeCols);

                _groupCheckers = new GroupKeyColumnChecker[_groupCount];
                for (int i = 0; i < _groupCount; i++)
                    _groupCheckers[i] = new GroupKeyColumnChecker(_leadingCursor, _parent._groupBinding.GroupColumnIndexes[i]);

                _aggregators = new KeepColumnAggregator[_parent._groupBinding.KeepColumnIndexes.Length];
                for (int i = 0; i < _aggregators.Length; i++)
                {
                    if (_active[i + _groupCount])
                        _aggregators[i] = KeepColumnAggregator.Create(_trailingCursor, _parent._groupBinding.KeepColumnIndexes[i]);
                }
            }

            public override ValueGetter<RowId> GetIdGetter()
            {
                return _trailingCursor.GetIdGetter();
            }

            public override bool IsColumnActive(int col)
            {
                _parent._groupBinding.CheckColumnInRange(col);
                return _active[col];
            }

            protected override bool MoveNextCore()
            {
                // If leading cursor is not started, start it.
                // But, if in moving it we find we've reached the end, we have the degenerate case where
                // there are no rows, in which case we ourselves should return false immedaitely.

                if (_leadingCursor.Position < 0 && !_leadingCursor.MoveNext())
                    return false;
                Ch.Assert(_leadingCursor.Position >= 0);

                // We are now in a "valid" place. Advance the leading cursor until it hits
                // the end of the group (or the end of the data).
                int groupSize = 0;
                while (_leadingCursor.Position >= 0 && IsSameGroup())
                {
                    groupSize++;
                    if (!_leadingCursor.MoveNext())
                        break;
                }

                // The group can only be empty if the leading cursor immediately reaches the end of the data.
                // This is handled by the check above.
                Ch.Assert(groupSize > 0);

                // Catch up with the trailing cursor and populate all the aggregates.
                // REVIEW: this could be done lazily, but still all aggregators together.
                foreach (var agg in _aggregators.Where(x => x != null))
                    agg.SetSize(groupSize);

                for (int i = 0; i < groupSize; i++)
                {
                    var res = _trailingCursor.MoveNext();
                    Ch.Assert(res);

                    foreach (var agg in _aggregators.Where(x => x != null))
                        agg.ReadValue(i);
                }

                return true;
            }

            private bool IsSameGroup()
            {
                bool result = true;
                foreach (var checker in _groupCheckers)
                {
                    // Even if the result is false, we need to call every checker so that they can memorize
                    // the current key value.
                    result = checker.IsSameKey() & result;
                }
                return result;
            }

            private bool _disposed;

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _leadingCursor.Dispose();
                    _trailingCursor.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                _parent._groupBinding.CheckColumnInRange(col);
                if (!_active[col])
                    throw Ch.ExceptParam(nameof(col), "Column #{0} is not active", col);

                if (col < _groupCount)
                    return _trailingCursor.GetGetter<TValue>(_parent._groupBinding.GroupColumnIndexes[col]);

                Ch.AssertValue(_aggregators[col - _groupCount]);
                return _aggregators[col - _groupCount].GetGetter<TValue>(Ch);
            }
        }
    }

    internal static partial class GroupingOperations
    {
        [TlcModule.EntryPoint(Name = "Transforms.CombinerByContiguousGroupId",
            Desc = GroupTransform.Summary,
            UserName = GroupTransform.UserName,
            ShortName = GroupTransform.ShortName)]
        public static CommonOutputs.TransformOutput Group(IHostEnvironment env, GroupTransform.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GroupTransform", input);
            var view = new GroupTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
