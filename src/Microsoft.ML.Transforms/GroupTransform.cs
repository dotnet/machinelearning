// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(GroupTransform.Summary, typeof(GroupTransform), typeof(GroupTransform.Arguments), typeof(SignatureDataTransform),
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
    public sealed class GroupTransform : TransformBase
    {
        public const string Summary = "Groups values of a scalar column into a vector, by a contiguous group ID";
        public const string UserName = "Group Transform";
        public const string ShortName = "Group";
        private const string RegistrationName = "GroupTransform";
        public const string LoaderSignature = "GroupTransform";

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
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Columns to group by", ShortName = "g", SortOrder = 1,
                Purpose = SpecialPurpose.ColumnSelector)]
            public string[] GroupKey;

            // The column names remain the same, there's no option to rename the column.
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to group together", ShortName = "col", SortOrder = 2)]
            public string[] Column;
        }

        private readonly GroupSchema _groupSchema;

        /// <summary>
        /// Initializes a new instance of <see cref="GroupTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="groupKey">Columns to group by</param>
        /// <param name="columns">Columns to group together</param>
        public GroupTransform(IHostEnvironment env, IDataView input, string groupKey, params string[] columns)
            : this(env, new Arguments() { GroupKey = new[] { groupKey }, Column = columns }, input)
        {
        }

        public GroupTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.GroupKey) > 0, nameof(args.GroupKey), "There must be at least one group key");

            _groupSchema = new GroupSchema(Host, Source.Schema, args.GroupKey, args.Column ?? new string[0]);
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
            // (schema)
            _groupSchema = new GroupSchema(input.Schema, host, ctx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // (schema)
            _groupSchema.Save(ctx);
        }

        public override long? GetRowCount(bool lazy = true)
        {
            // We have no idea how many total rows we'll have.
            return null;
        }

        public override Schema Schema => _groupSchema.AsSchema;

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            return new Cursor(this, predicate);
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            // There's no way to parallelize the processing: we can't ensure every group belongs to one batch.
            Host.AssertValue(predicate);
            return false;
        }

        public override bool CanShuffle { get { return false; } }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursorCore(predicate) };
        }

        /// <summary>
        /// For group columns, the schema information is intact.
        ///
        /// For keep columns, the type is Vector of original type and variable length.
        /// The only metadata preserved is the KeyNames and IsNormalized.
        ///
        /// All other columns are dropped.
        /// </summary>
        private sealed class GroupSchema : ISchema
        {
            private static readonly string[] _preservedMetadata =
                new[] { MetadataUtils.Kinds.IsNormalized, MetadataUtils.Kinds.KeyValues };

            private readonly IExceptionContext _ectx;
            private readonly ISchema _input;

            private readonly string[] _groupColumns;
            private readonly string[] _keepColumns;

            public readonly int[] GroupIds;
            public readonly int[] KeepIds;

            private readonly int _groupCount;
            private readonly ColumnType[] _columnTypes;

            private readonly Dictionary<string, int> _columnNameMap;

            public Schema AsSchema { get; }

            public GroupSchema(IExceptionContext ectx, ISchema inputSchema, string[] groupColumns, string[] keepColumns)
            {
                Contracts.AssertValue(ectx);
                _ectx = ectx;
                _ectx.AssertValue(inputSchema);
                _ectx.AssertNonEmpty(groupColumns);
                _ectx.AssertValue(keepColumns);
                _input = inputSchema;

                _groupColumns = groupColumns;
                GroupIds = GetColumnIds(inputSchema, groupColumns, x => _ectx.ExceptUserArg(nameof(Arguments.GroupKey), x));
                _groupCount = GroupIds.Length;

                _keepColumns = keepColumns;
                KeepIds = GetColumnIds(inputSchema, keepColumns, x => _ectx.ExceptUserArg(nameof(Arguments.Column), x));

                _columnTypes = BuildColumnTypes(_input, KeepIds);
                _columnNameMap = BuildColumnNameMap();

                AsSchema = Schema.Create(this);
            }

            public GroupSchema(ISchema inputSchema, IHostEnvironment env, ModelLoadContext ctx)
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
                _input = inputSchema;

                int g = ctx.Reader.ReadInt32();
                _ectx.CheckDecode(g > 0);
                _groupColumns = new string[g];
                for (int i = 0; i < g; i++)
                    _groupColumns[i] = ctx.LoadNonEmptyString();

                int k = ctx.Reader.ReadInt32();
                _ectx.CheckDecode(k >= 0);
                _keepColumns = new string[k];
                for (int i = 0; i < k; i++)
                    _keepColumns[i] = ctx.LoadNonEmptyString();

                GroupIds = GetColumnIds(inputSchema, _groupColumns, _ectx.Except);
                _groupCount = GroupIds.Length;

                KeepIds = GetColumnIds(inputSchema, _keepColumns, _ectx.Except);

                _columnTypes = BuildColumnTypes(_input, KeepIds);
                _columnNameMap = BuildColumnNameMap();

                AsSchema = Schema.Create(this);
            }

            private Dictionary<string, int> BuildColumnNameMap()
            {
                var map = new Dictionary<string, int>();
                for (int i = 0; i < _groupCount; i++)
                    map[_groupColumns[i]] = i;

                for (int i = 0; i < _keepColumns.Length; i++)
                    map[_keepColumns[i]] = i + _groupCount;

                return map;
            }

            private static ColumnType[] BuildColumnTypes(ISchema input, int[] ids)
            {
                var types = new ColumnType[ids.Length];
                for (int i = 0; i < ids.Length; i++)
                {
                    var srcType = input.GetColumnType(ids[i]);
                    Contracts.Assert(srcType.IsPrimitive);
                    types[i] = new VectorType(srcType.AsPrimitive, size: 0);
                }
                return types;
            }

            public void Save(ModelSaveContext ctx)
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

            private int[] GetColumnIds(ISchema schema, string[] names, Func<string, Exception> except)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertValue(names);

                var ids = new int[names.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    int col;
                    if (!schema.TryGetColumnIndex(names[i], out col))
                        throw except(string.Format("Could not find column '{0}'", names[i]));

                    var colType = schema.GetColumnType(col);
                    if (!colType.IsPrimitive)
                        throw except(string.Format("Column '{0}' has type '{1}', but must have a primitive type", names[i], colType));

                    ids[i] = col;
                }

                return ids;
            }

            public int ColumnCount { get { return _groupCount + KeepIds.Length; } }

            public void CheckColumnInRange(int col)
            {
                _ectx.Check(0 <= col && col < _groupCount + KeepIds.Length);
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                return _columnNameMap.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                CheckColumnInRange(col);
                if (col < _groupCount)
                    return _groupColumns[col];
                return _keepColumns[col - _groupCount];
            }

            public ColumnType GetColumnType(int col)
            {
                CheckColumnInRange(col);
                if (col < _groupCount)
                    return _input.GetColumnType(GroupIds[col]);
                return _columnTypes[col - _groupCount];
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                CheckColumnInRange(col);
                if (col < _groupCount)
                    return _input.GetMetadataTypes(GroupIds[col]);

                col -= _groupCount;
                var result = new List<KeyValuePair<string, ColumnType>>();
                foreach (var kind in _preservedMetadata)
                {
                    var colType = _input.GetMetadataTypeOrNull(kind, KeepIds[col]);
                    if (colType != null)
                        result.Add(colType.GetPair(kind));
                }

                return result;
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                CheckColumnInRange(col);
                if (col < _groupCount)
                    return _input.GetMetadataTypeOrNull(kind, GroupIds[col]);

                col -= _groupCount;
                if (_preservedMetadata.Contains(kind))
                    return _input.GetMetadataTypeOrNull(kind, KeepIds[col]);
                return null;
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                CheckColumnInRange(col);
                if (col < _groupCount)
                {
                    _input.GetMetadata(kind, GroupIds[col], ref value);
                    return;
                }

                col -= _groupCount;
                if (_preservedMetadata.Contains(kind))
                {
                    _input.GetMetadata(kind, KeepIds[col], ref value);
                    return;
                }

                throw _ectx.ExceptGetMetadata();
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
        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            /// <summary>
            /// This class keeps track of the previous group key and tests the current group key against the previous one.
            /// </summary>
            private sealed class GroupKeyColumnChecker
            {
                public readonly Func<bool> IsSameKey;

                private static Func<bool> MakeSameChecker<T>(IRow row, int col)
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

                public GroupKeyColumnChecker(IRow row, int col)
                {
                    Contracts.AssertValue(row);
                    var type = row.Schema.GetColumnType(col);

                    Func<IRow, int, Func<bool>> del = MakeSameChecker<int>;
                    var mi = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
                    IsSameKey = (Func<bool>)mi.Invoke(null, new object[] { row, col });
                }
            }

            // REVIEW: potentially, there could be other aggregators.
            // REVIEW: Currently, it always produces dense buffers. The anticipated use cases don't include many
            // default values at the moment.
            /// <summary>
            /// This class handles the aggregation of one 'keep' column into a vector. It wraps around an <see cref="IRow"/>'s
            /// column, reads the data and aggregates.
            /// </summary>
            private abstract class KeepColumnAggregator
            {
                public abstract ValueGetter<T> GetGetter<T>(IExceptionContext ctx);
                public abstract void SetSize(int size);
                public abstract void ReadValue(int position);

                public static KeepColumnAggregator Create(IRow row, int col)
                {
                    Contracts.AssertValue(row);
                    var colType = row.Schema.GetColumnType(col);
                    Contracts.Assert(colType.IsPrimitive);

                    var type = typeof(ListAggregator<>);

                    var cons = type.MakeGenericType(colType.RawType).GetConstructor(new[] { typeof(IRow), typeof(int) });
                    return cons.Invoke(new object[] { row, col }) as KeepColumnAggregator;
                }

                private sealed class ListAggregator<TValue> : KeepColumnAggregator
                {
                    private readonly ValueGetter<TValue> _srcGetter;
                    private readonly Delegate _getter;
                    private TValue[] _buffer;
                    private int _size;

                    public ListAggregator(IRow row, int col)
                    {
                        Contracts.AssertValue(row);
                        _srcGetter = row.GetGetter<TValue>(col);
                        _getter = (ValueGetter<VBuffer<TValue>>)Getter;
                    }

                    private void Getter(ref VBuffer<TValue> dst)
                    {
                        var values = (Utils.Size(dst.Values) < _size) ? new TValue[_size] : dst.Values;
                        Array.Copy(_buffer, values, _size);
                        dst = new VBuffer<TValue>(_size, values, dst.Indices);
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

            private readonly IRowCursor _leadingCursor;
            private readonly IRowCursor _trailingCursor;

            private readonly GroupKeyColumnChecker[] _groupCheckers;
            private readonly KeepColumnAggregator[] _aggregators;

            public override long Batch { get { return 0; } }

            public Schema Schema => _parent.Schema;

            public Cursor(GroupTransform parent, Func<int, bool> predicate)
                : base(parent.Host)
            {
                Ch.AssertValue(predicate);

                _parent = parent;
                var schema = _parent._groupSchema;
                _active = Utils.BuildArray(schema.ColumnCount, predicate);
                _groupCount = schema.GroupIds.Length;

                bool[] srcActiveLeading = new bool[_parent.Source.Schema.ColumnCount];
                foreach (var col in schema.GroupIds)
                    srcActiveLeading[col] = true;
                _leadingCursor = parent.Source.GetRowCursor(x => srcActiveLeading[x]);

                bool[] srcActiveTrailing = new bool[_parent.Source.Schema.ColumnCount];
                for (int i = 0; i < _groupCount; i++)
                {
                    if (_active[i])
                        srcActiveTrailing[schema.GroupIds[i]] = true;
                }
                for (int i = 0; i < schema.KeepIds.Length; i++)
                {
                    if (_active[i + _groupCount])
                        srcActiveTrailing[schema.KeepIds[i]] = true;
                }
                _trailingCursor = parent.Source.GetRowCursor(x => srcActiveTrailing[x]);

                _groupCheckers = new GroupKeyColumnChecker[_groupCount];
                for (int i = 0; i < _groupCount; i++)
                    _groupCheckers[i] = new GroupKeyColumnChecker(_leadingCursor, _parent._groupSchema.GroupIds[i]);

                _aggregators = new KeepColumnAggregator[_parent._groupSchema.KeepIds.Length];
                for (int i = 0; i < _aggregators.Length; i++)
                {
                    if (_active[i + _groupCount])
                        _aggregators[i] = KeepColumnAggregator.Create(_trailingCursor, _parent._groupSchema.KeepIds[i]);
                }
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return _trailingCursor.GetIdGetter();
            }

            public bool IsColumnActive(int col)
            {
                _parent._groupSchema.CheckColumnInRange(col);
                return _active[col];
            }

            protected override bool MoveNextCore()
            {
                // If leading cursor is not started, start it.
                if (_leadingCursor.State == CursorState.NotStarted)
                {
                    _leadingCursor.MoveNext();
                }

                if (_leadingCursor.State == CursorState.Done)
                {
                    // Leading cursor reached the end of the input on the previous MoveNext.
                    return false;
                }

                // Then, advance the leading cursor until it hits the end of the group (or the end of the data).
                int groupSize = 0;
                while (_leadingCursor.State == CursorState.Good && IsSameGroup())
                {
                    groupSize++;
                    _leadingCursor.MoveNext();
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

            public override void Dispose()
            {
                _leadingCursor.Dispose();
                _trailingCursor.Dispose();
                base.Dispose();
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                _parent._groupSchema.CheckColumnInRange(col);
                if (!_active[col])
                    throw Ch.ExceptParam(nameof(col), "Column #{0} is not active", col);

                if (col < _groupCount)
                    return _trailingCursor.GetGetter<TValue>(_parent._groupSchema.GroupIds[col]);

                Ch.AssertValue(_aggregators[col - _groupCount]);
                return _aggregators[col - _groupCount].GetGetter<TValue>(Ch);
            }
        }
    }

    public static partial class GroupingOperations
    {
        [TlcModule.EntryPoint(Name = "Transforms.CombinerByContiguousGroupId",
            Desc = GroupTransform.Summary,
            UserName = GroupTransform.UserName,
            ShortName = GroupTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""Group""]/*' />" })]
        public static CommonOutputs.TransformOutput Group(IHostEnvironment env, GroupTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GroupTransform", input);
            var view = new GroupTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
