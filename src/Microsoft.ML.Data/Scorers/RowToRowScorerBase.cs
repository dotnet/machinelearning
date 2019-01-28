// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Model;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for scoring rows independently. This assumes that all columns produced by the
    /// underlying <see cref="ISchemaBoundRowMapper"/> should be exposed, as well as zero or more
    /// "derived" columns.
    /// </summary>
    public abstract class RowToRowScorerBase : RowToRowMapperTransformBase, IDataScorerTransform
    {
        [BestFriend]
        private protected abstract class BindingsBase : ScorerBindingsBase
        {
            public readonly ISchemaBoundRowMapper RowMapper;

            protected BindingsBase(Schema schema, ISchemaBoundRowMapper mapper, string suffix, bool user, params string[] namesDerived)
                : base(schema, mapper, suffix, user, namesDerived)
            {
                RowMapper = mapper;
            }
        }

        [BestFriend]
        private protected readonly ISchemaBindableMapper Bindable;

        [BestFriend]
        private protected RowToRowScorerBase(IHostEnvironment env, IDataView input, string registrationName, ISchemaBindableMapper bindable)
            : base(env, registrationName, input)
        {
            Contracts.AssertValue(bindable);
            Bindable = bindable;
        }

        [BestFriend]
        private protected RowToRowScorerBase(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            ctx.LoadModel<ISchemaBindableMapper, SignatureLoadModel>(host, out Bindable, "SchemaBindableMapper");
        }

        public sealed override void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SaveModel(Bindable, "SchemaBindableMapper");
            SaveCore(ctx);
        }

        /// <summary>
        /// The main save method handles saving the _bindable. This should do everything else.
        /// </summary>
        [BestFriend]
        private protected abstract void SaveCore(ModelSaveContext ctx);

        /// <summary>
        /// For the ITransformTemplate implementation.
        /// </summary>
        public abstract IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource);

        /// <summary>
        /// Derived classes provide the specific bindings object.
        /// </summary>
        [BestFriend]
        private protected abstract BindingsBase GetBindings();

        /// <summary>
        /// Produces the set of active columns for the scorer (as a bool[] of length bindings.ColumnCount),
        /// a predicate for the needed active input columns, and a predicate for the needed active
        /// mapper columns.
        /// </summary>
        private static bool[] GetActive(BindingsBase bindings, Func<int, bool> predicate,
            out Func<int, bool> predicateInput, out Func<int, bool> predicateMapper)
        {
            var active = bindings.GetActive(predicate);
            Contracts.Assert(active.Length == bindings.ColumnCount);

            var activeInput = bindings.GetActiveInput(predicate);
            Contracts.Assert(activeInput.Length == bindings.Input.Count);

            // Get a predicate that determines which Mapper outputs are active.
            predicateMapper = bindings.GetActiveMapperColumns(active);

            // Now map those to active input columns.
            var predicateInputForMapper = bindings.RowMapper.GetDependencies(predicateMapper);

            // Combine the two sets of input columns.
            predicateInput =
                col => 0 <= col && col < activeInput.Length && (activeInput[col] || predicateInputForMapper(col));

            return active;
        }

        /// <summary>
        /// This produces either "true" or "null" according to whether <see cref="WantParallelCursors"/>
        /// returns true or false. Note that this will never return false. Any derived class
        /// must support (but not necessarily prefer) parallel cursors.
        /// </summary>
        protected sealed override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            if (WantParallelCursors(predicate))
                return true;
            return null;
        }

        /// <summary>
        /// This should return true iff parallel cursors are advantageous. Typically, this
        /// will return true iff some columns added by this scorer are active.
        /// </summary>
        protected abstract bool WantParallelCursors(Func<int, bool> predicate);

        protected override RowCursor GetRowCursorCore(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            Contracts.AssertValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            var bindings = GetBindings();
            Func<int, bool> inputPred;
            Func<int, bool> predicateMapper;
            var active = GetActive(bindings, predicate, out inputPred, out predicateMapper);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var input = Source.GetRowCursor(inputCols, rand);
            return new Cursor(Host, this, input, active, predicateMapper);
        }

        public override RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            var bindings = GetBindings();
            Func<int, bool> inputPred;
            Func<int, bool> predicateMapper;
            var active = GetActive(bindings, predicate, out inputPred, out predicateMapper);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var inputs = Source.GetRowCursorSet(inputCols, n, rand);
            Contracts.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && WantParallelCursors(predicate) && (Source.GetRowCount() ?? int.MaxValue) > n)
                inputs = DataViewUtils.CreateSplitCursors(Host, inputs[0], n);
            Contracts.AssertNonEmpty(inputs);

            var cursors = new RowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(Host, this, inputs[i], active, predicateMapper);
            return cursors;
        }

        protected override Delegate[] CreateGetters(Row input, Func<int, bool> active, out Action disp)
        {
            var bindings = GetBindings();
            Func<int, bool> predicateInput;
            Func<int, bool> predicateMapper;
            GetActive(bindings, active, out predicateInput, out predicateMapper);
            var output = bindings.RowMapper.GetRow(input, predicateMapper);
            Func<int, bool> activeInfos = iinfo => active(bindings.MapIinfoToCol(iinfo));
            disp = output.Dispose;
            return GetGetters(output, activeInfos);
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            var bindings = GetBindings();
            Func<int, bool> predicateInput;
            Func<int, bool> predicateMapper;
            GetActive(bindings, predicate, out predicateInput, out predicateMapper);
            return predicateInput;
        }

        /// <summary>
        /// Create and fill an array of getters of size InfoCount. The indices of the non-null entries in the
        /// result should be exactly those for which predicate(iinfo) is true.
        /// </summary>
        protected abstract Delegate[] GetGetters(Row output, Func<int, bool> predicate);

        protected static Delegate[] GetGettersFromRow(Row row, Func<int, bool> predicate)
        {
            Contracts.AssertValue(row);
            Contracts.AssertValue(predicate);

            var getters = new Delegate[row.Schema.Count];
            for (int col = 0; col < getters.Length; col++)
            {
                if (predicate(col))
                    getters[col] = GetGetterFromRow(row, col);
            }
            return getters;
        }

        protected static Delegate GetGetterFromRow(Row row, int col)
        {
            Contracts.AssertValue(row);
            Contracts.Assert(0 <= col && col < row.Schema.Count);
            Contracts.Assert(row.IsColumnActive(col));

            var type = row.Schema[col].Type;
            Func<Row, int, ValueGetter<int>> del = GetGetterFromRow<int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
            return (Delegate)meth.Invoke(null, new object[] { row, col });
        }

        protected static ValueGetter<T> GetGetterFromRow<T>(Row output, int col)
        {
            Contracts.AssertValue(output);
            Contracts.Assert(0 <= col && col < output.Schema.Count);
            Contracts.Assert(output.IsColumnActive(col));
            return output.GetGetter<T>(col);
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            var bindings = GetBindings();
            return bindings.MapColumnIndex(out isSrc, col);
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly BindingsBase _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;
            private readonly Row _output;
            private bool _disposed;

            public override Schema Schema { get; }

            public Cursor(IChannelProvider provider, RowToRowScorerBase parent, RowCursor input, bool[] active, Func<int, bool> predicateMapper)
                : base(provider, input)
            {
                Ch.AssertValue(parent);
                Ch.AssertValue(active);
                Ch.AssertValue(predicateMapper);

                _bindings = parent.GetBindings();
                Schema = parent.OutputSchema;
                Ch.Assert(active.Length == _bindings.ColumnCount);
                _active = active;

                _output = _bindings.RowMapper.GetRow(input, predicateMapper);
                try
                {
                    Ch.Assert(_output.Schema == _bindings.RowMapper.OutputSchema);
                    _getters = parent.GetGetters(_output, iinfo => active[_bindings.MapIinfoToCol(iinfo)]);
                }
                catch (Exception)
                {
                    _output.Dispose();
                    throw;
                }
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                    _output.Dispose();
                _disposed = true;
                base.Dispose(disposing);
            }

            public override bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active[col];
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.AssertValue(_getters);
                var getter = _getters[index];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }
        }
    }

    public abstract class ScorerArgumentsBase
    {
        // Output columns.

        [Argument(ArgumentType.AtMostOnce, HelpText = "Output column: The suffix to append to the default column names", ShortName = "ex")]
        public string Suffix;
    }

    /// <summary>
    /// Base bindings for a scorer based on an <see cref="ISchemaBoundMapper"/>. This assumes that input schema columns
    /// are echoed, followed by zero or more derived columns, followed by the mapper generated columns.
    /// The names of the derived columns and mapper generated columns have an optional suffix appended.
    /// </summary>
    [BestFriend]
    internal abstract class ScorerBindingsBase : ColumnBindingsBase
    {
        /// <summary>
        /// The schema bound mapper.
        /// </summary>
        public readonly ISchemaBoundMapper Mapper;

        /// <summary>
        /// The column name suffix. Non-null, but may be empty.
        /// </summary>
        public readonly string Suffix;

        /// <summary>
        /// The number of derived columns. InfoCount == DerivedColumnCount + Mapper.OutputSchema.ColumnCount.
        /// </summary>
        public readonly int DerivedColumnCount;

        private readonly uint _crtScoreSet;
        private readonly MetadataUtils.MetadataGetter<uint> _getScoreColumnSetId;

        protected ScorerBindingsBase(Schema input, ISchemaBoundMapper mapper, string suffix, bool user, params string[] namesDerived)
            : base(input, user, GetOutputNames(mapper, suffix, namesDerived))
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValueOrNull(suffix);
            Contracts.AssertValue(namesDerived);

            Mapper = mapper;
            DerivedColumnCount = namesDerived.Length;
            Suffix = suffix ?? "";

            int c;
            var max = input.GetMaxMetadataKind(out c, MetadataUtils.Kinds.ScoreColumnSetId);
            _crtScoreSet = checked(max + 1);
            _getScoreColumnSetId = GetScoreColumnSetId;
        }

        private static string[] GetOutputNames(ISchemaBoundMapper mapper, string suffix, string[] namesDerived)
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValueOrNull(suffix);
            Contracts.AssertValue(namesDerived);

            var schema = mapper.OutputSchema;
            int count = namesDerived.Length + schema.Count;
            var res = new string[count];
            int dst = 0;
            for (int i = 0; i < namesDerived.Length; i++)
                res[dst++] = namesDerived[i] + suffix;
            for (int i = 0; i < schema.Count; i++)
                res[dst++] = schema[i].Name + suffix;
            Contracts.Assert(dst == count);
            return res;
        }

        protected static KeyValuePair<RoleMappedSchema.ColumnRole, string>[] LoadBaseInfo(
            ModelLoadContext ctx, out string suffix)
        {
            // *** Binary format ***
            // int: id of the suffix
            // int: the number of input column roles
            // for each input column:
            //   int: id of the column role
            //   int: id of the column name
            suffix = ctx.LoadString();

            var count = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(count >= 0);

            var columns = new KeyValuePair<RoleMappedSchema.ColumnRole, string>[count];
            for (int i = 0; i < count; i++)
            {
                var role = ctx.LoadNonEmptyString();
                var name = ctx.LoadNonEmptyString();
                columns[i] = RoleMappedSchema.CreatePair(role, name);
            }

            return columns;
        }

        protected void SaveBase(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: id of the suffix
            // int: the number of input column roles
            // for each input column:
            //   int: id of the column role
            //   int: id of the column name
            ctx.SaveString(Suffix);

            var cols = Mapper.GetInputColumnRoles().ToArray();
            ctx.Writer.Write(cols.Length);
            foreach (var kvp in cols)
            {
                ctx.SaveNonEmptyString(kvp.Key.Value);
                ctx.SaveNonEmptyString(kvp.Value);
            }
        }

        public abstract void Save(ModelSaveContext ctx);

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Contracts.Assert(DerivedColumnCount <= iinfo && iinfo < InfoCount);
            return Mapper.OutputSchema[iinfo - DerivedColumnCount].Type;
        }

        protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);

            yield return MetadataUtils.ScoreColumnSetIdType.GetPair(MetadataUtils.Kinds.ScoreColumnSetId);
            if (iinfo < DerivedColumnCount)
                yield break;
            foreach (var pair in Mapper.OutputSchema[iinfo - DerivedColumnCount].Metadata.Schema.Select(c => new KeyValuePair<string, ColumnType>(c.Name, c.Type)))
                yield return pair;
        }

        protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            if (kind == MetadataUtils.Kinds.ScoreColumnSetId)
                return MetadataUtils.ScoreColumnSetIdType;
            if (iinfo < DerivedColumnCount)
                return null;
            return Mapper.OutputSchema[iinfo - DerivedColumnCount].Metadata.Schema.GetColumnOrNull(kind)?.Type;
        }

        protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            switch (kind)
            {
                case MetadataUtils.Kinds.ScoreColumnSetId:
                    _getScoreColumnSetId.Marshal(iinfo, ref value);
                    break;
                default:
                    if (iinfo < DerivedColumnCount)
                        throw MetadataUtils.ExceptGetMetadata();
                    Mapper.OutputSchema[iinfo - DerivedColumnCount].Metadata.GetValue(kind, ref value);
                    break;
            }
        }

        /// <summary>
        /// Returns a predicate indicating which Mapper columns are active based on the active scorer columns.
        /// This is virtual so scorers with computed columns can do the right thing.
        /// </summary>
        public virtual Func<int, bool> GetActiveMapperColumns(bool[] active)
        {
            Contracts.AssertValue(active);
            Contracts.Assert(active.Length == ColumnCount);

            return
                col =>
                {
                    Contracts.Assert(0 <= col && col < Mapper.OutputSchema.Count);
                    return 0 <= col && col < Mapper.OutputSchema.Count &&
                        active[MapIinfoToCol(col + DerivedColumnCount)];
                };
        }

        protected void GetScoreColumnSetId(int iinfo, ref uint dst)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            dst = _crtScoreSet;
        }
    }
}
