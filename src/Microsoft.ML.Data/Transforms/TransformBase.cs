// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for transforms.
    /// </summary>
    public abstract class TransformBase : IDataTransform
    {
        protected readonly IHost Host;

        public IDataView Source { get; }

        protected TransformBase(IHostEnvironment env, string name, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            env.CheckValue(input, nameof(input));

            Host = env.Register(name);
            Source = input;
        }

        protected TransformBase(IHost host, IDataView input)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckValue(input, nameof(input));

            Host = host;
            Source = input;
        }

        public abstract void Save(ModelSaveContext ctx);

        public abstract long? GetRowCount(bool lazy = true);

        public virtual bool CanShuffle { get { return Source.CanShuffle; } }

        public abstract Schema Schema { get; }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var rng = CanShuffle ? rand : null;
            bool? useParallel = ShouldUseParallelCursors(predicate);

            // When useParallel is null, let the input decide, so go ahead and ask for parallel.
            // When the input wants to be split, this puts the consolidation after this transform
            // instead of before. This is likely to produce better performance, for example, when
            // this is RangeFilter.
            IRowCursor curs;
            if (useParallel != false &&
                DataViewUtils.TryCreateConsolidatingCursor(out curs, this, predicate, Host, rng))
            {
                return curs;
            }

            return GetRowCursorCore(predicate, rng);
        }

        /// <summary>
        /// This returns false when this transform cannot support parallel cursors, null when it
        /// doesn't care, and true when it benefits from parallel cursors. For example, a transform
        /// that simply affects metadata, but not column values should return null, while a transform
        /// that does a bunch of computation should return true (if legal).
        /// </summary>
        protected abstract bool? ShouldUseParallelCursors(Func<int, bool> predicate);

        /// <summary>
        /// Create a single (non-parallel) row cursor.
        /// </summary>
        protected abstract IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null);

        public abstract IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null);
    }

    /// <summary>
    /// Base class for transforms that map single input row to single output row.
    /// </summary>
    public abstract class RowToRowTransformBase : TransformBase
    {
        protected RowToRowTransformBase(IHostEnvironment env, string name, IDataView input)
            : base(env, name, input)
        {
        }

        protected RowToRowTransformBase(IHost host, IDataView input)
            : base(host, input)
        {
        }

        public sealed override long? GetRowCount(bool lazy = true) { return Source.GetRowCount(lazy); }
    }

    /// <summary>
    /// Base class for transforms that filter out rows without changing the schema.
    /// </summary>
    public abstract class FilterBase : TransformBase, ITransformCanSavePfa
    {
        protected FilterBase(IHostEnvironment env, string name, IDataView input)
            : base(env, name, input)
        {
        }

        protected FilterBase(IHost host, IDataView input)
            : base(host, input)
        {
        }

        public override long? GetRowCount(bool lazy = true) { return null; }

        public sealed override Schema Schema { get { return Source.Schema; } }

        public virtual bool CanSavePfa => true;

        public virtual void SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            // Because filters do not modify the schema, this is a no-op.
        }
    }

    public abstract class RowToRowMapperTransformBase : RowToRowTransformBase, IRowToRowMapper
    {
        protected RowToRowMapperTransformBase(IHostEnvironment env, string name, IDataView input)
            : base(env, name, input)
        {
        }

        protected RowToRowMapperTransformBase(IHost host, IDataView input)
            : base(host, input)
        {
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            return GetDependenciesCore(predicate);
        }

        protected abstract Func<int, bool> GetDependenciesCore(Func<int, bool> predicate);

        Schema IRowToRowMapper.InputSchema => Source.Schema;

        public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
        {
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(active, nameof(active));
            Host.Check(input.Schema == Source.Schema, "Schema of input row must be the same as the schema the mapper is bound to");

            disposer = null;
            using (var ch = Host.Start("GetEntireRow"))
            {
                Action disp;
                var getters = CreateGetters(input, active, out disp);
                disposer += disp;
                return new Row(input, this, Schema, getters);
            }
        }

        protected abstract Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disp);

        protected abstract int MapColumnIndex(out bool isSrc, int col);

        private sealed class Row : IRow
        {
            private readonly Schema _schema;
            private readonly IRow _input;
            private readonly Delegate[] _getters;

            private readonly RowToRowMapperTransformBase _parent;

            public long Batch { get { return _input.Batch; } }

            public long Position { get { return _input.Position; } }

            public Schema Schema { get { return _schema; } }

            public Row(IRow input, RowToRowMapperTransformBase parent, Schema schema, Delegate[] getters)
            {
                _input = input;
                _parent = parent;
                _schema = schema;
                _getters = getters;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                bool isSrc;
                int index = _parent.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.GetGetter<TValue>(index);

                Contracts.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            public bool IsColumnActive(int col)
            {
                bool isSrc;
                int index = _parent.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.IsColumnActive((index));
                return _getters[index] != null;
            }
        }
    }

    /// <summary>
    /// Base class for transforms that operate row by row with each destination column using one
    /// source column. It provides an extension mechanism to allow a destination column to depend
    /// on multiple input columns.
    /// This class provides the implementation of ISchema and IRowCursor.
    /// </summary>
    public abstract class OneToOneTransformBase : RowToRowMapperTransformBase, ITransposeDataView, ITransformCanSavePfa,
        ITransformCanSaveOnnx
    {
        /// <summary>
        /// Information about an added column - the name of the new column, the index of the
        /// source column and the type of the source column.
        /// </summary>
        public sealed class ColInfo
        {
            public readonly string Name;
            public readonly int Source;
            public readonly ColumnType TypeSrc;
            public readonly VectorType SlotTypeSrc;

            public ColInfo(string name, int colSrc, ColumnType typeSrc, VectorType slotTypeSrc)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.Assert(colSrc >= 0);
                Contracts.AssertValue(typeSrc);
                Contracts.AssertValueOrNull(slotTypeSrc);
                Contracts.Assert(slotTypeSrc == null || typeSrc.ItemType.Equals(slotTypeSrc.ItemType));

                Name = name;
                Source = colSrc;
                TypeSrc = typeSrc;
                SlotTypeSrc = slotTypeSrc;
            }
        }

        // The schema class for this transform. This delegates to the parent transform whatever
        // it can't figure out.
        private sealed class Bindings : ColumnBindingsBase, ITransposeSchema
        {
            // The parent transform.
            private readonly OneToOneTransformBase _parent;
            // The source input transform schema, or null if the input was not a transpose dataview.
            private readonly ITransposeSchema _inputTransposed;

            /// <summary>
            /// Information about each added column.
            /// </summary>
            public readonly ColInfo[] Infos;

            private const string InvalidTypeErrorFormat = "Source column '{0}' has invalid type ('{1}'): {2}.";

            private Bindings(OneToOneTransformBase parent, ColInfo[] infos,
                ISchema input, bool user, string[] names)
                : base(input, user, names)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(parent.Host);
                Contracts.Assert(Utils.Size(infos) == InfoCount);

                _parent = parent;
                _inputTransposed = _parent.InputTranspose == null ? null : _parent.InputTranspose.TransposeSchema;
                Contracts.Assert((_inputTransposed == null) == (_parent.InputTranspose == null));
                Infos = infos;
            }

            public static Bindings Create(OneToOneTransformBase parent, OneToOneColumn[] column, ISchema input,
                ITransposeSchema transInput, Func<ColumnType, string> testType)
            {
                Contracts.AssertValue(parent);
                var host = parent.Host;
                host.CheckUserArg(Utils.Size(column) > 0, nameof(column));
                host.AssertValue(input);
                host.AssertValueOrNull(transInput);
                host.AssertValueOrNull(testType);

                var names = new string[column.Length];
                var infos = new ColInfo[column.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    var item = column[i];
                    host.CheckUserArg(item.TrySanitize(), nameof(OneToOneColumn.Name), "Invalid new column name");
                    names[i] = item.Name;

                    int colSrc;
                    if (!input.TryGetColumnIndex(item.Source, out colSrc))
                        throw host.ExceptUserArg(nameof(OneToOneColumn.Source), "Source column '{0}' not found", item.Source);

                    var type = input.GetColumnType(colSrc);
                    if (testType != null)
                    {
                        string reason = testType(type);
                        if (reason != null)
                            throw host.ExceptUserArg(nameof(OneToOneColumn.Source), InvalidTypeErrorFormat, item.Source, type, reason);
                    }

                    var slotType = transInput == null ? null : transInput.GetSlotType(colSrc);
                    infos[i] = new ColInfo(names[i], colSrc, type, slotType);
                }

                return new Bindings(parent, infos, input, true, names);
            }

            public static Bindings Create(OneToOneTransformBase parent, ModelLoadContext ctx, ISchema input,
                ITransposeSchema transInput, Func<ColumnType, string> testType)
            {
                Contracts.AssertValue(parent);
                var host = parent.Host;
                host.CheckValue(ctx, nameof(ctx));
                host.AssertValue(input);
                host.AssertValueOrNull(transInput);
                host.AssertValueOrNull(testType);

                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   int: id of input column name
                int cinfo = ctx.Reader.ReadInt32();
                host.CheckDecode(cinfo > 0);

                var names = new string[cinfo];
                var infos = new ColInfo[cinfo];
                for (int i = 0; i < cinfo; i++)
                {
                    string dst = ctx.LoadNonEmptyString();
                    names[i] = dst;

                    // Note that in old files, the source name may be null indicating that
                    // the source column has the same name as the added column.
                    string tmp = ctx.LoadStringOrNull();
                    string src = tmp ?? dst;
                    host.CheckDecode(!string.IsNullOrEmpty(src));

                    int colSrc;
                    if (!input.TryGetColumnIndex(src, out colSrc))
                        throw host.Except("Source column '{0}' is required but not found", src);
                    var type = input.GetColumnType(colSrc);
                    if (testType != null)
                    {
                        string reason = testType(type);
                        if (reason != null)
                            throw host.Except(InvalidTypeErrorFormat, src, type, reason);
                    }
                    var slotType = transInput == null ? null : transInput.GetSlotType(colSrc);
                    infos[i] = new ColInfo(dst, colSrc, type, slotType);
                }

                return new Bindings(parent, infos, input, false, names);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   int: id of input column name
                ctx.Writer.Write(Infos.Length);
                foreach (var info in Infos)
                {
                    ctx.SaveNonEmptyString(info.Name);
                    ctx.SaveNonEmptyString(Input.GetColumnName(info.Source));
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.ColumnCount];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                        _parent.ActivateSourceColumns(index, active);
                }

                return col => 0 <= col && col < active.Length && active[col];
            }

            // The methods below here delegate to the parent transform.

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                return _parent.GetColumnTypeCore(iinfo);
            }

            public VectorType GetSlotType(int col)
            {
                _parent.Host.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                bool isSrc;
                int index = MapColumnIndex(out isSrc, col);
                if (isSrc)
                {
                    if (_inputTransposed != null)
                        return _inputTransposed.GetSlotType(index);
                    return null;
                }
                return _parent.GetSlotTypeCore(index);
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
            {
                return _parent.Metadata.GetMetadataTypes(iinfo);
            }

            protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
            {
                return _parent.Metadata.GetMetadataTypeOrNull(kind, iinfo);
            }

            protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                _parent.Metadata.GetMetadata<TValue>(_parent.Host, kind, iinfo, ref value);
            }
        }

        // This is used to simply communicate information from a constructor to a bindings constructor.
        private sealed class ColumnTmp : OneToOneColumn
        {
        }

        private readonly Bindings _bindings;

        // The ColInfos are exposed to sub-classes. They should be considered readonly.
        protected readonly ColInfo[] Infos;
        // The _input as a transposed data view, non-null iff _input is a transposed data view.
        protected readonly ITransposeDataView InputTranspose;
        // The InputTranspose transpose schema, null iff InputTranspose is null.
        protected ITransposeSchema InputTransposeSchema => InputTranspose?.TransposeSchema;

        public virtual bool CanSavePfa => false;

        public virtual bool CanSaveOnnx(OnnxContext ctx) => false;

        protected OneToOneTransformBase(IHostEnvironment env, string name, OneToOneColumn[] column,
            IDataView input, Func<ColumnType, string> testType)
            : base(env, name, input)
        {
            Host.CheckUserArg(Utils.Size(column) > 0, nameof(column));
            Host.CheckValueOrNull(testType);
            InputTranspose = Source as ITransposeDataView;

            _bindings = Bindings.Create(this, column, Source.Schema, InputTransposeSchema, testType);
            Infos = _bindings.Infos;
            Metadata = new MetadataDispatcher(Infos.Length);
        }

        protected OneToOneTransformBase(IHost host, OneToOneColumn[] column,
            IDataView input, Func<ColumnType, string> testType)
            : base(host, input)
        {
            Host.CheckUserArg(Utils.Size(column) > 0, nameof(column));
            Host.CheckValueOrNull(testType);
            InputTranspose = Source as ITransposeDataView;

            _bindings = Bindings.Create(this, column, Source.Schema, InputTransposeSchema, testType);
            Infos = _bindings.Infos;
            Metadata = new MetadataDispatcher(Infos.Length);
        }

        protected OneToOneTransformBase(IHost host, ModelLoadContext ctx,
            IDataView input, Func<ColumnType, string> testType)
            : base(host, input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValueOrNull(testType);
            InputTranspose = Source as ITransposeDataView;

            _bindings = Bindings.Create(this, ctx, Source.Schema, InputTransposeSchema, testType);
            Infos = _bindings.Infos;
            Metadata = new MetadataDispatcher(Infos.Length);
        }

        /// <summary>
        /// Re-applying constructor.
        /// </summary>
        protected OneToOneTransformBase(IHostEnvironment env, string name, OneToOneTransformBase transform,
            IDataView newInput, Func<ColumnType, string> checkType)
            : base(env, name, newInput)
        {
            Host.CheckValueOrNull(checkType);
            InputTranspose = Source as ITransposeDataView;

            OneToOneColumn[] map = transform.Infos
                .Select(x => new ColumnTmp
                {
                    Name = x.Name,
                    Source = transform.Source.Schema[x.Source].Name,
                })
                .ToArray();

            _bindings = Bindings.Create(this, map, newInput.Schema, InputTransposeSchema, checkType);
            Infos = _bindings.Infos;
            Metadata = new MetadataDispatcher(Infos.Length);
        }

        protected MetadataDispatcher Metadata { get; }

        protected void SaveBase(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            _bindings.Save(ctx);
        }

        public void SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(CanSavePfa);

            var toHide = new List<string>();
            var toDeclare = new List<KeyValuePair<string, JToken>>();

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
            {
                var info = Infos[iinfo];
                var srcName = Source.Schema[info.Source].Name;
                string srcToken = ctx.TokenOrNullForName(srcName);
                if (srcToken == null)
                {
                    toHide.Add(info.Name);
                    continue;
                }
                var result = SaveAsPfaCore(ctx, iinfo, info, srcToken);
                if (result == null)
                {
                    toHide.Add(info.Name);
                    continue;
                }
                toDeclare.Add(new KeyValuePair<string, JToken>(info.Name, result));
            }
            ctx.Hide(toHide.ToArray());
            ctx.DeclareVar(toDeclare.ToArray());
        }

        public void SaveAsOnnx(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(CanSaveOnnx(ctx));

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
            {
                ColInfo info = Infos[iinfo];
                string sourceColumnName = Source.Schema[info.Source].Name;
                if (!ctx.ContainsColumn(sourceColumnName))
                {
                    ctx.RemoveColumn(info.Name, false);
                    continue;
                }

                if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(sourceColumnName),
                    ctx.AddIntermediateVariable(Schema[_bindings.MapIinfoToCol(iinfo)].Type, info.Name)))
                {
                    ctx.RemoveColumn(info.Name, true);
                }
            }
        }

        /// <summary>
        /// Called by <see cref="SaveAsPfa"/>. Should be implemented by subclasses that return
        /// <c>true</c> from <see cref="CanSavePfa"/>. Will be called
        /// </summary>
        /// <param name="ctx">The context. Can be used to declare cells, access other information,
        /// and whatnot. This method should not actually, however, declare the variable corresponding
        /// to the output column. The calling method will do that.</param>
        /// <param name="iinfo">The index of the output column whose PFA is being composed</param>
        /// <param name="info">The column info</param>
        /// <param name="srcToken">The token in the PFA corresponding to the source col</param>
        /// <returns>Shuold return the declaration corresponding to the value of this column. Will
        /// return <c>null</c> in the event that we do not know how to express this column as PFA</returns>
        protected virtual JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Host.AssertValue(ctx);
            Host.Assert(0 <= iinfo && iinfo < _bindings.InfoCount);
            Host.Assert(Infos[iinfo] == info);
            Host.AssertValue(srcToken);
            Host.Assert(CanSavePfa);
            return null;
        }

        protected virtual bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName,
            string dstVariableName) => false;

        public sealed override Schema Schema => _bindings.AsSchema;

        public ITransposeSchema TransposeSchema => _bindings;

        /// <summary>
        /// Return the (destination) column index for the indicated added column.
        /// </summary>
        protected int ColumnIndex(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return _bindings.MapIinfoToCol(iinfo);
        }

        protected abstract ColumnType GetColumnTypeCore(int iinfo);

        protected virtual VectorType GetSlotTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            // By default, none of the added columns are transposable.
            return null;
        }

        /// <summary>
        /// Activates the source column.
        /// Override when you don't need the source column to generate the value for this column or when you need
        /// other auxiliary source columns that iinfo destination column depends on.
        /// </summary>
        protected virtual void ActivateSourceColumns(int iinfo, bool[] active)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            active[Infos[iinfo].Source] = true;
        }

        /// <summary>
        /// Sub-classes implement this to provide, for a cursor, a getter delegate and optional disposer.
        /// If no action is needed when the cursor is Disposed, the override should set disposer to null,
        /// otherwise it should be set to a delegate to be invoked by the cursor's Dispose method. It's best
        /// for this action to be idempotent - calling it multiple times should be equivalent to calling it once.
        /// </summary>
        protected abstract Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer);

        protected ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            int src = Infos[iinfo].Source;
            Host.Assert(input.IsColumnActive(src));
            return input.GetGetter<T>(src);
        }

        protected Delegate GetSrcGetter(ColumnType typeDst, IRow row, int iinfo)
        {
            Host.CheckValue(typeDst, nameof(typeDst));
            Host.CheckValue(row, nameof(row));

            Func<IRow, int, ValueGetter<int>> del = GetSrcGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeDst.RawType);
            return (Delegate)methodInfo.Invoke(this, new object[] { row, iinfo });
        }

        /// <summary>
        /// This produces either "true" or "null" according to whether <see cref="WantParallelCursors"/>
        /// returns true or false. Note that this will never return false. Any derived class
        /// must support (but not necessarily prefer) parallel cursors.
        /// </summary>
        protected sealed override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            if (WantParallelCursors(predicate))
                return true;
            return null;
        }

        /// <summary>
        /// This should return true iff parallel cursors are advantageous. The default implementation
        /// returns true iff some columns added by this transform are active.
        /// </summary>
        protected virtual bool WantParallelCursors(Func<int, bool> predicate)
        {
            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            return _bindings.AnyNewColumnsActive(predicate);
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(Host, this, input, active);
        }

        public sealed override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && WantParallelCursors(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, this, inputs[i], active);
            return cursors;
        }

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to <see cref="GetSlotCursor"/>,
        /// on a column that is not transposable.
        /// </summary>
        protected Exception ExceptGetSlotCursor(int col)
        {
            Host.Assert(0 <= col && col < _bindings.ColumnCount);
            return Host.ExceptParam(nameof(col), "Bad call to GetSlotCursor on untransposable column '{0}'",
                Schema[col].Name);
        }

        public ISlotCursor GetSlotCursor(int col)
        {
            Host.CheckParam(0 <= col && col < _bindings.ColumnCount, nameof(col));

            bool isSrc;
            int index = _bindings.MapColumnIndex(out isSrc, col);
            if (isSrc)
            {
                if (InputTranspose != null)
                    return InputTranspose.GetSlotCursor(index);
                throw ExceptGetSlotCursor(col);
            }
            if (GetSlotTypeCore(index) == null)
                throw ExceptGetSlotCursor(col);
            return GetSlotCursorCore(index);
        }

        /// <summary>
        /// Implementors should note this only called if <see cref="GetSlotTypeCore"/>
        /// returns a non-null value for this <paramref name="iinfo"/>, so in principle
        /// it should always return a valid value, if called. This implementation throws,
        /// since the default implementation of <see cref="GetSlotTypeCore"/> will return
        /// null for all new columns, and so reaching this is only possible if there is a
        /// bug.
        /// </summary>
        protected virtual ISlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(false);
            throw Host.ExceptNotImpl("Data view indicated it could transpose a column, but apparently it could not");
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disposer)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            var getters = new Delegate[_bindings.InfoCount];
            disposer = null;
            using (var ch = Host.Start("CreateGetters"))
            {
                for (int iinfo = 0; iinfo < _bindings.InfoCount; iinfo++)
                {
                    if (!activeInfos(iinfo))
                        continue;
                    Action disp;
                    getters[iinfo] = GetGetterCore(ch, input, iinfo, out disp);
                    disposer += disp;
                }
                return getters;
            }
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;

            private readonly Delegate[] _getters;
            private readonly Action[] _disposers;

            public RowCursor(IChannelProvider provider, OneToOneTransformBase parent, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(parent);
                Ch.Assert(active == null || active.Length == parent._bindings.ColumnCount);

                _bindings = parent._bindings;
                _active = active;
                _getters = new Delegate[parent.Infos.Length];

                // Build the delegates.
                List<Action> disposers = null;
                for (int iinfo = 0; iinfo < _getters.Length; iinfo++)
                {
                    if (!IsColumnActive(parent._bindings.MapIinfoToCol(iinfo)))
                        continue;
                    Action disposer;
                    _getters[iinfo] = parent.GetGetterCore(Ch, Input, iinfo, out disposer);
                    if (disposer != null)
                        Utils.Add(ref disposers, disposer);
                }

                if (Utils.Size(disposers) > 0)
                    _disposers = disposers.ToArray();
            }

            public override void Dispose()
            {
                if (_disposers != null)
                {
                    foreach (var act in _disposers)
                        act();
                }
                base.Dispose();
            }

            public Schema Schema => _bindings.AsSchema;

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }
        }

        protected static string TestIsText(ColumnType type)
        {
            if (type.IsText)
                return null;
            return "Expected Text type";
        }

        protected static string TestIsTextItem(ColumnType type)
        {
            if (type.ItemType.IsText)
                return null;
            return "Expected Text type";
        }

        protected static string TestIsTextVector(ColumnType type)
        {
            if (type.ItemType.IsText && type.IsVector)
                return null;
            return "Expected vector of Text type";
        }

        protected static string TestIsFloatItem(ColumnType type)
        {
            if (type.ItemType == NumberType.Float)
                return null;
            return "Expected R4 or a vector of R4";
        }

        protected static string TestIsFloatVector(ColumnType type)
        {
            if (!type.IsVector || type.ItemType != NumberType.Float)
                return "Expected Float vector";

            return null;
        }

        protected static string TestIsKnownSizeFloatVector(ColumnType type)
        {
            if (!type.IsKnownSizeVector || type.ItemType != NumberType.Float)
                return "Expected Float vector of known size";

            return null;
        }

        protected static string TestIsKey(ColumnType type)
        {
            if (type.ItemType.KeyCount > 0)
                return null;
            return "Expected Key type of known cardinality";
        }
    }
}
