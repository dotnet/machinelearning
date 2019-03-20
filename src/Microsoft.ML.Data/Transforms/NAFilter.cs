// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// REVIEW: As soon as we stop writing sizeof(Float), or when we retire the double builds, we can remove this.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(NAFilter.Summary, typeof(NAFilter), typeof(NAFilter.Arguments), typeof(SignatureDataTransform),
    NAFilter.FriendlyName, NAFilter.ShortName, "MissingValueFilter", "MissingFilter")]

// REVIEW: Make sure that the "MissingFeatureFilter" signature is maintained for backwards compatibility,
// and this is not a bug.
[assembly: LoadableClass(NAFilter.Summary, typeof(NAFilter), null, typeof(SignatureLoadDataTransform),
    NAFilter.FriendlyName, NAFilter.LoaderSignature, "MissingFeatureFilter")]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="NAFilter"]'/>
    [BestFriend]
    internal sealed class NAFilter : FilterBase
    {
        private static class Defaults
        {
            public const bool Complement = false;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Column", Name = "Column", ShortName = "col", SortOrder = 1)]
            public string[] Columns;

            [Argument(ArgumentType.Multiple, HelpText = "If true, keep only rows that contain NA values, and filter the rest.")]
            public bool Complement = Defaults.Complement;
        }

        private sealed class ColInfo
        {
            public readonly int Index;
            public readonly DataViewType Type;

            public ColInfo(int index, DataViewType type)
            {
                Index = index;
                Type = type;
            }
        }

        public const string Summary = "Filters out rows that contain missing values.";
        public const string FriendlyName = "NA Filter";
        public const string ShortName = "NAFilter";

        public const string LoaderSignature = "MissingValueFilter";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MISFETFL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                // This is an older name and can be removed once we don't care about old code
                // being able to load this.
                loaderSignatureAlt: "MissingFeatureFilter",
                loaderAssemblyName: typeof(NAFilter).Assembly.FullName);
        }

        private readonly ColInfo[] _infos;
        private readonly Dictionary<int, int> _srcIndexToInfoIndex;
        private readonly bool _complement;
        private const string RegistrationName = "MissingValueFilter";

        /// <summary>
        /// Initializes a new instance of <see cref="NAFilter"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="complement">If true, keep only rows that contain NA values, and filter the rest.</param>
        /// <param name="columns">Name of the columns. Only these columns will be used to filter rows having 'NA' values.</param>
        public NAFilter(IHostEnvironment env, IDataView input, bool complement = Defaults.Complement, params string[] columns)
            : this(env, new Arguments() { Columns = columns, Complement = complement }, input)
        {
        }

        public NAFilter(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckValue(input, nameof(input));
            Host.CheckUserArg(Utils.Size(args.Columns) > 0, nameof(args.Columns));
            Host.CheckValue(env, nameof(env));

            _infos = new ColInfo[args.Columns.Length];
            _srcIndexToInfoIndex = new Dictionary<int, int>(_infos.Length);
            _complement = args.Complement;
            var schema = Source.Schema;
            for (int i = 0; i < _infos.Length; i++)
            {
                string src = args.Columns[i];
                int index;
                if (!schema.TryGetColumnIndex(src, out index))
                    throw Host.ExceptUserArg(nameof(args.Columns), "Source column '{0}' not found", src);
                if (_srcIndexToInfoIndex.ContainsKey(index))
                    throw Host.ExceptUserArg(nameof(args.Columns), "Source column '{0}' specified multiple times", src);

                var type = schema[index].Type;
                if (!TestType(type))
                    throw Host.ExceptUserArg(nameof(args.Columns), $"Column '{src}' has type {type} which does not support missing values, so we cannot filter on them", src);

                _infos[i] = new ColInfo(index, type);
                _srcIndexToInfoIndex.Add(index, i);
            }
        }

        public NAFilter(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of columns
            // int[]: ids of column names
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Single) || cbFloat == sizeof(Double));
            int cinfo = ctx.Reader.ReadInt32();
            Host.CheckDecode(cinfo > 0);

            _infos = new ColInfo[cinfo];
            _srcIndexToInfoIndex = new Dictionary<int, int>(_infos.Length);
            var schema = Source.Schema;
            for (int i = 0; i < cinfo; i++)
            {
                string src = ctx.LoadNonEmptyString();
                int index;
                if (!schema.TryGetColumnIndex(src, out index))
                    throw Host.ExceptSchemaMismatch(nameof(schema), "source", src);
                if (_srcIndexToInfoIndex.ContainsKey(index))
                    throw Host.Except("Source column '{0}' specified multiple times", src);

                var type = schema[index].Type;
                if (!TestType(type))
                    throw Host.ExceptSchemaMismatch(nameof(schema), "source", src, "scalar or vector of float, double or KeyType", type.ToString());

                _infos[i] = new ColInfo(index, type);
                _srcIndexToInfoIndex.Add(index, i);
            }
        }

        public static NAFilter Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NAFilter(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of columns
            // int[]: ids of column names
            ctx.Writer.Write(sizeof(float));
            Host.Assert(_infos.Length > 0);
            ctx.Writer.Write(_infos.Length);
            foreach (var info in _infos)
                ctx.SaveNonEmptyString(Source.Schema[info.Index].Name);
        }

        private static bool TestType(DataViewType type)
        {
            Contracts.AssertValue(type);

            var itemType = (type as VectorType)?.ItemType ?? type;
            if (itemType == NumberDataViewType.Single)
                return true;
            if (itemType == NumberDataViewType.Double)
                return true;
            if (itemType is KeyType)
                return true;
            return false;
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // This transform has no preference.
            return null;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            Func<int, bool> inputPred = GetActive(predicate, out bool[] active);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var input = Source.GetRowCursor(inputCols, rand);
            return new Cursor(this, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            Func<int, bool> inputPred = GetActive(predicate, out bool[] active);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var inputs = Source.GetRowCursorSet(inputCols, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new DataViewRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(this, inputs[i], active);
            return cursors;
        }

        private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
        {
            Host.AssertValue(predicate);
            active = new bool[Source.Schema.Count];
            bool[] activeInput = new bool[Source.Schema.Count];
            for (int i = 0; i < active.Length; i++)
                activeInput[i] = active[i] = predicate(i);
            for (int i = 0; i < _infos.Length; i++)
                activeInput[_infos[i].Index] = true;
            return col => activeInput[col];
        }

        private sealed class Cursor : LinkedRowFilterCursorBase
        {
            private abstract class Value
            {
                protected readonly Cursor Cursor;

                protected Value(Cursor cursor)
                {
                    Contracts.AssertValue(cursor);
                    Cursor = cursor;
                }

                public abstract bool Refresh();

                public abstract Delegate GetGetter();

                public static Value Create(Cursor cursor, ColInfo info)
                {
                    Contracts.AssertValue(cursor);
                    Contracts.AssertValue(info);

                    MethodInfo meth;
                    if (info.Type is VectorType vecType)
                    {
                        Func<Cursor, ColInfo, ValueVec<int>> d = CreateVec<int>;
                        meth = d.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(vecType.ItemType.RawType);
                    }
                    else
                    {
                        Func<Cursor, ColInfo, ValueOne<int>> d = CreateOne<int>;
                        meth = d.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(info.Type.RawType);
                    }
                    return (Value)meth.Invoke(null, new object[] { cursor, info });
                }

                private static ValueOne<T> CreateOne<T>(Cursor cursor, ColInfo info)
                {
                    Contracts.AssertValue(cursor);
                    Contracts.AssertValue(info);
                    Contracts.Assert(!(info.Type is VectorType));
                    Contracts.Assert(info.Type.RawType == typeof(T));

                    var getSrc = cursor.Input.GetGetter<T>(cursor.Input.Schema[info.Index]);
                    var hasBad = Data.Conversion.Conversions.Instance.GetIsNAPredicate<T>(info.Type);
                    return new ValueOne<T>(cursor, getSrc, hasBad);
                }

                private static ValueVec<T> CreateVec<T>(Cursor cursor, ColInfo info)
                {
                    Contracts.AssertValue(cursor);
                    Contracts.AssertValue(info);
                    Contracts.Assert(info.Type is VectorType);
                    Contracts.Assert(info.Type.RawType == typeof(VBuffer<T>));

                    var getSrc = cursor.Input.GetGetter<VBuffer<T>>(cursor.Input.Schema[info.Index]);
                    var hasBad = Data.Conversion.Conversions.Instance.GetHasMissingPredicate<T>((VectorType)info.Type);
                    return new ValueVec<T>(cursor, getSrc, hasBad);
                }

                private abstract class TypedValue<T> : Value
                {
                    private readonly ValueGetter<T> _getSrc;
                    private readonly InPredicate<T> _hasBad;
                    public T Src;

                    protected TypedValue(Cursor cursor, ValueGetter<T> getSrc, InPredicate<T> hasBad)
                        : base(cursor)
                    {
                        Contracts.AssertValue(getSrc);
                        Contracts.AssertValue(hasBad);
                        _getSrc = getSrc;
                        _hasBad = hasBad;
                    }

                    public override bool Refresh()
                    {
                        _getSrc(ref Src);
                        return !_hasBad(in Src);
                    }
                }

                private sealed class ValueOne<T> : TypedValue<T>
                {
                    private readonly ValueGetter<T> _getter;

                    public ValueOne(Cursor cursor, ValueGetter<T> getSrc, InPredicate<T> hasBad)
                        : base(cursor, getSrc, hasBad)
                    {
                        _getter = GetValue;
                    }

                    public void GetValue(ref T dst)
                    {
                        Contracts.Check(Cursor.IsGood);
                        dst = Src;
                    }

                    public override Delegate GetGetter()
                    {
                        return _getter;
                    }
                }

                private sealed class ValueVec<T> : TypedValue<VBuffer<T>>
                {
                    private readonly ValueGetter<VBuffer<T>> _getter;

                    public ValueVec(Cursor cursor, ValueGetter<VBuffer<T>> getSrc, InPredicate<VBuffer<T>> hasBad)
                        : base(cursor, getSrc, hasBad)
                    {
                        _getter = GetValue;
                    }

                    public void GetValue(ref VBuffer<T> dst)
                    {
                        Contracts.Check(Cursor.IsGood);
                        Src.CopyTo(ref dst);
                    }

                    public override Delegate GetGetter()
                    {
                        return _getter;
                    }
                }
            }

            private readonly NAFilter _parent;
            private readonly Value[] _values;

            public Cursor(NAFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent.Host, input, parent.OutputSchema, active)
            {
                _parent = parent;
                _values = new Value[_parent._infos.Length];
                for (int i = 0; i < _parent._infos.Length; i++)
                    _values[i] = Value.Create(this, _parent._infos[i]);
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.Check(IsColumnActive(column));

                ValueGetter<TValue> fn;
                if (TryGetColumnValueGetter(column.Index, out fn))
                    return fn;
                return Input.GetGetter<TValue>(column);
            }

            /// <summary>
            /// Gets the appropriate column value getter for a mapped column. If the column
            /// is not mapped, this returns false with the out parameters getting default values.
            /// If the column is mapped but the TValue is of the wrong type, an exception is
            /// thrown.
            /// </summary>
            private bool TryGetColumnValueGetter<TValue>(int col, out ValueGetter<TValue> fn)
            {
                Ch.Assert(IsColumnActive(Schema[col]));

                int index;
                if (!_parent._srcIndexToInfoIndex.TryGetValue(col, out index))
                {
                    fn = null;
                    return false;
                }

                fn = _values[index].GetGetter() as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return true;
            }

            protected override bool Accept()
            {
                for (int i = 0; i < _parent._infos.Length; i++)
                {
                    if (!_values[i].Refresh())
                        return _parent._complement;
                }
                return !_parent._complement;
            }
        }
    }
}
