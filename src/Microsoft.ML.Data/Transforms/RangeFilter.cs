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
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(RangeFilter.Summary, typeof(RangeFilter), typeof(RangeFilter.Options), typeof(SignatureDataTransform),
    RangeFilter.UserName, "RangeFilter")]

[assembly: LoadableClass(RangeFilter.Summary, typeof(RangeFilter), null, typeof(SignatureLoadDataTransform),
    RangeFilter.UserName, RangeFilter.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    // REVIEW: Should we support filtering on multiple columns/vector typed columns?
    /// <summary>
    /// Filters a dataview on a column of type Single, Double or Key (contiguous).
    /// Keeps the values that are in the specified min/max range. NaNs are always filtered out.
    /// If the input is a Key type, the min/max are considered percentages of the number of values.
    /// </summary>
    [BestFriend]
    internal sealed class RangeFilter : FilterBase
    {
        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Column", ShortName = "col", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Column;

            [Argument(ArgumentType.Multiple, HelpText = "Minimum value (0 to 1 for key types)")]
            public Double? Min;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum value (0 to 1 for key types)")]
            public Double? Max;

            [Argument(ArgumentType.Multiple, HelpText = "If true, keep the values that fall outside the range.")]
            public bool Complement;

            [Argument(ArgumentType.Multiple, HelpText = "If true, include in the range the values that are equal to min.")]
            public bool IncludeMin = true;

            [Argument(ArgumentType.Multiple, HelpText = "If true, include in the range the values that are equal to max.")]
            public bool? IncludeMax;
        }

        public const string Summary = "Filters a dataview on a column of type Single, Double or Key (contiguous). Keeps the values that are in the specified min/max range. "
            + "NaNs are always filtered out. If the input is a Key type, the min/max are considered percentages of the number of values.";

        public const string LoaderSignature = "RangeFilter";
        public const string UserName = "Range Filter";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RNGFILTR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RangeFilter).Assembly.FullName);
        }

        private const string RegistrationName = "RangeFilter";

        private readonly int _index;
        private readonly DataViewType _type;
        private readonly Double _min;
        private readonly Double _max;
        private readonly bool _complement;
        private readonly bool _includeMin;
        private readonly bool _includeMax;

        /// <summary>
        /// Initializes a new instance of <see cref="RangeFilter"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="column">Name of the input column.</param>
        /// <param name="lowerBound">Minimum value (0 to 1 for key types).</param>
        /// <param name="upperBound">Maximum value (0 to 1 for key types).</param>
        /// <param name="includeUpperBound">Whether to include the upper bound.</param>
        public RangeFilter(IHostEnvironment env, IDataView input, string column, Double lowerBound, Double upperBound, bool includeUpperBound)
            : this(env, new Options() { Column = column, Min = lowerBound, Max = upperBound, IncludeMax = includeUpperBound }, input)
        {
        }

        public RangeFilter(IHostEnvironment env, Options options, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(options, nameof(options));

            var schema = Source.Schema;
            if (!schema.TryGetColumnIndex(options.Column, out _index))
                throw Host.ExceptUserArg(nameof(options.Column), "Source column '{0}' not found", options.Column);

            using (var ch = Host.Start("Checking parameters"))
            {
                _type = schema[_index].Type;
                if (!IsValidRangeFilterColumnType(ch, _type))
                    throw ch.ExceptUserArg(nameof(options.Column), "Column '{0}' does not have compatible type", options.Column);
                if (_type is KeyType)
                {
                    if (options.Min < 0)
                    {
                        ch.Warning("specified min less than zero, will be ignored");
                        options.Min = null;
                    }
                    if (options.Max > 1)
                    {
                        ch.Warning("specified max greater than one, will be ignored");
                        options.Max = null;
                    }
                }
                if (options.Min == null && options.Max == null)
                    throw ch.ExceptUserArg(nameof(options.Min), "At least one of min and max must be specified.");
                _min = options.Min ?? Double.NegativeInfinity;
                _max = options.Max ?? Double.PositiveInfinity;
                if (!(_min <= _max))
                    throw ch.ExceptUserArg(nameof(options.Min), "min must be less than or equal to max");
                _complement = options.Complement;
                _includeMin = options.IncludeMin;
                _includeMax = options.IncludeMax ?? (options.Max == null || (_type is KeyType && _max >= 1));
            }
        }

        private RangeFilter(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // int: id of column name
            // double: min
            // double: max
            // byte: complement
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(float));

            var column = ctx.LoadNonEmptyString();
            var schema = Source.Schema;
            if (!schema.TryGetColumnIndex(column, out _index))
                throw Host.ExceptSchemaMismatch(nameof(schema), "source", column);

            _type = schema[_index].Type;
            if (_type != NumberDataViewType.Single && _type != NumberDataViewType.Double && _type.GetKeyCount() == 0)
                throw Host.ExceptSchemaMismatch(nameof(schema), "source", column, "float, double or KeyType", _type.ToString());

            _min = ctx.Reader.ReadDouble();
            _max = ctx.Reader.ReadDouble();
            if (!(_min <= _max))
                throw Host.Except("min", "min must be less than or equal to max");
            _complement = ctx.Reader.ReadBoolByte();
            _includeMin = ctx.Reader.ReadBoolByte();
            _includeMax = ctx.Reader.ReadBoolByte();
        }

        public static RangeFilter Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new RangeFilter(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // int: id of column name
            // double: min
            // double: max
            // byte: complement
            // byte: includeMin
            // byte: includeMax
            ctx.Writer.Write(sizeof(float));
            ctx.SaveNonEmptyString(Source.Schema[_index].Name);
            Host.Assert(_min < _max);
            ctx.Writer.Write(_min);
            ctx.Writer.Write(_max);
            ctx.Writer.WriteBoolByte(_complement);
            ctx.Writer.WriteBoolByte(_includeMin);
            ctx.Writer.WriteBoolByte(_includeMax);
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
            return CreateCursorCore(input, active);
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
                cursors[i] = CreateCursorCore(inputs[i], active);
            return cursors;
        }

        private DataViewRowCursor CreateCursorCore(DataViewRowCursor input, bool[] active)
        {
            if (_type == NumberDataViewType.Single)
                return new SingleRowCursor(this, input, active);
            if (_type == NumberDataViewType.Double)
                return new DoubleRowCursor(this, input, active);
            Host.Assert(_type is KeyType);
            return RowCursorBase.CreateKeyRowCursor(this, input, active);
        }

        private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
        {
            Host.AssertValue(predicate);
            active = new bool[Source.Schema.Count];
            bool[] activeInput = new bool[Source.Schema.Count];
            for (int i = 0; i < active.Length; i++)
                activeInput[i] = active[i] = predicate(i);
            activeInput[_index] = true;
            return col => activeInput[col];
        }

        public static bool IsValidRangeFilterColumnType(IExceptionContext ectx, DataViewType type)
        {
            ectx.CheckValue(type, nameof(type));

            return type == NumberDataViewType.Single || type == NumberDataViewType.Double || type.GetKeyCount() > 0;
        }

        private abstract class RowCursorBase : LinkedRowFilterCursorBase
        {
            protected readonly RangeFilter Parent;
            protected readonly Func<Double, bool> CheckBounds;
            private readonly Double _min;
            private readonly Double _max;

            protected RowCursorBase(RangeFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent.Host, input, parent.OutputSchema, active)
            {
                Parent = parent;
                _min = Parent._min;
                _max = Parent._max;
                if (Parent._includeMin)
                {
                    if (Parent._includeMax)
                        CheckBounds = Parent._complement ? (Func<Double, bool>)TestNotCC : TestCC;
                    else
                        CheckBounds = Parent._complement ? (Func<Double, bool>)TestNotCO : TestCO;
                }
                else
                {
                    if (Parent._includeMax)
                        CheckBounds = Parent._complement ? (Func<Double, bool>)TestNotOC : TestOC;
                    else
                        CheckBounds = Parent._complement ? (Func<Double, bool>)TestNotOO : TestOO;
                }
            }

            // The following methods test if the value is in range, out of range including or excluding the range limits
            // O - Open
            // C - Close
            // N - Not in range
            private bool TestOO(Double value) => _min < value && value < _max;
            private bool TestCO(Double value) => _min <= value && value < _max;
            private bool TestOC(Double value) => _min < value && value <= _max;
            private bool TestCC(Double value) => _min <= value && value <= _max;
            private bool TestNotOO(Double value) => _min >= value || value >= _max;
            private bool TestNotCO(Double value) => _min > value || value >= _max;
            private bool TestNotOC(Double value) => _min >= value || value > _max;
            private bool TestNotCC(Double value) => _min > value || value > _max;

            protected abstract Delegate GetGetter();
            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.Check(0 <= column.Index && column.Index < Schema.Count);
                Ch.Check(IsColumnActive(column));

                if (column.Index != Parent._index)
                    return Input.GetGetter<TValue>(column);
                var fn = GetGetter() as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));

                return fn;
            }

            public static DataViewRowCursor CreateKeyRowCursor(RangeFilter filter, DataViewRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type is KeyType);
                Func<RangeFilter, DataViewRowCursor, bool[], DataViewRowCursor> del = CreateKeyRowCursor<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(filter._type.RawType);
                return (DataViewRowCursor)methodInfo.Invoke(null, new object[] { filter, input, active });
            }

            private static DataViewRowCursor CreateKeyRowCursor<TSrc>(RangeFilter filter, DataViewRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type is KeyType);
                return new KeyRowCursor<TSrc>(filter, input, active);
            }
        }

        private sealed class SingleRowCursor : RowCursorBase
        {
            private readonly ValueGetter<Single> _srcGetter;
            private readonly ValueGetter<Single> _getter;
            private Single _value;

            public SingleRowCursor(RangeFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type == NumberDataViewType.Single);
                _srcGetter = Input.GetGetter<Single>(Input.Schema[Parent._index]);
                _getter =
                    (ref Single value) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        value = _value;
                    };
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type == NumberDataViewType.Single);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type == NumberDataViewType.Single);
                _srcGetter(ref _value);
                return CheckBounds(_value);
            }
        }

        private sealed class DoubleRowCursor : RowCursorBase
        {
            private readonly ValueGetter<Double> _srcGetter;
            private readonly ValueGetter<Double> _getter;
            private Double _value;

            public DoubleRowCursor(RangeFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type == NumberDataViewType.Double);
                _srcGetter = Input.GetGetter<Double>(Input.Schema[Parent._index]);
                _getter =
                    (ref Double value) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        value = _value;
                    };
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type == NumberDataViewType.Double);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type == NumberDataViewType.Double);
                _srcGetter(ref _value);
                return CheckBounds(_value);
            }
        }

        private sealed class KeyRowCursor<T> : RowCursorBase
        {
            private readonly ValueGetter<T> _srcGetter;
            private readonly ValueGetter<T> _getter;
            private T _value;
            private readonly ValueMapper<T, ulong> _conv;
            private readonly ulong _count;

            public KeyRowCursor(RangeFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type.GetKeyCount() > 0);
                _count = Parent._type.GetKeyCount();
                _srcGetter = Input.GetGetter<T>(Input.Schema[Parent._index]);
                _getter =
                    (ref T dst) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        dst = _value;
                    };
                bool identity;
                _conv = Data.Conversion.Conversions.Instance.GetStandardConversion<T, ulong>(Parent._type, NumberDataViewType.UInt64, out identity);
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type is KeyType);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type is KeyType);
                _srcGetter(ref _value);
                ulong value = 0;
                _conv(in _value, ref value);
                if (value == 0 || value > _count)
                    return false;
                if (!CheckBounds(((uint)value - 0.5) / _count))
                    return false;
                return true;
            }
        }
    }
}
