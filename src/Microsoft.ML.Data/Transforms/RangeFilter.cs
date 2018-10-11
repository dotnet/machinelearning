// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(RangeFilter.Summary, typeof(RangeFilter), typeof(RangeFilter.Arguments), typeof(SignatureDataTransform),
    RangeFilter.UserName, "RangeFilter")]

[assembly: LoadableClass(RangeFilter.Summary, typeof(RangeFilter), null, typeof(SignatureLoadDataTransform),
    RangeFilter.UserName, RangeFilter.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Should we support filtering on multiple columns/vector typed columns?
    /// <summary>
    /// Filters a dataview on a column of type Single, Double or Key (contiguous).
    /// Keeps the values that are in the specified min/max range. NaNs are always filtered out.
    /// If the input is a Key type, the min/max are considered percentages of the number of values.
    /// </summary>
    public sealed class RangeFilter : FilterBase
    {
        public sealed class Arguments : TransformInputBase
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
        private readonly ColumnType _type;
        private readonly Double _min;
        private readonly Double _max;
        private readonly bool _complement;
        private readonly bool _includeMin;
        private readonly bool _includeMax;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="column">Name of the input column.</param>
        /// <param name="minimum">Minimum value (0 to 1 for key types).</param>
        /// <param name="maximum">Maximum value (0 to 1 for key types).</param>
        public RangeFilter(IHostEnvironment env, IDataView input, string column, Double? minimum = null, Double? maximum = null)
            : this(env, new Arguments() { Column = column, Min = minimum, Max = maximum }, input)
        {
        }

        public RangeFilter(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));

            var schema = Source.Schema;
            if (!schema.TryGetColumnIndex(args.Column, out _index))
                throw Host.ExceptUserArg(nameof(args.Column), "Source column '{0}' not found", args.Column);

            using (var ch = Host.Start("Checking parameters"))
            {
                _type = schema.GetColumnType(_index);
                if (!IsValidRangeFilterColumnType(ch, _type))
                    throw ch.ExceptUserArg(nameof(args.Column), "Column '{0}' does not have compatible type", args.Column);
                if (_type.IsKey)
                {
                    if (args.Min < 0)
                    {
                        ch.Warning("specified min less than zero, will be ignored");
                        args.Min = null;
                    }
                    if (args.Max > 1)
                    {
                        ch.Warning("specified max greater than one, will be ignored");
                        args.Max = null;
                    }
                }
                if (args.Min == null && args.Max == null)
                    throw ch.ExceptUserArg(nameof(args.Min), "At least one of min and max must be specified.");
                _min = args.Min ?? Double.NegativeInfinity;
                _max = args.Max ?? Double.PositiveInfinity;
                if (!(_min <= _max))
                    throw ch.ExceptUserArg(nameof(args.Min), "min must be less than or equal to max");
                _complement = args.Complement;
                _includeMin = args.IncludeMin;
                _includeMax = args.IncludeMax ?? (args.Max == null || (_type.IsKey && _max >= 1));
                ch.Done();
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
            Host.CheckDecode(cbFloat == sizeof(Float));

            var column = ctx.LoadNonEmptyString();
            var schema = Source.Schema;
            if (!schema.TryGetColumnIndex(column, out _index))
                throw Host.Except("column", "Source column '{0}' not found", column);

            _type = schema.GetColumnType(_index);
            if (_type != NumberType.R4 && _type != NumberType.R8 && _type.KeyCount == 0)
                throw Host.Except("column", "Column '{0}' does not have compatible type", column);

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

        public override void Save(ModelSaveContext ctx)
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
            ctx.Writer.Write(sizeof(Float));
            ctx.SaveNonEmptyString(Source.Schema.GetColumnName(_index));
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

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            bool[] active;
            Func<int, bool> inputPred = GetActive(predicate, out active);
            var input = Source.GetRowCursor(inputPred, rand);
            return CreateCursorCore(input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            bool[] active;
            Func<int, bool> inputPred = GetActive(predicate, out active);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = CreateCursorCore(inputs[i], active);
            return cursors;
        }

        private IRowCursor CreateCursorCore(IRowCursor input, bool[] active)
        {
            if (_type == NumberType.R4)
                return new SingleRowCursor(this, input, active);
            if (_type == NumberType.R8)
                return new DoubleRowCursor(this, input, active);
            Host.Assert(_type.IsKey);
            return RowCursorBase.CreateKeyRowCursor(this, input, active);
        }

        private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
        {
            Host.AssertValue(predicate);
            active = new bool[Source.Schema.ColumnCount];
            bool[] activeInput = new bool[Source.Schema.ColumnCount];
            for (int i = 0; i < active.Length; i++)
                activeInput[i] = active[i] = predicate(i);
            activeInput[_index] = true;
            return col => activeInput[col];
        }

        public static bool IsValidRangeFilterColumnType(IExceptionContext ectx, ColumnType type)
        {
            ectx.CheckValue(type, nameof(type));

            return type == NumberType.R4 || type == NumberType.R8 || type.KeyCount > 0;
        }

        private abstract class RowCursorBase : LinkedRowFilterCursorBase
        {
            protected readonly RangeFilter Parent;
            protected readonly Func<Double, bool> CheckBounds;
            private readonly Double _min;
            private readonly Double _max;

            protected RowCursorBase(RangeFilter parent, IRowCursor input, bool[] active)
                : base(parent.Host, input, parent.Schema, active)
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

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(0 <= col && col < Schema.ColumnCount);
                Ch.Check(IsColumnActive(col));

                if (col != Parent._index)
                    return Input.GetGetter<TValue>(col);
                var fn = GetGetter() as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));

                return fn;
            }

            public static IRowCursor CreateKeyRowCursor(RangeFilter filter, IRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type.IsKey);
                Func<RangeFilter, IRowCursor, bool[], IRowCursor> del = CreateKeyRowCursor<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(filter._type.RawType);
                return (IRowCursor)methodInfo.Invoke(null, new object[] { filter, input, active });
            }

            private static IRowCursor CreateKeyRowCursor<TSrc>(RangeFilter filter, IRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type.IsKey);
                return new KeyRowCursor<TSrc>(filter, input, active);
            }
        }

        private sealed class SingleRowCursor : RowCursorBase
        {
            private readonly ValueGetter<Single> _srcGetter;
            private readonly ValueGetter<Single> _getter;
            private Single _value;

            public SingleRowCursor(RangeFilter parent, IRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type == NumberType.R4);
                _srcGetter = Input.GetGetter<Single>(Parent._index);
                _getter =
                    (ref Single value) =>
                    {
                        Ch.Check(IsGood);
                        value = _value;
                    };
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type == NumberType.R4);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type == NumberType.R4);
                _srcGetter(ref _value);
                return CheckBounds(_value);
            }
        }

        private sealed class DoubleRowCursor : RowCursorBase
        {
            private readonly ValueGetter<Double> _srcGetter;
            private readonly ValueGetter<Double> _getter;
            private Double _value;

            public DoubleRowCursor(RangeFilter parent, IRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type == NumberType.R8);
                _srcGetter = Input.GetGetter<Double>(Parent._index);
                _getter =
                    (ref Double value) =>
                    {
                        Ch.Check(IsGood);
                        value = _value;
                    };
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type == NumberType.R8);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type == NumberType.R8);
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
            private readonly int _count;

            public KeyRowCursor(RangeFilter parent, IRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type.KeyCount > 0);
                _count = Parent._type.KeyCount;
                _srcGetter = Input.GetGetter<T>(Parent._index);
                _getter =
                    (ref T dst) =>
                    {
                        Ch.Check(IsGood);
                        dst = _value;
                    };
                bool identity;
                _conv = Conversions.Instance.GetStandardConversion<T, ulong>(Parent._type, NumberType.U8, out identity);
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type.IsKey);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type.IsKey);
                _srcGetter(ref _value);
                ulong value = 0;
                _conv(ref _value, ref value);
                if (value == 0 || value > (ulong)_count)
                    return false;
                if (!CheckBounds(((Double)(uint)value - 0.5) / _count))
                    return false;
                return true;
            }
        }
    }
}
