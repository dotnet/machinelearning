// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(GenerateNumberTransform.Summary, typeof(GenerateNumberTransform), typeof(GenerateNumberTransform.Options), typeof(SignatureDataTransform),
    GenerateNumberTransform.UserName, GenerateNumberTransform.LoadName, "GenerateNumber", GenerateNumberTransform.ShortName)]

[assembly: LoadableClass(GenerateNumberTransform.Summary, typeof(GenerateNumberTransform), null, typeof(SignatureLoadDataTransform),
    GenerateNumberTransform.UserName, GenerateNumberTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(RandomNumberGenerator))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This transform adds columns containing either random numbers distributed
    /// uniformly between 0 and 1 or an auto-incremented integer starting at zero.
    /// It will be used in conjunction with a filter transform to create random
    /// partitions of the data, used in cross validation.
    /// </summary>
    [BestFriend]
    internal sealed class GenerateNumberTransform : RowToRowTransformBase
    {
        public sealed class Column
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use an auto-incremented integer starting at zero instead of a random number", ShortName = "cnt")]
            public bool? UseCounter;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed")]
            public uint? Seed;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                int ich = str.IndexOf(':');
                if (ich < 0)
                {
                    Name = str;
                    return true;
                }

                if (0 < ich && ich < str.Length - 1)
                {
                    Name = str.Substring(0, ich);
                    uint tmp;
                    var result = uint.TryParse(str.Substring(ich + 1), out tmp);
                    if (result)
                        Seed = tmp;
                    return result;
                }

                return false;
            }
        }

        private static class Defaults
        {
            public const bool UseCounter = false;
            public const uint Seed = 42;
        }

        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:seed)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use an auto-incremented integer starting at zero instead of a random number", ShortName = "cnt")]
            public bool UseCounter = Defaults.UseCounter;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed")]
            public uint Seed = Defaults.Seed;
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            public readonly bool[] UseCounter;
            public readonly TauswortheHybrid.State[] States;

            private Bindings(bool[] useCounter, TauswortheHybrid.State[] states,
                DataViewSchema input, bool user, string[] names)
                : base(input, user, names)
            {
                Contracts.Assert(Utils.Size(useCounter) == InfoCount);
                Contracts.Assert(Utils.Size(states) == InfoCount);
                UseCounter = useCounter;
                States = states;
            }

            public static Bindings Create(Options options, DataViewSchema input)
            {
                var names = new string[options.Columns.Length];
                var useCounter = new bool[options.Columns.Length];
                var states = new TauswortheHybrid.State[options.Columns.Length];
                for (int i = 0; i < options.Columns.Length; i++)
                {
                    var item = options.Columns[i];
                    names[i] = item.Name;
                    useCounter[i] = item.UseCounter ?? options.UseCounter;
                    if (!useCounter[i])
                        states[i] = new TauswortheHybrid.State(item.Seed ?? options.Seed);
                }

                return new Bindings(useCounter, states, input, true, names);
            }

            public static Bindings Create(ModelLoadContext ctx, DataViewSchema input)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(input);

                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   byte: useCounter
                //   if !useCounter
                //     uint: seed0
                //     uint: seed1
                //     uint: seed2
                //     uint: seed3
                int size = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(size > 0);

                var names = new string[size];
                var useCounter = new bool[size];
                var states = new TauswortheHybrid.State[size];
                for (int i = 0; i < size; i++)
                {
                    names[i] = ctx.LoadNonEmptyString();
                    useCounter[i] = ctx.Reader.ReadBoolByte();
                    if (!useCounter[i])
                        states[i] = TauswortheHybrid.State.Load(ctx.Reader);
                }

                return new Bindings(useCounter, states, input, false, names);
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   byte: useCounter
                //   if !useCounter
                //     uint: seed0
                //     uint: seed1
                //     uint: seed2
                //     uint: seed3
                int size = InfoCount;

                ctx.Writer.Write(size);
                for (int i = 0; i < size; i++)
                {
                    ctx.SaveNonEmptyString(GetColumnNameCore(i));
                    ctx.Writer.WriteBoolByte(UseCounter[i]);
                    if (!UseCounter[i])
                        States[i].Save(ctx.Writer);
                }
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                return UseCounter[iinfo] ? NumberDataViewType.Int64 : NumberDataViewType.Single;
            }

            protected override IEnumerable<KeyValuePair<string, DataViewType>> GetAnnotationTypesCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                var items = base.GetAnnotationTypesCore(iinfo);
                if (!UseCounter[iinfo])
                    items.Prepend(BooleanDataViewType.Instance.GetPair(AnnotationUtils.Kinds.IsNormalized));
                return items;
            }

            protected override DataViewType GetAnnotationTypeCore(string kind, int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                if (kind == AnnotationUtils.Kinds.IsNormalized && !UseCounter[iinfo])
                    return BooleanDataViewType.Instance;
                return base.GetAnnotationTypeCore(kind, iinfo);
            }

            protected override void GetAnnotationCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                if (kind == AnnotationUtils.Kinds.IsNormalized && !UseCounter[iinfo])
                {
                    AnnotationUtils.Marshal<bool, TValue>(IsNormalized, iinfo, ref value);
                    return;
                }

                base.GetAnnotationCore(kind, iinfo, ref value);
            }

            private void IsNormalized(int iinfo, ref bool dst)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                dst = true;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = GetActiveInput(predicate);
                Contracts.Assert(active.Length == Input.Count);
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        internal const string Summary = "Adds a column with a generated number sequence.";
        internal const string UserName = "Generate Number Transform";
        internal const string ShortName = "Generate";

        public const string LoadName = "GenerateNumberTransform";
        public const string LoaderSignature = "GenNumTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GEN NUMT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(GenerateNumberTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "GenerateNumber";

        /// <summary>
        /// Initializes a new instance of <see cref="GenerateNumberTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="seed">Seed to start random number generator.</param>
        /// <param name="useCounter">Use an auto-incremented integer starting at zero instead of a random number.</param>
        public GenerateNumberTransform(IHostEnvironment env, IDataView input, string name, uint? seed = null, bool useCounter = Defaults.UseCounter)
            : this(env, new Options() { Columns = new[] { new Column() { Name = name } }, Seed = seed ?? Defaults.Seed, UseCounter = useCounter }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public GenerateNumberTransform(IHostEnvironment env, Options options, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            _bindings = Bindings.Create(options, Source.Schema);
        }

        private GenerateNumberTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(float));
            _bindings = Bindings.Create(ctx, Source.Schema);
        }

        public static GenerateNumberTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new GenerateNumberTransform(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(float)
            // bindings
            ctx.Writer.Write(sizeof(float));
            _bindings.Save(ctx);
        }

        public override DataViewSchema OutputSchema => _bindings.AsSchema;

        public override bool CanShuffle { get { return false; } }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Can't use parallel cursors iff some of our columns are active, otherwise, don't care.
            if (_bindings.AnyNewColumnsActive(predicate))
                return false;
            return null;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var input = Source.GetRowCursor(inputCols);
            return new Cursor(Host, _bindings, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            var inputPred = _bindings.GetDependencies(predicate);
            var inputCols = Source.Schema.Where(x => inputPred(x.Index));

            var active = _bindings.GetActive(predicate);
            DataViewRowCursor input;

            if (n > 1 && ShouldUseParallelCursors(predicate) != false)
            {
                var inputs = Source.GetRowCursorSet(inputCols, n);
                Host.AssertNonEmpty(inputs);

                if (inputs.Length != 1)
                {
                    var cursors = new DataViewRowCursor[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                        cursors[i] = new Cursor(Host, _bindings, inputs[i], active);
                    return cursors;
                }
                input = inputs[0];
            }
            else
                input = Source.GetRowCursor(inputCols);

            return new DataViewRowCursor[] { new Cursor(Host, _bindings, input, active) };
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;
            private readonly float[] _values;
            private readonly TauswortheHybrid[] _rngs;
            private readonly long[] _lastCounters;

            public Cursor(IChannelProvider provider, Bindings bindings, DataViewRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.CheckValue(bindings, nameof(bindings));
                Ch.CheckValue(input, nameof(input));
                Ch.CheckParam(active == null || active.Length == bindings.ColumnCount, nameof(active));

                _bindings = bindings;
                _active = active;
                var length = _bindings.InfoCount;
                _getters = new Delegate[length];
                _values = new float[length];
                _rngs = new TauswortheHybrid[length];
                _lastCounters = new long[length];
                for (int iinfo = 0; iinfo < length; iinfo++)
                {
                    _getters[iinfo] = _bindings.UseCounter[iinfo] ? MakeGetter() : (Delegate)MakeGetter(iinfo);
                    if (!_bindings.UseCounter[iinfo] && IsColumnActive(Schema[_bindings.MapIinfoToCol(iinfo)]))
                    {
                        _rngs[iinfo] = new TauswortheHybrid(_bindings.States[iinfo]);
                        _lastCounters[iinfo] = -1;
                    }
                }
            }

            public override DataViewSchema Schema => _bindings.AsSchema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _bindings.ColumnCount);
                return _active == null || _active[column.Index];
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

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.GetGetter<TValue>(Input.Schema[index]);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private ValueGetter<long> MakeGetter()
            {
                return (ref long value) =>
                {
                    Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                    value = Input.Position;
                };
            }

            private void EnsureValue(ref long lastCounter, ref float value, TauswortheHybrid rng)
            {
                Ch.Assert(lastCounter <= Input.Position);
                while (lastCounter < Input.Position)
                {
                    value = rng.NextSingle();
                    lastCounter++;
                }
            }

            private ValueGetter<float> MakeGetter(int iinfo)
            {
                return (ref float value) =>
                {
                    Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                    Ch.Assert(!_bindings.UseCounter[iinfo]);
                    EnsureValue(ref _lastCounters[iinfo], ref _values[iinfo], _rngs[iinfo]);
                    value = _values[iinfo];
                };
            }
        }
    }

    internal static class RandomNumberGenerator
    {
        [TlcModule.EntryPoint(Name = "Transforms.RandomNumberGenerator", Desc = GenerateNumberTransform.Summary, UserName = GenerateNumberTransform.UserName, ShortName = GenerateNumberTransform.ShortName)]
        public static CommonOutputs.TransformOutput Generate(IHostEnvironment env, GenerateNumberTransform.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GenerateNumber", input);
            var xf = new GenerateNumberTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
