// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using Float = System.Single;

[assembly: LoadableClass(NumberGeneratingTransformer.Summary, typeof(NumberGeneratingTransformer), typeof(NumberGeneratingTransformer.Arguments), typeof(SignatureDataTransform),
    NumberGeneratingTransformer.UserName, NumberGeneratingTransformer.LoadName, "GenerateNumber", NumberGeneratingTransformer.ShortName)]

[assembly: LoadableClass(NumberGeneratingTransformer.Summary, typeof(NumberGeneratingTransformer), null, typeof(SignatureLoadDataTransform),
    NumberGeneratingTransformer.UserName, NumberGeneratingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(RandomNumberGenerator))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This transform adds columns containing either random numbers distributed
    /// uniformly between 0 and 1 or an auto-incremented integer starting at zero.
    /// It will be used in conjunction with a filter transform to create random
    /// partitions of the data, used in cross validation.
    /// </summary>
    public sealed class NumberGeneratingTransformer : RowToRowTransformBase
    {
        public sealed class Column
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use an auto-incremented integer starting at zero instead of a random number", ShortName = "cnt")]
            public bool? UseCounter;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed")]
            public uint? Seed;

            public static Column Parse(string str)
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

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:seed)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

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
                ISchema input, bool user, string[] names)
                : base(input, user, names)
            {
                Contracts.Assert(Utils.Size(useCounter) == InfoCount);
                Contracts.Assert(Utils.Size(states) == InfoCount);
                UseCounter = useCounter;
                States = states;
            }

            public static Bindings Create(Arguments args, ISchema input)
            {
                var names = new string[args.Column.Length];
                var useCounter = new bool[args.Column.Length];
                var states = new TauswortheHybrid.State[args.Column.Length];
                for (int i = 0; i < args.Column.Length; i++)
                {
                    var item = args.Column[i];
                    names[i] = item.Name;
                    useCounter[i] = item.UseCounter ?? args.UseCounter;
                    if (!useCounter[i])
                        states[i] = new TauswortheHybrid.State(item.Seed ?? args.Seed);
                }

                return new Bindings(useCounter, states, input, true, names);
            }

            public static Bindings Create(ModelLoadContext ctx, ISchema input)
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

            public void Save(ModelSaveContext ctx)
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

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                return UseCounter[iinfo] ? NumberType.I8 : NumberType.Float;
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                var items = base.GetMetadataTypesCore(iinfo);
                if (!UseCounter[iinfo])
                    items.Prepend(BoolType.Instance.GetPair(MetadataUtils.Kinds.IsNormalized));
                return items;
            }

            protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                if (kind == MetadataUtils.Kinds.IsNormalized && !UseCounter[iinfo])
                    return BoolType.Instance;
                return base.GetMetadataTypeCore(kind, iinfo);
            }

            protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                if (kind == MetadataUtils.Kinds.IsNormalized && !UseCounter[iinfo])
                {
                    MetadataUtils.Marshal<bool, TValue>(IsNormalized, iinfo, ref value);
                    return;
                }

                base.GetMetadataCore(kind, iinfo, ref value);
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
                Contracts.Assert(active.Length == Input.ColumnCount);
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
                loaderAssemblyName: typeof(NumberGeneratingTransformer).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "GenerateNumber";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="useCounter">Use an auto-incremented integer starting at zero instead of a random number.</param>
        public NumberGeneratingTransformer(IHostEnvironment env, IDataView input, string name, bool useCounter = Defaults.UseCounter)
            : this(env, new Arguments() { Column = new[] { new Column() { Name = name } }, UseCounter = useCounter }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public NumberGeneratingTransformer(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            _bindings = Bindings.Create(args, Source.Schema);
        }

        private NumberGeneratingTransformer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Float));
            _bindings = Bindings.Create(ctx, Source.Schema);
        }

        public static NumberGeneratingTransformer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NumberGeneratingTransformer(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            ctx.Writer.Write(sizeof(Float));
            _bindings.Save(ctx);
        }

        public override Schema Schema => _bindings.AsSchema;

        public override bool CanShuffle { get { return false; } }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Can't use parallel cursors iff some of our columns are active, otherwise, don't care.
            if (_bindings.AnyNewColumnsActive(predicate))
                return false;
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred);
            return new RowCursor(Host, _bindings, input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            IRowCursor input;

            if (n > 1 && ShouldUseParallelCursors(predicate) != false)
            {
                var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n);
                Host.AssertNonEmpty(inputs);

                if (inputs.Length != 1)
                {
                    var cursors = new IRowCursor[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                        cursors[i] = new RowCursor(Host, _bindings, inputs[i], active);
                    return cursors;
                }
                input = inputs[0];
            }
            else
                input = Source.GetRowCursor(inputPred);

            consolidator = null;
            return new IRowCursor[] { new RowCursor(Host, _bindings, input, active) };
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;
            private readonly Float[] _values;
            private readonly TauswortheHybrid[] _rngs;
            private readonly long[] _lastCounters;

            public RowCursor(IChannelProvider provider, Bindings bindings, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.CheckValue(bindings, nameof(bindings));
                Ch.CheckValue(input, nameof(input));
                Ch.CheckParam(active == null || active.Length == bindings.ColumnCount, nameof(active));

                _bindings = bindings;
                _active = active;
                var length = _bindings.InfoCount;
                _getters = new Delegate[length];
                _values = new Float[length];
                _rngs = new TauswortheHybrid[length];
                _lastCounters = new long[length];
                for (int iinfo = 0; iinfo < length; iinfo++)
                {
                    _getters[iinfo] = _bindings.UseCounter[iinfo] ? MakeGetter() : (Delegate)MakeGetter(iinfo);
                    if (!_bindings.UseCounter[iinfo] && IsColumnActive(_bindings.MapIinfoToCol(iinfo)))
                    {
                        _rngs[iinfo] = new TauswortheHybrid(_bindings.States[iinfo]);
                        _lastCounters[iinfo] = -1;
                    }
                }
            }

            public Schema Schema => _bindings.AsSchema;

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

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

            private ValueGetter<long> MakeGetter()
            {
                return (ref long value) =>
                {
                    Ch.Check(IsGood);
                    value = Input.Position;
                };
            }

            private void EnsureValue(ref long lastCounter, ref Float value, TauswortheHybrid rng)
            {
                Ch.Assert(lastCounter <= Input.Position);
                while (lastCounter < Input.Position)
                {
                    value = rng.NextSingle();
                    lastCounter++;
                }
            }

            private ValueGetter<Float> MakeGetter(int iinfo)
            {
                return (ref Float value) =>
                {
                    Ch.Check(IsGood);
                    Ch.Assert(!_bindings.UseCounter[iinfo]);
                    EnsureValue(ref _lastCounters[iinfo], ref _values[iinfo], _rngs[iinfo]);
                    value = _values[iinfo];
                };
            }
        }
    }

    public static class RandomNumberGenerator
    {
        [TlcModule.EntryPoint(Name = "Transforms.RandomNumberGenerator", Desc = NumberGeneratingTransformer.Summary, UserName = NumberGeneratingTransformer.UserName, ShortName = NumberGeneratingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Generate(IHostEnvironment env, NumberGeneratingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GenerateNumber", input);
            var xf = new NumberGeneratingTransformer(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
