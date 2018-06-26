// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(BootstrapSampleTransform.Summary, typeof(BootstrapSampleTransform), typeof(BootstrapSampleTransform.Arguments), typeof(SignatureDataTransform),
    BootstrapSampleTransform.UserName, "BootstrapSampleTransform", "BootstrapSample")]

[assembly: LoadableClass(BootstrapSampleTransform.Summary, typeof(BootstrapSampleTransform), null, typeof(SignatureLoadDataTransform),
    BootstrapSampleTransform.UserName, BootstrapSampleTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(BootstrapSample))]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This class approximates bootstrap sampling of a dataview.
    /// </summary>
    public sealed class BootstrapSampleTransform : FilterBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether this is the out-of-bag sample, that is, all those rows that are not selected by the transform.",
                ShortName = "comp")]
            public bool Complement;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed. If unspecified random state will be instead derived from the environment.")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should attempt to shuffle the source data. By default on, but can be turned off for efficiency.", ShortName = "si")]
            public bool ShuffleInput = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "When shuffling the output, the number of output rows to keep in that pool. Note that shuffling of output is completely distinct from shuffling of input.", ShortName = "pool")]
            public int PoolSize = 1000;
        }

        internal const string Summary = "Approximate bootstrap sampling.";
        internal const string UserName = "Bootstrap Sample Transform";

        public const string LoaderSignature = "BootstrapSampleTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BTSAMPXF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        internal const string RegistrationName = "BootstrapSample";

        public override bool CanShuffle { get { return false; } }

        private readonly bool _complement;
        private readonly TauswortheHybrid.State _state;
        private readonly bool _shuffleInput;
        private readonly int _poolSize;

        public BootstrapSampleTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(args.PoolSize >= 0, nameof(args.PoolSize), "Cannot be negative");

            _complement = args.Complement;
            _state = new TauswortheHybrid.State(args.Seed ?? (uint)Host.Rand.Next());
            _shuffleInput = args.ShuffleInput;
            _poolSize = args.PoolSize;
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="complement">Whether this is the out-of-bag sample, that is, all those rows that are not selected by the transform.</param>
        /// <param name="seed">The random seed. If unspecified random state will be instead derived from the environment.</param>
        /// <param name="shuffleInput">Whether we should attempt to shuffle the source data. By default on, but can be turned off for efficiency.</param>
        /// <param name="poolSize">When shuffling the output, the number of output rows to keep in that pool. Note that shuffling of output is completely distinct from shuffling of input.</param>
        public BootstrapSampleTransform(IHostEnvironment env, IDataView input, bool complement = false, uint? seed = null, bool shuffleInput = true, int poolSize = 1000)
            : this(env, new Arguments() { Complement = complement, Seed = seed, ShuffleInput = shuffleInput, PoolSize = poolSize }, input)
        {
            
        }

        private BootstrapSampleTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            host.AssertValue(ctx);
            host.AssertValue(input);

            // *** Binary format ***
            // byte: is the compliment sample, that is, an out-of-bag sample
            // uint: seed0
            // uint: seed1
            // uint: seed2
            // uint: seed3
            // byte: input source should be shuffled
            // int: size of the output pool size

            _complement = ctx.Reader.ReadBoolByte();
            _state = TauswortheHybrid.State.Load(ctx.Reader);
            _shuffleInput = ctx.Reader.ReadBoolByte();
            _poolSize = ctx.Reader.ReadInt32();
            Host.CheckDecode(_poolSize >= 0);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // byte: is the compliment sample, that is, an out-of-bag sample
            // uint: seed0
            // uint: seed1
            // uint: seed2
            // uint: seed3
            // byte: input source should be shuffled
            // int: size of the output pool size

            ctx.Writer.WriteBoolByte(_complement);
            _state.Save(ctx.Writer);
            ctx.Writer.WriteBoolByte(_shuffleInput);
            Host.Assert(_poolSize >= 0);
            ctx.Writer.Write(_poolSize);
        }

        public static BootstrapSampleTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new BootstrapSampleTransform(h, ctx, input));
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return false;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            // We do not use the input random because this cursor does not support shuffling.
            var rgen = new TauswortheHybrid(_state);
            var input = Source.GetRowCursor(predicate, _shuffleInput ? new TauswortheHybrid(rgen) : null);
            IRowCursor cursor = new RowCursor(this, input, rgen);
            if (_poolSize > 1)
                cursor = ShuffleTransform.GetShuffledCursor(Host, _poolSize, cursor, new TauswortheHybrid(rgen));
            return cursor;
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            var cursor = GetRowCursorCore(predicate, rand);
            consolidator = null;
            return new IRowCursor[] { cursor };
        }

        private sealed class RowCursor : LinkedRootCursorBase<IRowCursor>, IRowCursor
        {
            private int _remaining;
            private readonly BootstrapSampleTransform _parent;
            private readonly IRandom _rgen;

            public override long Batch { get { return 0; } }

            public ISchema Schema { get { return Input.Schema; } }

            public RowCursor(BootstrapSampleTransform parent, IRowCursor input, IRandom rgen)
                : base(parent.Host, input)
            {
                Ch.AssertValue(rgen);
                _parent = parent;
                _rgen = rgen;
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                var inputIdGetter = Input.GetIdGetter();
                return
                    (ref UInt128 val) =>
                    {
                        inputIdGetter(ref val);
                        val = val.Combine(new UInt128((ulong)_remaining, 0));
                    };
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                return Input.GetGetter<TValue>(col);
            }

            public bool IsColumnActive(int col)
            {
                return Input.IsColumnActive(col);
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(State != CursorState.Done);
                Ch.Assert(_remaining >= 0);
                while (_remaining == 0 && Input.MoveNext())
                {
                    _remaining = Stats.SampleFromPoisson(_rgen, 1);
                    if (_parent._complement)
                        _remaining = _remaining == 0 ? 1 : 0;
                }
                return _remaining-- > 0;
            }
        }
    }

    public static class BootstrapSample
    {
        [TlcModule.EntryPoint(Name = "Transforms.ApproximateBootstrapSampler", Desc = BootstrapSampleTransform.Summary, UserName = BootstrapSampleTransform.UserName, ShortName = BootstrapSampleTransform.RegistrationName)]
        public static CommonOutputs.TransformOutput GetSample(IHostEnvironment env, BootstrapSampleTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "BootstrapSample", input);
            var view = new BootstrapSampleTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
