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
using System.Text;

[assembly: LoadableClass(typeof(LabelIndicatingTransformer), typeof(LabelIndicatingTransformer.Arguments), typeof(SignatureDataTransform),
    LabelIndicatingTransformer.UserName, LabelIndicatingTransformer.LoadName, "LabelIndicator")]
[assembly: LoadableClass(typeof(LabelIndicatingTransformer), null, typeof(SignatureLoadDataTransform), LabelIndicatingTransformer.UserName,
    LabelIndicatingTransformer.LoaderSignature)]
[assembly: LoadableClass(typeof(void), typeof(LabelIndicatingTransformer), null, typeof(SignatureEntryPointModule), LabelIndicatingTransformer.LoadName)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Remaps multiclass labels to binary T,F labels, primarily for use with OVA.
    /// </summary>
    public sealed class LabelIndicatingTransformer : OneToOneTransformBase
    {
        internal const string Summary = "Remaps labels from multiclass to binary, for OVA.";
        internal const string UserName = "Label Indicator Transform";
        public const string LoaderSignature = "LabelIndicatorTransform";
        public const string LoadName = LoaderSignature;

        private readonly int[] _classIndex;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LBINDTRN",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LabelIndicatingTransformer).Assembly.FullName);
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The positive example class for binary classification.", ShortName = "index")]
            public int? ClassIndex;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label of the positive class.", ShortName = "index")]
            public int ClassIndex;
        }

        public static LabelIndicatingTransformer Create(IHostEnvironment env,
            ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch => new LabelIndicatingTransformer(h, ctx, input));
        }

        public static LabelIndicatingTransformer Create(IHostEnvironment env,
            Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch => new LabelIndicatingTransformer(h, args, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveBase(ctx);
            ctx.Writer.WriteIntStream(_classIndex);
        }

        private static string TestIsMulticlassLabel(ColumnType type)
        {
            if (type.KeyCount > 0 || type == NumberType.R4 || type == NumberType.R8)
                return null;
            return $"Label column type is not supported for binary remapping: {type}. Supported types: key, float, double.";
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LabelIndicatingTransformer"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="classIndex">Label of the positive class.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the input column.  If this is null '<paramref name="name"/>' will be used.</param>
        public LabelIndicatingTransformer(IHostEnvironment env,
            IDataView input,
            int classIndex,
            string name,
            string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, ClassIndex = classIndex }, input)
        {
        }

        public LabelIndicatingTransformer(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, LoadName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsMulticlassLabel)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));
            _classIndex = new int[Infos.Length];

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
                _classIndex[iinfo] = args.Column[iinfo].ClassIndex ?? args.ClassIndex;

            Metadata.Seal();
        }

        private LabelIndicatingTransformer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsMulticlassLabel)
        {
            Host.AssertValue(ctx);
            Host.AssertNonEmpty(Infos);

            _classIndex = new int[Infos.Length];

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
                _classIndex[iinfo] = ctx.Reader.ReadInt32();

            Metadata.Seal();
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return BoolType.Instance;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input,
            int iinfo, out Action disposer)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            return GetGetter(ch, input, iinfo);
        }

        private ValueGetter<bool> GetGetter(IChannel ch, IRow input, int iinfo)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);

            var info = Infos[iinfo];
            ch.Assert(TestIsMulticlassLabel(info.TypeSrc) == null);

            if (info.TypeSrc.KeyCount > 0)
            {
                var srcGetter = input.GetGetter<uint>(info.Source);
                var src = default(uint);
                uint cls = (uint)(_classIndex[iinfo] + 1);

                return
                    (ref bool dst) =>
                    {
                        srcGetter(ref src);
                        dst = src == cls;
                    };
            }
            if (info.TypeSrc == NumberType.R4)
            {
                var srcGetter = input.GetGetter<float>(info.Source);
                var src = default(float);

                return
                    (ref bool dst) =>
                    {
                        srcGetter(ref src);
                        dst = src == _classIndex[iinfo];
                    };
            }
            if (info.TypeSrc == NumberType.R8)
            {
                var srcGetter = input.GetGetter<double>(info.Source);
                var src = default(double);

                return
                    (ref bool dst) =>
                    {
                        srcGetter(ref src);
                        dst = src == _classIndex[iinfo];
                    };
            }
            throw Host.ExceptNotSupp($"Label column type is not supported for binary remapping: {info.TypeSrc}. Supported types: key, float, double.");
        }

        [TlcModule.EntryPoint(Name = "Transforms.LabelIndicator", Desc = "Label remapper used by OVA", UserName = "LabelIndicator",
            ShortName = "LabelIndictator")]
        public static CommonOutputs.TransformOutput LabelIndicator(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("LabelIndictator");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
