// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(LabelIndicatorTransform), typeof(LabelIndicatorTransform.Options), typeof(SignatureDataTransform),
    LabelIndicatorTransform.UserName, LabelIndicatorTransform.LoadName, "LabelIndicator")]
[assembly: LoadableClass(typeof(LabelIndicatorTransform), null, typeof(SignatureLoadDataTransform), LabelIndicatorTransform.UserName,
    LabelIndicatorTransform.LoaderSignature)]
[assembly: LoadableClass(typeof(void), typeof(LabelIndicatorTransform), null, typeof(SignatureEntryPointModule), LabelIndicatorTransform.LoadName)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Remaps multiclass labels to binary T,F labels, primarily for use with OVA.
    /// </summary>
    [BestFriend]
    internal sealed class LabelIndicatorTransform : OneToOneTransformBase
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
                loaderAssemblyName: typeof(LabelIndicatorTransform).Assembly.FullName);
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The positive example class for binary classification.", ShortName = "index")]
            public int? ClassIndex;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label of the positive class.", ShortName = "index")]
            public int ClassIndex;
        }

        public static LabelIndicatorTransform Create(IHostEnvironment env,
            ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch => new LabelIndicatorTransform(h, ctx, input));
        }

        public static LabelIndicatorTransform Create(IHostEnvironment env,
            Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch => new LabelIndicatorTransform(h, options, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveBase(ctx);
            ctx.Writer.WriteIntStream(_classIndex);
        }

        private static string TestIsMulticlassLabel(DataViewType type)
        {
            if (type.GetKeyCount() > 0 || type == NumberDataViewType.Single || type == NumberDataViewType.Double)
                return null;
            return $"Label column type is not supported for binary remapping: {type}. Supported types: key, float, double.";
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LabelIndicatorTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="classIndex">Label of the positive class.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the input column.  If this is null '<paramref name="name"/>' will be used.</param>
        public LabelIndicatorTransform(IHostEnvironment env,
            IDataView input,
            int classIndex,
            string name,
            string source = null)
            : this(env, new Options() { Columns = new[] { new Column() { Source = source ?? name, Name = name } }, ClassIndex = classIndex }, input)
        {
        }

        public LabelIndicatorTransform(IHostEnvironment env, Options options, IDataView input)
            : base(env, LoadName, Contracts.CheckRef(options, nameof(options)).Columns,
                input, TestIsMulticlassLabel)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(options.Columns));
            _classIndex = new int[Infos.Length];

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
                _classIndex[iinfo] = options.Columns[iinfo].ClassIndex ?? options.ClassIndex;

            Metadata.Seal();
        }

        private LabelIndicatorTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsMulticlassLabel)
        {
            Host.AssertValue(ctx);
            Host.AssertNonEmpty(Infos);

            _classIndex = new int[Infos.Length];

            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
                _classIndex[iinfo] = ctx.Reader.ReadInt32();

            Metadata.Seal();
        }

        protected override DataViewType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return BooleanDataViewType.Instance;
        }

        protected override Delegate GetGetterCore(IChannel ch, DataViewRow input,
            int iinfo, out Action disposer)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            return GetGetter(ch, input, iinfo);
        }

        private ValueGetter<bool> GetGetter(IChannel ch, DataViewRow input, int iinfo)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);

            var info = Infos[iinfo];
            var column = input.Schema[info.Source];
            ch.Assert(TestIsMulticlassLabel(info.TypeSrc) == null);

            if (info.TypeSrc.GetKeyCount() > 0)
            {
                var srcGetter = input.GetGetter<uint>(column);
                var src = default(uint);
                uint cls = (uint)(_classIndex[iinfo] + 1);

                return
                    (ref bool dst) =>
                    {
                        srcGetter(ref src);
                        dst = src == cls;
                    };
            }
            if (info.TypeSrc == NumberDataViewType.Single)
            {
                var srcGetter = input.GetGetter<float>(column);
                var src = default(float);

                return
                    (ref bool dst) =>
                    {
                        srcGetter(ref src);
                        dst = src == _classIndex[iinfo];
                    };
            }
            if (info.TypeSrc == NumberDataViewType.Double)
            {
                var srcGetter = input.GetGetter<double>(column);
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
        public static CommonOutputs.TransformOutput LabelIndicator(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("LabelIndictator");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
    }
}
