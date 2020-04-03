using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{

    public sealed class SrCnnEntireTransformer : OneToOneTransformerBase
    {
        internal const string Summary = "This transformer detct anomalies for input timeseries using SRCNN";
        internal const string LoaderSignature = "SrCnnEntireTransformer";
        internal const string UserName = "SrCnn Entire Anomaly Detection";
        internal const string ShortName = "srcnn entire";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.", ShortName = "tgt", SortOrder = 2)]
            public string Target;

            [Argument(ArgumentType.Required, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
                ShortName = "thre", SortOrder = 106)]
            public double Threshold = 0.3;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The detection mode, affect output vector length",
                ShortName = "mode", SortOrder = 107)]
            public SrCnnDetectMode SrCnnDetectMode = SrCnnDetectMode.AnomalyOnly;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The sensitivity of boundary",
                ShortName = "sens", SortOrder = 108)]
            public Double Sensitivity = 99;
        }

        internal readonly string InputColumnName;

        internal readonly string OutputColumnName;

        internal double Threshold { get; }

        internal SrCnnDetectMode SrCnnDetectMode { get; }

        internal double Sensitivity { get; }

        internal int OutputLength { get; }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SRENTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SrCnnEntireTransformer).Assembly.FullName);
        }

        internal SrCnnEntireTransformer(IHostEnvironment env, Options options, IDataView input)
            :base(Contracts.CheckRef(env, nameof(env)).Register(LoaderSignature), new[] { (options.Target, options.Source) })
        {
            InputColumnName = options.Source;
            OutputColumnName = options.Target;

            Host.CheckUserArg(options.Threshold >= 0 && options.Threshold <= 1, nameof(Options.Threshold), "Must be in [0,1]");
            Threshold = options.Threshold;

            if (options.SrCnnDetectMode.Equals(SrCnnDetectMode.AnomalyOnly))
            {
                Sensitivity = options.Sensitivity;
                OutputLength = 3;
            }
            else if (options.SrCnnDetectMode.Equals(SrCnnDetectMode.AnomalyAndMargin))
            {
                Host.CheckUserArg(options.Sensitivity >= 0 && options.Sensitivity <= 100, nameof(Options.Sensitivity), "Must be in [0,100]");
                Sensitivity = options.Sensitivity;
                OutputLength = 7;
            }
        }

        private SrCnnEntireTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            //TODO
        }

        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SrCnnEntireTransformer(env, options, input).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnEntireTransformer(host, ctx).MakeDataTransform(input);
        }

        internal static SrCnnEntireTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnEntireTransformer(host, ctx);
        }

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            //TO DO
            throw new NotImplementedException();
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly SrCnnEntireTransformer _parent;
            private readonly VBuffer<ReadOnlyMemory<Char>> _slotNames;

            public Mapper(SrCnnEntireTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                if (_parent.OutputLength == 3)
                {
                    _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Alert".AsMemory(), "Raw Score".AsMemory(), "Mag".AsMemory() });
                }
                else if (_parent.OutputLength == 7)
                {
                    _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Is Anomaly".AsMemory(), "Anomaly Score".AsMemory(), "Mag".AsMemory(),
                        "Expected Value".AsMemory(), "Boundary Unit".AsMemory(), "Upper Boundary".AsMemory(), "Lower Boundary".AsMemory() });
                }
                //TODO
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var meta = new DataViewSchema.Annotations.Builder();
                meta.AddSlotNames(_parent.OutputLength, GetSlotNames);
                var info = new DataViewSchema.DetachedColumn[1];
                info[0] = new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Double, _parent.OutputLength), meta.ToAnnotations());
                return info;
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst) => _slotNames.CopyTo(ref dst, 0, _parent.OutputLength);

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                //TODO
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(SrCnnTsPoint);
                var getSrc = input.GetGetter<SrCnnTsPoint>(input.Schema[ColMapNewToOld[iinfo]]);

                disposer = null;

                ValueGetter<VBuffer<double>> del =
                    (ref VBuffer<double> dst) =>
                    {
                        getSrc(ref src);
                        if (src == null)
                            return;
                        var result = VBufferEditor.Create(ref dst, _parent.OutputLength);
                        result.Values.Fill(Double.NaN);
                        result.Values[0] = 1;//IsAnomaly
                        result.Values[1] = 0.5;//AnomalyScore
                        result.Values[2] = 0.0;//Mag
                        result.Values[3] = src.Value;//ExpectedValue
                        result.Values[4] = 0.5 * src.Value;//Boundary Unit
                        result.Values[5] = result.Values[3] + result.Values[4];//UpperBoundary
                        result.Values[6] = result.Values[3] - result.Values[4];//UpperBoundary

                        dst = result.Commit();
                    };

                return del;
            }
        }
    }
}
