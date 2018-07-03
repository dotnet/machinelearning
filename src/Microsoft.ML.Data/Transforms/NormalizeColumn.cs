// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(NormalizeTransform.MinMaxNormalizerSummary, typeof(NormalizeTransform), typeof(NormalizeTransform.MinMaxArguments), typeof(SignatureDataTransform),
    NormalizeTransform.MinMaxNormalizerUserName, "MinMaxNormalizer", NormalizeTransform.MinMaxNormalizerShortName)]

[assembly: LoadableClass(NormalizeTransform.MeanVarNormalizerSummary, typeof(NormalizeTransform), typeof(NormalizeTransform.MeanVarArguments), typeof(SignatureDataTransform),
    NormalizeTransform.MeanVarNormalizerUserName, "MeanVarNormalizer", NormalizeTransform.MeanVarNormalizerShortName, "ZScoreNormalizer", "ZScore", "GaussianNormalizer", "Gaussian")]

[assembly: LoadableClass(NormalizeTransform.LogMeanVarNormalizerSummary, typeof(NormalizeTransform), typeof(NormalizeTransform.LogMeanVarArguments), typeof(SignatureDataTransform),
    NormalizeTransform.LogMeanVarNormalizerUserName, "LogMeanVarNormalizer", NormalizeTransform.LogMeanVarNormalizerShortName, "LogNormalNormalizer", "LogNormal")]

[assembly: LoadableClass(NormalizeTransform.BinNormalizerSummary, typeof(NormalizeTransform), typeof(NormalizeTransform.BinArguments), typeof(SignatureDataTransform),
    NormalizeTransform.BinNormalizerUserName, "BinNormalizer", NormalizeTransform.BinNormalizerShortName)]

[assembly: LoadableClass(NormalizeTransform.SupervisedBinNormalizerSummary, typeof(NormalizeTransform), typeof(NormalizeTransform.SupervisedBinArguments), typeof(SignatureDataTransform),
    NormalizeTransform.SupervisedBinNormalizerUserName, "SupervisedBinNormalizer", NormalizeTransform.SupervisedBinNormalizerShortName)]

[assembly: LoadableClass(typeof(NormalizeTransform.AffineColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "Affine Normalizer", AffineNormSerializationUtils.LoaderSignature)]

[assembly: LoadableClass(typeof(NormalizeTransform.CdfColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "CDF Normalizer", NormalizeTransform.CdfColumnFunction.LoaderSignature)]

[assembly: LoadableClass(NormalizeTransform.BinNormalizerSummary, typeof(NormalizeTransform.BinColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "Bin Normalizer", NormalizeTransform.BinColumnFunction.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed partial class NormalizeTransform
    {
        public abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of examples used to train the normalizer", ShortName = "maxtrain")]
            public long? MaxTrainingExamples;

            protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (MaxTrainingExamples != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        // REVIEW: Support different aggregators on different columns, eg, MinMax vs Variance/ZScore.
        public abstract class FixZeroColumnBase : ColumnBase
        {
            // REVIEW: This only allows mapping either zero or min to zero. It might make sense to allow also max, midpoint and mean to be mapped to zero.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to map zero to zero, preserving sparsity", ShortName = "zero")]
            public bool? FixZero;

            protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (FixZero != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        public sealed class AffineColumn : FixZeroColumnBase
        {
            public static AffineColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new AffineColumn();
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

        public sealed class BinColumn : FixZeroColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bins, power of 2 recommended", ShortName = "bins")]
            [TGUI(Label = "Max number of bins")]
            public int? NumBins;

            public static BinColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new BinColumn();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NumBins != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class LogNormalColumn : ColumnBase
        {
            public static LogNormalColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new LogNormalColumn();
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

        private static class Defaults
        {
            public const bool FixZero = true;
            public const bool MeanVarCdf = false;
            public const bool LogMeanVarCdf = true;
            public const int NumBins = 1024;
            public const int MinBinSize = 10;
        }

        public abstract class FixZeroArgumentsBase : ArgumentsBase
        {
            // REVIEW: This only allows mapping either zero or min to zero. It might make sense to allow also max, midpoint and mean to be mapped to zero.
            // REVIEW: Convert this to bool? or even an enum{Auto, No, Yes}, and automatically map zero to zero when it is null/Auto.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to map zero to zero, preserving sparsity", ShortName = "zero")]
            public bool FixZero = Defaults.FixZero;
        }

        public abstract class AffineArgumentsBase : FixZeroArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public AffineColumn[] Column;

            public override OneToOneColumn[] GetColumns() => Column;
        }

        public sealed class MinMaxArguments : AffineArgumentsBase
        {
        }

        public sealed class MeanVarArguments : AffineArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use CDF as the output", ShortName = "cdf")]
            public bool UseCdf = Defaults.MeanVarCdf;
        }

        public sealed class LogMeanVarArguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use CDF as the output", ShortName = "cdf")]
            public bool UseCdf = Defaults.LogMeanVarCdf;

            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public LogNormalColumn[] Column;

            public override OneToOneColumn[] GetColumns() => Column;
        }

        public abstract class BinArgumentsBase : FixZeroArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public BinColumn[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bins, power of 2 recommended", ShortName = "bins")]
            [TGUI(Label = "Max number of bins")]
            public int NumBins = Defaults.NumBins;

            public override OneToOneColumn[] GetColumns() => Column;
        }

        public sealed class BinArguments : BinArgumentsBase
        {
        }

        public sealed class SupervisedBinArguments : BinArgumentsBase
        {
            // REVIEW: factor in a loss function / optimization algorithm to make it work better in regression case
            [Argument(ArgumentType.Required, HelpText = "Label column for supervised binning", ShortName = "label,lab",
                Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of examples per bin")]
            public int MinBinSize = Defaults.MinBinSize;
        }

        public const string MinMaxNormalizerSummary = "Normalizes the data based on the observed minimum and maximum values of the data.";
        public const string MeanVarNormalizerSummary = "Normalizes the data based on the computed mean and variance of the data.";
        public const string LogMeanVarNormalizerSummary = "Normalizes the data based on the computed mean and variance of the logarithm of the data.";
        public const string BinNormalizerSummary = "The values are assigned into equidensity bins and a value is mapped to its bin_number/number_of_bins.";
        public const string SupervisedBinNormalizerSummary = "Similar to BinNormalizer, but calculates bins based on correlation with the label column, not equi-density. "
            + "The new value is bin_number / number_of_bins.";

        public const string MinMaxNormalizerUserName = "Min-Max Normalizer";
        public const string MeanVarNormalizerUserName = "MeanVar Normalizer";
        public const string LogMeanVarNormalizerUserName = "LogMeanVar Normalizer";
        public const string BinNormalizerUserName = "Binning Normalizer";
        public const string SupervisedBinNormalizerUserName = "Supervised Binning Normalizer";

        public const string MinMaxNormalizerShortName = "MinMax";
        public const string MeanVarNormalizerShortName = "MeanVar";
        public const string LogMeanVarNormalizerShortName = "LogMeanVar";
        public const string BinNormalizerShortName = "Bin";
        public const string SupervisedBinNormalizerShortName = "SupBin";

        /// <summary>
        /// A helper method to create MinMaxNormalizer transform for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        public static NormalizeTransform CreateMinMaxNormalizer(IHostEnvironment env, IDataView input, string name, string source = null)
        {
            var args = new MinMaxArguments()
            {
                Column = new[] { new AffineColumn(){
                        Source = source ?? name,
                        Name = name
                    }
                }
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Public create method corresponding to SignatureDataTransform.
        /// </summary>
        public static NormalizeTransform Create(IHostEnvironment env, MinMaxArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize(MinMax)");
            NormalizeTransform func;
            using (var ch = h.Start("Training"))
            {
                func = Create(h, args, input, MinMaxUtils.CreateBuilder);
                ch.Done();
            }
            return func;
        }

        /// <summary>
        /// A helper method to create MeanVarNormalizer transform for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// /// <param name="useCdf">Whether to use CDF as the output.</param>
        public static NormalizeTransform CreateMeanVarNormalizer(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            bool useCdf = Defaults.MeanVarCdf)
        {
            var args = new MeanVarArguments()
            {
                Column = new[] { new AffineColumn(){
                        Source = source ?? name,
                        Name = name
                    }
                },
                UseCdf = useCdf
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Public create method corresponding to SignatureDataTransform.
        /// </summary>
        public static NormalizeTransform Create(IHostEnvironment env, MeanVarArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize(MeanVar)");
            NormalizeTransform func;
            using (var ch = h.Start("Training"))
            {
                func = Create(h, args, input, MeanVarUtils.CreateBuilder);
                ch.Done();
            }
            return func;
        }

        /// <summary>
        /// A helper method to create LogMeanVarNormalizer transform for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// /// <param name="useCdf">Whether to use CDF as the output.</param>
        public static NormalizeTransform CreateLogMeanVarNormalizer(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            bool useCdf = Defaults.LogMeanVarCdf)
        {
            var args = new LogMeanVarArguments()
            {
                Column = new[] { new LogNormalColumn(){
                        Source = source ?? name,
                        Name = name
                    }
                },
                UseCdf = useCdf
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Public create method corresponding to SignatureDataTransform.
        /// </summary>
        public static NormalizeTransform Create(IHostEnvironment env, LogMeanVarArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize(LogMeanVar)");
            NormalizeTransform func;
            using (var ch = h.Start("Training"))
            {
                func = Create(h, args, input, LogMeanVarUtils.CreateBuilder);
                ch.Done();
            }
            return func;
        }

        public static NormalizeTransform CreateBinningNormalizer(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int numBins = Defaults.NumBins)
        {
            var args = new BinArguments()
            {
                Column = new[] { new BinColumn(){
                        Source = source ?? name,
                        Name = name
                    }
                },
                NumBins = numBins
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Public create method corresponding to SignatureDataTransform.
        /// </summary>
        public static NormalizeTransform Create(IHostEnvironment env, BinArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize(Bin)");
            NormalizeTransform func;
            using (var ch = h.Start("Training"))
            {
                func = Create(h, args, input, BinUtils.CreateBuilder);
                ch.Done();
            }
            return func;
        }

        public static NormalizeTransform CreateSupervisedBinningNormalizer(IHostEnvironment env,
            IDataView input,
            string labelColumn,
            string name,
            string source = null,
            int numBins = Defaults.NumBins,
            int minBinSize = Defaults.MinBinSize)
        {
            var args = new SupervisedBinArguments()
            {
                Column = new[] { new BinColumn(){
                        Source = source ?? name,
                        Name = name
                    }
                },
                LabelColumn = labelColumn,
                NumBins = numBins,
                MinBinSize = minBinSize
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Public create method corresponding to SignatureDataTransform.
        /// </summary>
        public static NormalizeTransform Create(IHostEnvironment env, SupervisedBinArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize(SupervisedBin)");
            NormalizeTransform func;
            using (var ch = h.Start("Training"))
            {
                var labelColumnId = SupervisedBinUtils.GetLabelColumnId(ch, input.Schema, args.LabelColumn);
                func = Create(h, args, input, SupervisedBinUtils.CreateBuilder, labelColumnId);
                ch.Done();
            }
            return func;
        }

        public abstract partial class AffineColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            // The only derived classes are private inner classes
            private AffineColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            public abstract void Save(ModelSaveContext ctx);

            public abstract JToken PfaInfo(BoundPfaContext ctx, JToken srcToken);

            public abstract bool OnnxInfo(OnnxContext ctx, OnnxUtils.NodeProtoWrapper nodeProtoWrapper, int featureCount);

            public abstract Delegate GetGetter(IRow input, int icol);

            public abstract void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc);

            public static AffineColumnFunction Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc.IsNumber)
                {
                    if (typeSrc == NumberType.R4)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberType.R8)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                else if (typeSrc.ItemType.IsNumber)
                {
                    if (typeSrc.ItemType == NumberType.R4)
                        return Sng.ImplVec.Create(ctx, host, typeSrc);
                    if (typeSrc.ItemType == NumberType.R8)
                        return Dbl.ImplVec.Create(ctx, host, typeSrc);
                }
                throw host.ExceptUserArg(nameof(AffineArgumentsBase.Column), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc.ToString());
            }

            private abstract class ImplOne<TFloat> : AffineColumnFunction
            {
                protected readonly TFloat Scale;
                protected readonly TFloat Offset;

                protected ImplOne(IHost host, TFloat scale, TFloat offset)
                    : base(host)
                {
                    Scale = scale;
                    Offset = offset;
                }

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.RawType == typeof(TFloat));
                    bldr.AddPrimitive("AffineScale", typeSrc, Scale);
                    bldr.AddPrimitive("AffineOffset", typeSrc, Offset);
                }
            }

            private abstract class ImplVec<TFloat> : AffineColumnFunction
            {
                protected readonly TFloat[] Scale;
                protected readonly TFloat[] Offset;
                protected readonly int[] IndicesNonZeroOffset;

                protected ImplVec(IHost host, TFloat[] scale, TFloat[] offset, int[] indicesNonZeroOffset)
                    : base(host)
                {
                    Host.AssertValue(scale);
                    Host.AssertValueOrNull(offset);
                    Host.Assert(indicesNonZeroOffset == null || offset != null);
                    Host.Assert(Offset == null || Offset.Length == Scale.Length);

                    Scale = scale;
                    Offset = offset;
                    IndicesNonZeroOffset = indicesNonZeroOffset;
                }

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.VectorSize == Scale.Length);
                    Host.Check(typeSrc.ItemType.RawType == typeof(TFloat));
                    bldr.AddGetter<VBuffer<TFloat>>("AffineScale", typeSrc, ScaleMetadataGetter);
                    if (Offset != null)
                        bldr.AddGetter<VBuffer<TFloat>>("AffineOffset", typeSrc, OffsetMetadataGetter);
                }

                private void ScaleMetadataGetter(int col, ref VBuffer<TFloat> dst)
                {
                    var src = new VBuffer<TFloat>(Scale.Length, Scale);
                    src.CopyTo(ref dst);
                }

                private void OffsetMetadataGetter(int col, ref VBuffer<TFloat> dst)
                {
                    Host.AssertValue(Offset);
                    var src = new VBuffer<TFloat>(Offset.Length, Offset);
                    src.CopyTo(ref dst);
                }
            }
        }

        public abstract partial class CdfColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            // The only derived classes are private inner classes
            private CdfColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            public abstract void Save(ModelSaveContext ctx);

            public JToken PfaInfo(BoundPfaContext ctx, JToken srcToken)
            {
                return null;
            }

            public bool OnnxInfo(OnnxContext ctx, OnnxUtils.NodeProtoWrapper nodeProtoWrapper, int featureCount)
            {
                return false;
            }

            public abstract Delegate GetGetter(IRow input, int icol);

            public static CdfColumnFunction Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc.IsNumber)
                {
                    if (typeSrc == NumberType.R4)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberType.R8)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                else if (typeSrc.ItemType.IsNumber)
                {
                    if (typeSrc.ItemType == NumberType.R4)
                        return Sng.ImplVec.Create(ctx, host, typeSrc);
                    if (typeSrc.ItemType == NumberType.R8)
                        return Dbl.ImplVec.Create(ctx, host, typeSrc);
                }
                throw host.ExceptUserArg(nameof(AffineArgumentsBase.Column), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc);
            }

            public abstract void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc);

            private abstract class ImplOne<TFloat> : CdfColumnFunction
            {
                protected readonly TFloat Mean;
                protected readonly TFloat Stddev;
                protected readonly bool UseLog;

                protected ImplOne(IHost host, TFloat mean, TFloat stddev, bool useLog)
                    : base(host)
                {
                    Mean = mean;
                    Stddev = stddev;
                    UseLog = useLog;
                }

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.RawType == typeof(TFloat));
                    bldr.AddPrimitive("CdfMean", typeSrc, Mean);
                    bldr.AddPrimitive("CdfStdDev", typeSrc, Stddev);
                    bldr.AddPrimitive("CdfUseLog", BoolType.Instance, (DvBool)UseLog);
                }
            }

            private abstract class ImplVec<TFloat> : CdfColumnFunction
            {
                protected readonly TFloat[] Mean;
                protected readonly TFloat[] Stddev;
                protected readonly bool UseLog;

                protected ImplVec(IHost host, TFloat[] mean, TFloat[] stddev, bool useLog)
                    : base(host)
                {
                    Host.AssertValue(mean);
                    Host.AssertValue(stddev);
                    Host.Assert(mean.Length == stddev.Length);
                    Mean = mean;
                    Stddev = stddev;
                    UseLog = useLog;
                }

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.VectorSize == Mean.Length);
                    Host.Check(typeSrc.ItemType.RawType == typeof(TFloat));
                    bldr.AddGetter<VBuffer<TFloat>>("CdfMean", typeSrc, MeanMetadataGetter);
                    bldr.AddGetter<VBuffer<TFloat>>("CdfStdDev", typeSrc, StddevMetadataGetter);
                    bldr.AddPrimitive("CdfUseLog", BoolType.Instance, (DvBool)UseLog);
                }

                private void MeanMetadataGetter(int col, ref VBuffer<TFloat> dst)
                {
                    var src = new VBuffer<TFloat>(Mean.Length, Mean);
                    src.CopyTo(ref dst);
                }

                private void StddevMetadataGetter(int col, ref VBuffer<TFloat> dst)
                {
                    var src = new VBuffer<TFloat>(Stddev.Length, Stddev);
                    src.CopyTo(ref dst);
                }
            }

            public const string LoaderSignature = "CdfNormalizeFunction";

            public static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "CDFNORMF",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }
        }

        public abstract partial class BinColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            protected BinColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            public abstract void Save(ModelSaveContext ctx);

            public JToken PfaInfo(BoundPfaContext ctx, JToken srcToken)
            {
                return null;
            }

            public bool OnnxInfo(OnnxContext ctx, OnnxUtils.NodeProtoWrapper nodeProtoWrapper, int featureCount)
            {
                return false;
            }

            public abstract Delegate GetGetter(IRow input, int icol);

            public void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc)
            {
                // REVIEW: How to attach information on the bins, to metadata?
            }

            public static BinColumnFunction Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc.IsNumber)
                {
                    if (typeSrc == NumberType.R4)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberType.R8)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                if (typeSrc.IsVector && typeSrc.ItemType.IsNumber)
                {
                    if (typeSrc.ItemType == NumberType.R4)
                        return Sng.ImplVec.Create(ctx, host, typeSrc);
                    if (typeSrc.ItemType == NumberType.R8)
                        return Dbl.ImplVec.Create(ctx, host, typeSrc);
                }
                throw host.ExceptUserArg(nameof(BinArguments.Column), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc);
            }

            public const string LoaderSignature = "BinNormalizeFunction";

            public static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "BINNORMF",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }
        }

        private abstract class OneColumnFunctionBuilderBase<TFloat> : IColumnFunctionBuilder
        {
            protected IHost Host;
            protected readonly long Lim;
            protected long Rem;
            private readonly ValueGetter<TFloat> _getSrc;

            protected OneColumnFunctionBuilderBase(IHost host, long lim, ValueGetter<TFloat> getSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
                Rem = lim;
                Lim = lim;
                _getSrc = getSrc;
            }

            public bool ProcessValue()
            {
                TFloat tmp = default(TFloat);
                _getSrc(ref tmp);
                return ProcessValue(ref tmp);
            }

            protected virtual bool ProcessValue(ref TFloat val)
            {
                Host.Assert(Rem >= 0);
                if (Rem == 0)
                    return false;
                Rem--;
                return true;
            }

            public abstract IColumnFunction CreateColumnFunction();
        }

        private abstract class VecColumnFunctionBuilderBase<TFloat> : IColumnFunctionBuilder
        {
            protected IHost Host;
            protected readonly long Lim;
            protected long Rem;
            private readonly ValueGetter<VBuffer<TFloat>> _getSrc;
            private VBuffer<TFloat> _buffer;

            protected VecColumnFunctionBuilderBase(IHost host, long lim, ValueGetter<VBuffer<TFloat>> getSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
                Rem = lim;
                Lim = lim;
                _getSrc = getSrc;
            }

            public bool ProcessValue()
            {
                _getSrc(ref _buffer);
                return ProcessValue(ref _buffer);
            }

            protected virtual bool ProcessValue(ref VBuffer<TFloat> buffer)
            {
                Host.Assert(Rem >= 0);
                if (Rem == 0)
                    return false;
                Rem--;
                return true;
            }

            public abstract IColumnFunction CreateColumnFunction();
        }

        private abstract class SupervisedBinFunctionBuilderBase : IColumnFunctionBuilder
        {
            protected readonly IHost Host;
            protected readonly long Lim;
            protected long Rem;

            protected readonly List<int> Labels;
            protected readonly int LabelCardinality;
            private readonly ValueGetter<int> _labelGetterSrc;

            protected SupervisedBinFunctionBuilderBase(IHost host, long lim, int labelColId, IRow dataRow)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
                Rem = lim;
                Lim = lim;
                Labels = new List<int>();
                _labelGetterSrc = GetLabelGetter(dataRow, labelColId, out LabelCardinality);
            }

            private ValueGetter<int> GetLabelGetter(IRow row, int col, out int labelCardinality)
            {
                // The label column type is checked as part of args validation.
                var type = row.Schema.GetColumnType(col);
                Host.Assert(type.IsKey || type.IsNumber);

                if (type.IsKey)
                {
                    Host.Assert(type.KeyCount > 0);
                    labelCardinality = type.KeyCount;

                    int size = type.KeyCount;
                    ulong src = 0;
                    var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberType.U8, row, col);
                    return
                        (ref int dst) =>
                        {
                            getSrc(ref src);
                            // The value should fall between 0 and _labelCardinality inclusive, where 0 is considered
                            // missing/invalid (this is the contract of the KeyType). However, we still handle the
                            // cases of too large values correctly (by treating them as invalid).
                            if (src <= (ulong)size)
                                dst = (int)src - 1;
                            else
                                dst = -1;
                        };
                }
                else
                {
                    // REVIEW: replace with trainable binning for numeric value
                    labelCardinality = 2; // any numeric column is split into 0 and 1

                    Double src = 0;
                    var getSrc = RowCursorUtils.GetGetterAs<Double>(NumberType.R8, row, col);
                    return
                        (ref int dst) =>
                        {
                            getSrc(ref src);
                            // NaN maps to -1.
                            if (src > 0)
                                dst = 1;
                            else if (src <= 0)
                                dst = 0;
                            else
                                dst = -1;
                        };
                }
            }

            public virtual bool ProcessValue()
            {
                Host.Assert(Rem >= 0);
                if (Rem == 0)
                    return false;
                Rem--;
                int label = 0;
                _labelGetterSrc(ref label);
                var accept = label >= 0 && AcceptColumnValue(); // skip examples with negative label
                if (accept)
                    Labels.Add(label);
                return true;
            }

            public abstract IColumnFunction CreateColumnFunction();

            protected abstract bool AcceptColumnValue();
        }

        private abstract class OneColumnSupervisedBinFunctionBuilderBase<TFloat> : SupervisedBinFunctionBuilderBase
        {
            private readonly ValueGetter<TFloat> _colGetterSrc;
            protected readonly List<TFloat> ColValues;

            protected OneColumnSupervisedBinFunctionBuilderBase(IHost host, long lim, int valueColId, int labelColId,
                IRow dataRow)
                : base(host, lim, labelColId, dataRow)
            {
                _colGetterSrc = dataRow.GetGetter<TFloat>(valueColId);
                ColValues = new List<TFloat>();
            }

            protected override bool AcceptColumnValue()
            {
                TFloat colValue = default(TFloat);
                _colGetterSrc(ref colValue);
                var result = AcceptColumnValue(ref colValue);
                if (result)
                    ColValues.Add(colValue);
                return result;
            }

            protected abstract bool AcceptColumnValue(ref TFloat colValue);
        }

        private abstract class VecColumnSupervisedBinFunctionBuilderBase<TFloat> : SupervisedBinFunctionBuilderBase
        {
            private readonly ValueGetter<VBuffer<TFloat>> _colValueGetter;
            private VBuffer<TFloat> _buffer;

            protected readonly List<TFloat>[] ColValues;
            protected readonly int ColumnSlotCount;

            protected VecColumnSupervisedBinFunctionBuilderBase(IHost host, long lim, int valueColId, int labelColId, IRow dataRow)
                : base(host, lim, labelColId, dataRow)
            {
                _colValueGetter = dataRow.GetGetter<VBuffer<TFloat>>(valueColId);
                var valueColType = dataRow.Schema.GetColumnType(valueColId);
                Host.Assert(valueColType.IsKnownSizeVector);
                ColumnSlotCount = valueColType.ValueCount;

                ColValues = new List<TFloat>[ColumnSlotCount];
                for (int i = 0; i < ColumnSlotCount; i++)
                    ColValues[i] = new List<TFloat>();

                _buffer = default(VBuffer<TFloat>);
            }

            protected override bool AcceptColumnValue()
            {
                _colValueGetter(ref _buffer);
                bool result = AcceptColumnValue(ref _buffer);
                if (result)
                {
                    if (_buffer.IsDense)
                    {
                        var values = _buffer.Values;
                        for (int i = 0; i < ColumnSlotCount; i++)
                            ColValues[i].Add(values[i]);
                    }
                    else
                    {
                        var indices = _buffer.Indices;
                        var values = _buffer.Values;
                        int k = 0;
                        for (int i = 0; i < _buffer.Count; i++)
                        {
                            var val = values[i];
                            var index = indices[i];
                            while (k < index)
                                ColValues[k++].Add(default(TFloat));

                            ColValues[k++].Add(val);
                        }

                        while (k < ColumnSlotCount)
                            ColValues[k++].Add(default(TFloat));
                    }
                }
                return result;
            }

            protected abstract bool AcceptColumnValue(ref VBuffer<TFloat> buffer);
        }

        private static partial class MinMaxUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(MinMaxArguments args, IHost host,
                int icol, int srcIndex, ColumnType srcType, IRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                if (srcType.IsNumber)
                {
                    if (srcType == NumberType.R4)
                        return Sng.MinMaxOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Single>(srcIndex));
                    if (srcType == NumberType.R8)
                        return Dbl.MinMaxOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Double>(srcIndex));
                }
                if (srcType.IsVector && srcType.ItemType.IsNumber)
                {
                    if (srcType.ItemType == NumberType.R4)
                        return Sng.MinMaxVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Single>>(srcIndex));
                    if (srcType.ItemType == NumberType.R8)
                        return Dbl.MinMaxVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Double>>(srcIndex));
                }
                throw host.ExceptUserArg(nameof(args.Column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.", args.Column[icol].Source, srcType.ToString());
            }
        }

        private static partial class MeanVarUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(MeanVarArguments args, IHost host,
                int icol, int srcIndex, ColumnType srcType, IRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                if (srcType.IsNumber)
                {
                    if (srcType == NumberType.R4)
                        return Sng.MeanVarOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Single>(srcIndex));
                    if (srcType == NumberType.R8)
                        return Dbl.MeanVarOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Double>(srcIndex));
                }
                if (srcType.IsVector && srcType.ItemType.IsNumber)
                {
                    if (srcType.ItemType == NumberType.R4)
                        return Sng.MeanVarVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Single>>(srcIndex));
                    if (srcType.ItemType == NumberType.R8)
                        return Dbl.MeanVarVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Double>>(srcIndex));
                }
                throw host.ExceptUserArg(nameof(args.Column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.", args.Column[icol].Source, srcType.ToString());
            }
        }

        private static partial class LogMeanVarUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(LogMeanVarArguments args, IHost host,
                int icol, int srcIndex, ColumnType srcType, IRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                if (srcType.IsNumber)
                {
                    if (srcType == NumberType.R4)
                        return Sng.MeanVarOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Single>(srcIndex));
                    if (srcType == NumberType.R8)
                        return Dbl.MeanVarOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Double>(srcIndex));
                }
                if (srcType.IsVector && srcType.ItemType.IsNumber)
                {
                    if (srcType.ItemType == NumberType.R4)
                        return Sng.MeanVarVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Single>>(srcIndex));
                    if (srcType.ItemType == NumberType.R8)
                        return Dbl.MeanVarVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Double>>(srcIndex));
                }
                throw host.ExceptUserArg(nameof(args.Column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.", args.Column[icol].Source, srcType.ToString());
            }
        }

        private static partial class BinUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(BinArguments args, IHost host,
                int icol, int srcIndex, ColumnType srcType, IRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                if (srcType.IsNumber)
                {
                    if (srcType == NumberType.R4)
                        return Sng.BinOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Single>(srcIndex));
                    if (srcType == NumberType.R8)
                        return Dbl.BinOneColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<Double>(srcIndex));
                }
                if (srcType.IsVector && srcType.ItemType.IsNumber)
                {
                    if (srcType.ItemType == NumberType.R4)
                        return Sng.BinVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Single>>(srcIndex));
                    if (srcType.ItemType == NumberType.R8)
                        return Dbl.BinVecColumnFunctionBuilder.Create(args, host, icol, srcType, cursor.GetGetter<VBuffer<Double>>(srcIndex));
                }
                throw host.ExceptUserArg(nameof(args.Column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.", args.Column[icol].Source, srcType.ToString());
            }
        }

        private static class SupervisedBinUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(SupervisedBinArguments args, IHost host,
                int icol, int srcIndex, ColumnType srcType, IRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                // checking for label column
                host.CheckUserArg(!string.IsNullOrWhiteSpace(args.LabelColumn), nameof(args.LabelColumn), "Must specify the label column name");
                int labelColumnId = GetLabelColumnId(host, cursor.Schema, args.LabelColumn);
                var labelColumnType = cursor.Schema.GetColumnType(labelColumnId);
                if (labelColumnType.IsKey)
                    host.CheckUserArg(labelColumnType.KeyCount > 0, nameof(args.LabelColumn), "Label column must have a known cardinality");
                else
                    host.CheckUserArg(labelColumnType.IsNumber, nameof(args.LabelColumn), "Label column must be a number or a key type");

                if (srcType.IsNumber)
                {
                    if (srcType == NumberType.R4)
                        return Sng.SupervisedBinOneColumnFunctionBuilder.Create(args, host, icol, srcIndex, labelColumnId, cursor);
                    if (srcType == NumberType.R8)
                        return Dbl.SupervisedBinOneColumnFunctionBuilder.Create(args, host, icol, srcIndex, labelColumnId, cursor);
                }
                if (srcType.IsVector && srcType.ItemType.IsNumber)
                {
                    if (srcType.ItemType == NumberType.R4)
                        return Sng.SupervisedBinVecColumnFunctionBuilder.Create(args, host, icol, srcIndex, labelColumnId, cursor);
                    if (srcType.ItemType == NumberType.R8)
                        return Dbl.SupervisedBinVecColumnFunctionBuilder.Create(args, host, icol, srcIndex, labelColumnId, cursor);
                }

                throw host.ExceptUserArg(nameof(args.Column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.", args.Column[icol].Source, srcType.ToString());
            }

            public static int GetLabelColumnId(IExceptionContext host, ISchema schema, string labelColumnName)
            {
                Contracts.AssertValue(host);
                host.AssertValue(schema);
                int labelColumnId;
                if (!schema.TryGetColumnIndex(labelColumnName, out labelColumnId))
                    throw host.ExceptUserArg(nameof(SupervisedBinArguments.LabelColumn), "Label column '{0}' not found", labelColumnName);
                return labelColumnId;
            }
        }
    }

    public static partial class AffineNormSerializationUtils
    {
        public const string LoaderSignature = "AffineNormExec";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "AFF NORM",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Sparse representation
                verWrittenCur: 0x00010003, // Scales multiply instead of divide
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature);
        }
    }

    public static partial class BinNormSerializationUtils
    {
        public const string LoaderSignature = "BinNormExec";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BIN NORM",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }
    }

    internal static class MeanVarUtils
    {
        internal static void AdjustForZeros(ref Double mean, ref Double m2, ref long count, long numZeros)
        {
            Contracts.Assert(m2 >= 0);
            Contracts.Assert(count >= 0);
            Contracts.Assert(numZeros >= 0);

            if (numZeros == 0)
                return;
            count += numZeros;
            var delta = 0 - mean;
            mean += delta * numZeros / count;
            var d2 = delta * (0 - mean);
            Contracts.Assert(d2 >= 0);
            m2 += d2 * numZeros;
            Contracts.Assert(m2 >= 0);
        }
    }
}
