// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(NormalizeTransform.MinMaxNormalizerSummary, typeof(IDataTransform), typeof(NormalizeTransform), typeof(NormalizeTransform.MinMaxArguments), typeof(SignatureDataTransform),
    NormalizeTransform.MinMaxNormalizerUserName, "MinMaxNormalizer", NormalizeTransform.MinMaxNormalizerShortName)]

[assembly: LoadableClass(NormalizeTransform.MeanVarNormalizerSummary, typeof(IDataTransform), typeof(NormalizeTransform), typeof(NormalizeTransform.MeanVarArguments), typeof(SignatureDataTransform),
    NormalizeTransform.MeanVarNormalizerUserName, "MeanVarNormalizer", NormalizeTransform.MeanVarNormalizerShortName, "ZScoreNormalizer", "ZScore", "GaussianNormalizer", "Gaussian")]

[assembly: LoadableClass(NormalizeTransform.LogMeanVarNormalizerSummary, typeof(IDataTransform), typeof(NormalizeTransform), typeof(NormalizeTransform.LogMeanVarArguments), typeof(SignatureDataTransform),
    NormalizeTransform.LogMeanVarNormalizerUserName, "LogMeanVarNormalizer", NormalizeTransform.LogMeanVarNormalizerShortName, "LogNormalNormalizer", "LogNormal")]

[assembly: LoadableClass(NormalizeTransform.BinNormalizerSummary, typeof(IDataTransform), typeof(NormalizeTransform), typeof(NormalizeTransform.BinArguments), typeof(SignatureDataTransform),
    NormalizeTransform.BinNormalizerUserName, "BinNormalizer", NormalizeTransform.BinNormalizerShortName)]

[assembly: LoadableClass(typeof(NormalizeTransform.AffineColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "Affine Normalizer", AffineNormSerializationUtils.LoaderSignature)]

[assembly: LoadableClass(typeof(NormalizeTransform.CdfColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "CDF Normalizer", NormalizeTransform.CdfColumnFunction.LoaderSignature)]

[assembly: LoadableClass(NormalizeTransform.BinNormalizerSummary, typeof(NormalizeTransform.BinColumnFunction), null, typeof(SignatureLoadColumnFunction),
    "Bin Normalizer", NormalizeTransform.BinColumnFunction.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The normalize transform for support of normalization via the <see cref="IDataTransform"/> mechanism.
    /// More contemporaneous API usage of normalization ought to use <see cref="NormalizingEstimator"/>
    /// and <see cref="NormalizingTransformer"/> rather than this structure.
    /// </summary>
    internal sealed partial class NormalizeTransform
    {
        public abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of examples used to train the normalizer",
                Name = "MaxTrainingExamples", ShortName = "maxtrain")]
            public long? MaximumExampleCount;

            private protected ColumnBase()
            {
            }

            private protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (MaximumExampleCount != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        // REVIEW: Support different aggregators on different columns, eg, MinMax vs Variance/ZScore.
        public abstract class ControlZeroColumnBase : ColumnBase
        {
            // REVIEW: This only allows mapping either zero or min to zero. It might make sense to allow also max, midpoint and mean to be mapped to zero.
            [Argument(ArgumentType.AtMostOnce, Name="FixZero", HelpText = "Whether to map zero to zero, preserving sparsity", ShortName = "zero")]
            public bool? EnsureZeroUntouched;

            private protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (EnsureZeroUntouched != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        public sealed class AffineColumn : ControlZeroColumnBase
        {
            internal static AffineColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new AffineColumn();
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

        public sealed class BinColumn : ControlZeroColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bins, power of 2 recommended", ShortName = "bins")]
            [TGUI(Label = "Max number of bins")]
            public int? NumBins;

            internal static BinColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new BinColumn();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NumBins != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class LogNormalColumn : ColumnBase
        {
            internal static LogNormalColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new LogNormalColumn();
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

        private static class Defaults
        {
            public const bool EnsureZeroUntouched = true;
            public const bool MeanVarCdf = false;
            public const bool LogMeanVarCdf = true;
            public const int NumBins = 1024;
            public const int MinBinSize = 10;
        }

        public abstract class ControlZeroArgumentsBase : ArgumentsBase
        {
            // REVIEW: This only allows mapping either zero or min to zero. It might make sense to allow also max, midpoint and mean to be mapped to zero.
            // REVIEW: Convert this to bool? or even an enum{Auto, No, Yes}, and automatically map zero to zero when it is null/Auto.
            [Argument(ArgumentType.AtMostOnce, Name = "FixZero", HelpText = "Whether to map zero to zero, preserving sparsity", ShortName = "zero")]
            public bool EnsureZeroUntouched = Defaults.EnsureZeroUntouched;
        }

        public abstract class AffineArgumentsBase : ControlZeroArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public AffineColumn[] Columns;

            public override OneToOneColumn[] GetColumns() => Columns;
        }

        public sealed class MinMaxArguments : AffineArgumentsBase
        {
        }

        public sealed class MeanVarArguments : AffineArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use CDF as the output", ShortName = "cdf")]
            public bool UseCdf = Defaults.MeanVarCdf;
        }

        public abstract class ArgumentsBase : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of examples used to train the normalizer",
                Name = "MaxTrainingExamples", ShortName = "maxtrain")]
            public long MaximumExampleCount = 1000000000;

            public abstract OneToOneColumn[] GetColumns();

            public string TestType(DataViewType type)
            {
                DataViewType itemType = type;
                if (type is VectorType vectorType)
                {
                    // We require vectors to be of known size.
                    if (!vectorType.IsKnownSize)
                        return "Expected known size vector";

                    itemType = vectorType.ItemType;
                }

                if (itemType != NumberDataViewType.Single && itemType != NumberDataViewType.Double)
                    return "Expected R4 or R8 item type";

                return null;
            }
        }

        public sealed class LogMeanVarArguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use CDF as the output", ShortName = "cdf")]
            public bool UseCdf = Defaults.LogMeanVarCdf;

            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public LogNormalColumn[] Columns;

            public override OneToOneColumn[] GetColumns() => Columns;
        }

        public abstract class BinArgumentsBase : ControlZeroArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public BinColumn[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bins, power of 2 recommended", ShortName = "bins")]
            [TGUI(Label = "Max number of bins")]
            public int NumBins = Defaults.NumBins;

            public override OneToOneColumn[] GetColumns() => Columns;
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

        internal const string MinMaxNormalizerSummary = "Normalizes the data based on the observed minimum and maximum values of the data.";
        internal const string MeanVarNormalizerSummary = "Normalizes the data based on the computed mean and variance of the data.";
        internal const string LogMeanVarNormalizerSummary = "Normalizes the data based on the computed mean and variance of the logarithm of the data.";
        internal const string BinNormalizerSummary = "The values are assigned into equidensity bins and a value is mapped to its bin_number/number_of_bins.";
        internal const string SupervisedBinNormalizerSummary = "Similar to BinNormalizer, but calculates bins based on correlation with the label column, not equi-density. "
            + "The new value is bin_number / number_of_bins.";

        internal const string MinMaxNormalizerUserName = "Min-Max Normalizer";
        internal const string MeanVarNormalizerUserName = "MeanVar Normalizer";
        internal const string LogMeanVarNormalizerUserName = "LogMeanVar Normalizer";
        internal const string BinNormalizerUserName = "Binning Normalizer";
        internal const string SupervisedBinNormalizerUserName = "Supervised Binning Normalizer";

        internal const string MinMaxNormalizerShortName = "MinMax";
        internal const string MeanVarNormalizerShortName = "MeanVar";
        internal const string LogMeanVarNormalizerShortName = "LogMeanVar";
        internal const string BinNormalizerShortName = "Bin";
        internal const string SupervisedBinNormalizerShortName = "SupBin";

        /// <summary>
        /// A helper method to create a MinMax normalizer.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="outputColumnName">Name of the output column.</param>
        /// <param name="inputColumnName">Name of the column to be transformed. If this is null '<paramref name="outputColumnName"/>' will be used.</param>
        public static IDataView CreateMinMaxNormalizer(IHostEnvironment env, IDataView input, string outputColumnName, string inputColumnName = null)
        {
            Contracts.CheckValue(env, nameof(env));

            var normalizer = new NormalizingEstimator(env, new NormalizingEstimator.MinMaxColumnOptions(outputColumnName, inputColumnName ?? outputColumnName));
            return normalizer.Fit(input).MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method corresponding to SignatureDataTransform.
        /// </summary>
        internal static IDataTransform Create(IHostEnvironment env, MinMaxArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = args.Columns
                .Select(col => new NormalizingEstimator.MinMaxColumnOptions(
                    col.Name,
                    col.Source ?? col.Name,
                    col.MaximumExampleCount ?? args.MaximumExampleCount,
                    col.EnsureZeroUntouched ?? args.EnsureZeroUntouched))
                .ToArray();
            var normalizer = new NormalizingEstimator(env, columns);
            return normalizer.Fit(input).MakeDataTransform(input);
        }

        // Factory method corresponding to SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, MeanVarArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = args.Columns
                .Select(col => new NormalizingEstimator.MeanVarianceColumnOptions(
                    col.Name,
                    col.Source ?? col.Name,
                    col.MaximumExampleCount ?? args.MaximumExampleCount,
                    col.EnsureZeroUntouched ?? args.EnsureZeroUntouched))
                .ToArray();
            var normalizer = new NormalizingEstimator(env, columns);
            return normalizer.Fit(input).MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method corresponding to SignatureDataTransform.
        /// </summary>
        internal static IDataTransform Create(IHostEnvironment env, LogMeanVarArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = args.Columns
                .Select(col => new NormalizingEstimator.LogMeanVarianceColumnOptions(
                    col.Name,
                    col.Source ?? col.Name,
                    col.MaximumExampleCount ?? args.MaximumExampleCount,
                    args.UseCdf))
                .ToArray();
            var normalizer = new NormalizingEstimator(env, columns);
            return normalizer.Fit(input).MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method corresponding to SignatureDataTransform.
        /// </summary>
        internal static IDataTransform Create(IHostEnvironment env, BinArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = args.Columns
                .Select(col => new NormalizingEstimator.BinningColumnOptions(
                    col.Name,
                    col.Source ?? col.Name,
                    col.MaximumExampleCount ?? args.MaximumExampleCount,
                    col.EnsureZeroUntouched ?? args.EnsureZeroUntouched,
                    col.NumBins ?? args.NumBins))
                .ToArray();
            var normalizer = new NormalizingEstimator(env, columns);
            return normalizer.Fit(input).MakeDataTransform(input);
        }

        internal abstract partial class AffineColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            // The only derived classes are private inner classes
            private AffineColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

            private protected abstract void SaveModel(ModelSaveContext ctx);

            public abstract JToken PfaInfo(BoundPfaContext ctx, JToken srcToken);
            public bool CanSaveOnnx(OnnxContext ctx) => true;
            public abstract bool OnnxInfo(OnnxContext ctx, OnnxNode nodeProtoWrapper, int featureCount);

            public abstract Delegate GetGetter(DataViewRow input, int icol);

            public abstract void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc);

            public abstract NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams();

            public static AffineColumnFunction Create(ModelLoadContext ctx, IHost host, DataViewType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc is NumberDataViewType)
                {
                    if (typeSrc == NumberDataViewType.Single)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberDataViewType.Double)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                else if (typeSrc is VectorType vectorType && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.ImplVec.Create(ctx, host, vectorType);
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.ImplVec.Create(ctx, host, vectorType);
                }
                throw host.ExceptUserArg(nameof(AffineArgumentsBase.Columns), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc.ToString());
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

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.RawType == typeof(TFloat));
                    bldr.AddPrimitive("AffineScale", typeSrc, Scale);
                    bldr.AddPrimitive("AffineOffset", typeSrc, Offset);
                }

                public override NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams()
                    => new NormalizingTransformer.AffineNormalizerModelParameters<TFloat>(Scale, Offset);

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

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.GetVectorSize() == Scale.Length);
                    Host.Check(typeSrc.GetItemType().RawType == typeof(TFloat));
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

                public override NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams()
                    => new NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<TFloat>>(ImmutableArray.Create(Scale), ImmutableArray.Create(Offset));
            }
        }

        internal abstract partial class CdfColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            // The only derived classes are private inner classes
            private CdfColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

            private protected abstract void SaveModel(ModelSaveContext ctx);

            public JToken PfaInfo(BoundPfaContext ctx, JToken srcToken) => null;

            public bool CanSaveOnnx(OnnxContext ctx) => false;

            public bool OnnxInfo(OnnxContext ctx, OnnxNode nodeProtoWrapper, int featureCount)
                => throw Host.ExceptNotSupp();

            public abstract Delegate GetGetter(DataViewRow input, int icol);
            public abstract void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc);
            public abstract NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams();

            public static CdfColumnFunction Create(ModelLoadContext ctx, IHost host, DataViewType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc is NumberDataViewType)
                {
                    if (typeSrc == NumberDataViewType.Single)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberDataViewType.Double)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                else if (typeSrc is VectorType vectorType && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.ImplVec.Create(ctx, host, vectorType);
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.ImplVec.Create(ctx, host, vectorType);
                }
                throw host.ExceptUserArg(nameof(AffineArgumentsBase.Columns), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc);
            }

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

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.RawType == typeof(TFloat));
                    bldr.AddPrimitive("CdfMean", typeSrc, Mean);
                    bldr.AddPrimitive("CdfStdDev", typeSrc, Stddev);
                    bldr.AddPrimitive("CdfUseLog", BooleanDataViewType.Instance, UseLog);
                }

                public override NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams()
                    => new NormalizingTransformer.CdfNormalizerModelParameters<TFloat>(Mean, Stddev, UseLog);
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

                public override void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc)
                {
                    Host.CheckValue(bldr, nameof(bldr));
                    Host.CheckValue(typeSrc, nameof(typeSrc));
                    Host.Check(typeSrc.GetVectorSize() == Mean.Length);
                    Host.Check(typeSrc.GetItemType().RawType == typeof(TFloat));
                    bldr.AddGetter<VBuffer<TFloat>>("CdfMean", typeSrc, MeanMetadataGetter);
                    bldr.AddGetter<VBuffer<TFloat>>("CdfStdDev", typeSrc, StddevMetadataGetter);
                    bldr.AddPrimitive("CdfUseLog", BooleanDataViewType.Instance, UseLog);
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

                public override NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams()
                    => new NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<TFloat>>(ImmutableArray.Create(Mean), ImmutableArray.Create(Stddev), UseLog);
            }

            public const string LoaderSignature = "CdfNormalizeFunction";

            public static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "CDFNORMF",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CdfColumnFunction).Assembly.FullName);
            }
        }

        internal abstract partial class BinColumnFunction : IColumnFunction
        {
            protected readonly IHost Host;

            protected BinColumnFunction(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

            private protected abstract void SaveModel(ModelSaveContext ctx);

            public JToken PfaInfo(BoundPfaContext ctx, JToken srcToken) => null;

            public bool CanSaveOnnx(OnnxContext ctx) => false;

            public bool OnnxInfo(OnnxContext ctx, OnnxNode nodeProtoWrapper, int featureCount)
                => throw Host.ExceptNotSupp();

            public abstract Delegate GetGetter(DataViewRow input, int icol);

            public void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc)
            {
                // REVIEW: How to attach information on the bins, to metadata?
            }

            public abstract NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams();

            public static BinColumnFunction Create(ModelLoadContext ctx, IHost host, DataViewType typeSrc)
            {
                Contracts.CheckValue(host, nameof(host));
                if (typeSrc is NumberDataViewType)
                {
                    if (typeSrc == NumberDataViewType.Single)
                        return Sng.ImplOne.Create(ctx, host, typeSrc);
                    if (typeSrc == NumberDataViewType.Double)
                        return Dbl.ImplOne.Create(ctx, host, typeSrc);
                }
                if (typeSrc is VectorType vectorType && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.ImplVec.Create(ctx, host, vectorType);
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.ImplVec.Create(ctx, host, vectorType);
                }
                throw host.ExceptUserArg(nameof(BinArguments.Columns), "Wrong column type. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {0}.", typeSrc);
            }

            public const string LoaderSignature = "BinNormalizeFunction";

            public static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "BINNORMF",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinColumnFunction).Assembly.FullName);
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
                return ProcessValue(in tmp);
            }

            protected virtual bool ProcessValue(in TFloat val)
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
                return ProcessValue(in _buffer);
            }

            protected virtual bool ProcessValue(in VBuffer<TFloat> buffer)
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

            protected SupervisedBinFunctionBuilderBase(IHost host, long lim, int labelColId, DataViewRow dataRow)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
                Rem = lim;
                Lim = lim;
                Labels = new List<int>();
                _labelGetterSrc = GetLabelGetter(dataRow, labelColId, out LabelCardinality);
            }

            private ValueGetter<int> GetLabelGetter(DataViewRow row, int col, out int labelCardinality)
            {
                // The label column type is checked as part of args validation.
                var type = row.Schema[col].Type;
                Host.Assert(type is KeyType || type is NumberDataViewType);

                if (type is KeyType keyType)
                {
                    Host.Assert(type.GetKeyCountAsInt32(Host) > 0);
                    labelCardinality = type.GetKeyCountAsInt32(Host);

                    int size = type.GetKeyCountAsInt32(Host);
                    ulong src = 0;
                    var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberDataViewType.UInt64, row, col);
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
                    var getSrc = RowCursorUtils.GetGetterAs<Double>(NumberDataViewType.Double, row, col);
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
                DataViewRow dataRow)
                : base(host, lim, labelColId, dataRow)
            {
                _colGetterSrc = dataRow.GetGetter<TFloat>(dataRow.Schema[valueColId]);
                ColValues = new List<TFloat>();
            }

            protected override bool AcceptColumnValue()
            {
                TFloat colValue = default(TFloat);
                _colGetterSrc(ref colValue);
                var result = AcceptColumnValue(in colValue);
                if (result)
                    ColValues.Add(colValue);
                return result;
            }

            protected abstract bool AcceptColumnValue(in TFloat colValue);
        }

        private abstract class VecColumnSupervisedBinFunctionBuilderBase<TFloat> : SupervisedBinFunctionBuilderBase
        {
            private readonly ValueGetter<VBuffer<TFloat>> _colValueGetter;
            private VBuffer<TFloat> _buffer;

            protected readonly List<TFloat>[] ColValues;
            protected readonly int ColumnSlotCount;

            protected VecColumnSupervisedBinFunctionBuilderBase(IHost host, long lim, int valueColId, int labelColId, DataViewRow dataRow)
                : base(host, lim, labelColId, dataRow)
            {
                var valueCol = dataRow.Schema[valueColId];
                _colValueGetter = dataRow.GetGetter<VBuffer<TFloat>>(valueCol);

                Host.Assert(valueCol.Type.IsKnownSizeVector());
                ColumnSlotCount = valueCol.Type.GetValueCount();

                ColValues = new List<TFloat>[ColumnSlotCount];
                for (int i = 0; i < ColumnSlotCount; i++)
                    ColValues[i] = new List<TFloat>();

                _buffer = default(VBuffer<TFloat>);
            }

            protected override bool AcceptColumnValue()
            {
                _colValueGetter(ref _buffer);
                bool result = AcceptColumnValue(in _buffer);
                if (result)
                {
                    if (_buffer.IsDense)
                    {
                        var values = _buffer.GetValues();
                        for (int i = 0; i < ColumnSlotCount; i++)
                            ColValues[i].Add(values[i]);
                    }
                    else
                    {
                        var indices = _buffer.GetIndices();
                        var values = _buffer.GetValues();
                        int k = 0;
                        for (int i = 0; i < values.Length; i++)
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

            protected abstract bool AcceptColumnValue(in VBuffer<TFloat> buffer);
        }

        internal static partial class MinMaxUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(MinMaxArguments args, IHost host,
                int icol, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                return CreateBuilder(new NormalizingEstimator.MinMaxColumnOptions(
                    args.Columns[icol].Name,
                    args.Columns[icol].Source ?? args.Columns[icol].Name,
                    args.Columns[icol].MaximumExampleCount ?? args.MaximumExampleCount,
                    args.Columns[icol].EnsureZeroUntouched ?? args.EnsureZeroUntouched), host, srcIndex, srcType, cursor);
            }

            public static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.MinMaxColumnOptions column, IHost host,
                int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                var srcColumn = cursor.Schema[srcIndex];
                if (srcType is NumberDataViewType)
                {
                    if (srcType == NumberDataViewType.Single)
                        return Sng.MinMaxOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Single>(srcColumn));
                    if (srcType == NumberDataViewType.Double)
                        return Dbl.MinMaxOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Double>(srcColumn));
                }
                if (srcType is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.MinMaxVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Single>>(srcColumn));
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.MinMaxVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Double>>(srcColumn));
                }
                throw host.ExceptParam(nameof(srcType), "Wrong column type for input column. Expected: float, double, Vec<float, n> or Vec<double, n>. Got: {0}.", srcType.ToString());
            }
        }

        internal static partial class MeanVarUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(MeanVarArguments args, IHost host,
                int icol, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                return CreateBuilder(new NormalizingEstimator.MeanVarianceColumnOptions(
                    args.Columns[icol].Name,
                    args.Columns[icol].Source ?? args.Columns[icol].Name,
                    args.Columns[icol].MaximumExampleCount ?? args.MaximumExampleCount,
                    args.Columns[icol].EnsureZeroUntouched ?? args.EnsureZeroUntouched,
                    args.UseCdf), host, srcIndex, srcType, cursor);
            }

            public static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.MeanVarianceColumnOptions column, IHost host,
                int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                var srcColumn = cursor.Schema[srcIndex];
                if (srcType is NumberDataViewType)
                {
                    if (srcType == NumberDataViewType.Single)
                        return Sng.MeanVarOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Single>(srcColumn));
                    if (srcType == NumberDataViewType.Double)
                        return Dbl.MeanVarOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Double>(srcColumn));
                }
                if (srcType is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.MeanVarVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Single>>(srcColumn));
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.MeanVarVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Double>>(srcColumn));
                }
                throw host.ExceptParam(nameof(srcType), "Wrong column type for input column. Expected: float, double, Vec<float, n> or Vec<double, n>. Got: {0}.", srcType.ToString());
            }

        }

        internal static partial class LogMeanVarUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(LogMeanVarArguments args, IHost host,
                int icol, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                return CreateBuilder(new NormalizingEstimator.LogMeanVarianceColumnOptions(
                    args.Columns[icol].Name,
                    args.Columns[icol].Source ?? args.Columns[icol].Name,
                    args.Columns[icol].MaximumExampleCount ?? args.MaximumExampleCount,
                    args.UseCdf), host, srcIndex, srcType, cursor);
            }

            public static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.LogMeanVarianceColumnOptions column, IHost host,
                int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(column);

                var srcColumn = cursor.Schema[srcIndex];
                if (srcType is NumberDataViewType)
                {
                    if (srcType == NumberDataViewType.Single)
                        return Sng.MeanVarOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Single>(srcColumn));
                    if (srcType == NumberDataViewType.Double)
                        return Dbl.MeanVarOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Double>(srcColumn));
                }
                if (srcType is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.MeanVarVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Single>>(srcColumn));
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.MeanVarVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Double>>(srcColumn));
                }
                throw host.ExceptUserArg(nameof(column), "Wrong column type for column {0}. Expected: float, double, Vec<float, n> or Vec<double, n>. Got: {1}.", column.InputColumnName, srcType.ToString());
            }
        }

        internal static partial class BinUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(BinArguments args, IHost host,
                int icol, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                return CreateBuilder(new NormalizingEstimator.BinningColumnOptions(
                    args.Columns[icol].Name,
                    args.Columns[icol].Source ?? args.Columns[icol].Name,
                    args.Columns[icol].MaximumExampleCount ?? args.MaximumExampleCount,
                    args.Columns[icol].EnsureZeroUntouched ?? args.EnsureZeroUntouched,
                    args.Columns[icol].NumBins ?? args.NumBins), host, srcIndex, srcType, cursor);
            }

            public static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.BinningColumnOptions column, IHost host,
                int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);

                var srcColumn = cursor.Schema[srcIndex];
                if (srcType is NumberDataViewType)
                {
                    if (srcType == NumberDataViewType.Single)
                        return Sng.BinOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Single>(srcColumn));
                    if (srcType == NumberDataViewType.Double)
                        return Dbl.BinOneColumnFunctionBuilder.Create(column, host, srcType, cursor.GetGetter<Double>(srcColumn));
                }
                if (srcType is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.BinVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Single>>(srcColumn));
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.BinVecColumnFunctionBuilder.Create(column, host, vectorType, cursor.GetGetter<VBuffer<Double>>(srcColumn));
                }
                throw host.ExceptParam(nameof(column), "Wrong column type for column {0}. Expected: float, double, Vec<float, n> or Vec<double, n>. Got: {1}.", column.InputColumnName, srcType.ToString());
            }
        }

        internal static class SupervisedBinUtils
        {
            public static IColumnFunctionBuilder CreateBuilder(SupervisedBinArguments args, IHost host,
                int icol, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);
                host.AssertValue(args);

                // checking for label column
                host.CheckUserArg(!string.IsNullOrWhiteSpace(args.LabelColumn), nameof(args.LabelColumn), "Must specify the label column name");
                int labelColumnId = GetLabelColumnId(host, cursor.Schema, args.LabelColumn);
                var labelColumnType = cursor.Schema[labelColumnId].Type;
                if (labelColumnType is KeyType labelKeyType)
                    host.CheckUserArg(labelKeyType.Count > 0, nameof(args.LabelColumn), "Label column must have a known cardinality");
                else
                    host.CheckUserArg(labelColumnType is NumberDataViewType, nameof(args.LabelColumn), "Label column must be a number or a key type");

                return CreateBuilder(
                    new NormalizingEstimator.SupervisedBinningColumOptions(
                        args.Columns[icol].Name,
                        args.Columns[icol].Source ?? args.Columns[icol].Name,
                        args.LabelColumn ?? DefaultColumnNames.Label,
                        args.Columns[icol].MaximumExampleCount ?? args.MaximumExampleCount,
                        args.Columns[icol].EnsureZeroUntouched ?? args.EnsureZeroUntouched,
                        args.Columns[icol].NumBins ?? args.NumBins,
                        args.MinBinSize),
                    host, labelColumnId, srcIndex, srcType, cursor);
            }

            public static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.SupervisedBinningColumOptions column, IHost host,
                 string labelColumn, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                int labelColumnId = GetLabelColumnId(host, cursor.Schema, labelColumn);
                return CreateBuilder(column, host, labelColumnId, srcIndex, srcType, cursor);
            }

            private static IColumnFunctionBuilder CreateBuilder(NormalizingEstimator.SupervisedBinningColumOptions column, IHost host,
                int labelColumnId, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
            {
                Contracts.AssertValue(host);

                if (srcType is NumberDataViewType)
                {
                    if (srcType == NumberDataViewType.Single)
                        return Sng.SupervisedBinOneColumnFunctionBuilder.Create(column, host, srcIndex, labelColumnId, cursor);
                    if (srcType == NumberDataViewType.Double)
                        return Dbl.SupervisedBinOneColumnFunctionBuilder.Create(column, host, srcIndex, labelColumnId, cursor);
                }
                if (srcType is VectorType vectorType && vectorType.ItemType is NumberDataViewType)
                {
                    if (vectorType.ItemType == NumberDataViewType.Single)
                        return Sng.SupervisedBinVecColumnFunctionBuilder.Create(column, host, srcIndex, labelColumnId, cursor);
                    if (vectorType.ItemType == NumberDataViewType.Double)
                        return Dbl.SupervisedBinVecColumnFunctionBuilder.Create(column, host, srcIndex, labelColumnId, cursor);
                }

                throw host.ExceptParam(nameof(column), "Wrong column type for column {0}. Expected: R4, R8, Vec<R4, n> or Vec<R8, n>. Got: {1}.",
                    column.InputColumnName,
                    srcType.ToString());
            }

            public static int GetLabelColumnId(IExceptionContext host, DataViewSchema schema, string labelColumnName)
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

    internal static partial class AffineNormSerializationUtils
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AffineNormSerializationUtils).Assembly.FullName);
        }
    }

    internal static partial class BinNormSerializationUtils
    {
        public const string LoaderSignature = "BinNormExec";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BIN NORM",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinNormSerializationUtils).Assembly.FullName);
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
