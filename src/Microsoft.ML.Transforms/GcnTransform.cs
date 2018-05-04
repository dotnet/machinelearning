// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(LpNormNormalizerTransform.GcnSummary, typeof(LpNormNormalizerTransform), typeof(LpNormNormalizerTransform.GcnArguments), typeof(SignatureDataTransform),
    LpNormNormalizerTransform.UserNameGn, "GcnTransform", LpNormNormalizerTransform.ShortNameGn)]

[assembly: LoadableClass(LpNormNormalizerTransform.GcnSummary, typeof(LpNormNormalizerTransform), null, typeof(SignatureLoadDataTransform),
    LpNormNormalizerTransform.UserNameGn, LpNormNormalizerTransform.LoaderSignature, LpNormNormalizerTransform.LoaderSignatureOld)]

[assembly: LoadableClass(LpNormNormalizerTransform.Summary, typeof(LpNormNormalizerTransform), typeof(LpNormNormalizerTransform.Arguments), typeof(SignatureDataTransform),
    LpNormNormalizerTransform.UserNameLP, "LpNormNormalizer", LpNormNormalizerTransform.ShortNameLP)]

[assembly: EntryPointModule(typeof(LpNormalization))]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Lp-Norm (vector/row-wise) normalization transform. Has the following two set of arguments:
    /// 1- Lp-Norm normalizer arguments:
    ///    Normalize rows individually by rescaling them to unit norm (L2, L1 or LInf).
    ///    Performs the following operation on a vector X:
    ///         Y = (X - M) / D, where M is mean and D is either L2 norm, L1 norm or LInf norm.
    ///    Scaling inputs to unit norms is a common operation for text classification or clustering.
    /// 2- Global contrast normalization (GCN) arguments:
    ///    Performs the following operation on a vector X:
    ///         Y = (s * X - M) / D, where s is a scale, M is mean and D is either L2 norm or standard deviation.
    ///    Usage examples and Matlab code:
    ///    <see href="http://www.cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf"/>
    /// </summary>
    public sealed class LpNormNormalizerTransform : OneToOneTransformBase
    {
        /// <summary>
        /// The kind of unit norm vectors are rescaled to. This enumeration is serialized.
        /// </summary>
        public enum NormalizerKind : byte
        {
            L2Norm = 0,
            StdDev = 1,
            L1Norm = 2,
            LInf = 3
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public NormalizerKind NormKind = NormalizerKind.L2Norm;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 2)]
            public bool SubMean = false;
        }

        public sealed class GcnArguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public GcnColumn[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 1)]
            public bool SubMean = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm", ShortName = "useStd")]
            public bool UseStdDev = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public Float Scale = 1;
        }

        public abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing")]
            public bool? SubMean;

            protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (SubMean != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        public sealed class Column : ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public NormalizerKind? NormKind;

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
                if (NormKind != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class GcnColumn : ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm")]
            public bool? UseStdDev;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public Float? Scale;

            public static GcnColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new GcnColumn();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (UseStdDev != null || Scale != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        private sealed class ColInfoEx
        {
            public readonly bool SubtractMean;
            public readonly NormalizerKind NormKind;
            public readonly Float Scale;

            public ColInfoEx(Column col, Arguments args)
            {
                SubtractMean = col.SubMean ?? args.SubMean;
                NormKind = col.NormKind ?? args.NormKind;
                Scale = 1;
            }

            public ColInfoEx(GcnColumn col, GcnArguments args)
            {
                SubtractMean = col.SubMean ?? args.SubMean;
                NormKind = (col.UseStdDev ?? args.UseStdDev) ? NormalizerKind.StdDev : NormalizerKind.L2Norm;
                Scale = col.Scale ?? args.Scale;
                Contracts.CheckUserArg(0 < Scale && Scale < Float.PositiveInfinity, nameof(args.Scale), "scale must be a positive finite value");
            }

            public ColInfoEx(ModelLoadContext ctx, bool normKindSerialized)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // byte: subMean
                // byte: NormKind
                // Float: scale
                SubtractMean = ctx.Reader.ReadBoolByte();
                byte normKindVal = ctx.Reader.ReadByte();
                Contracts.CheckDecode(Enum.IsDefined(typeof(NormalizerKind), normKindVal));
                NormKind = (NormalizerKind)normKindVal;
                // Note: In early versions, a bool option (useStd) to whether to normalize by StdDev rather than
                // L2 norm was used. normKind was added in version=verVectorNormalizerSupported.
                // normKind was defined in a way such that the serialized boolean (0: use StdDev, 1: use L2) is
                // still valid.
                Contracts.CheckDecode(normKindSerialized ||
                    (NormKind == NormalizerKind.L2Norm || NormKind == NormalizerKind.StdDev));
                Scale = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(0 < Scale && Scale < Float.PositiveInfinity);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // byte: subMean
                // byte: NormKind
                // Float: scale
                ctx.Writer.WriteBoolByte(SubtractMean);
                ctx.Writer.Write((byte)NormKind);
                Contracts.Assert(0 < Scale && Scale < Float.PositiveInfinity);
                ctx.Writer.Write(Scale);
            }
        }

        internal const string GcnSummary = "Performs a global contrast normalization on input values: Y = (s * X - M) / D, where s is a scale, M is mean and D is "
            + "either L2 norm or standard deviation.";
        internal const string UserNameGn = "Global Contrast Normalization Transform";
        internal const string ShortNameGn = "Gcn";

        internal const string Summary = "Normalize vectors (rows) individually by rescaling them to unit norm (L2, L1 or LInf). Performs the following operation on a vector X: "
            + "Y = (X - M) / D, where M is mean and D is either L2 norm, L1 norm or LInf norm.";

        internal const string UserNameLP = "Lp-Norm Normalizer";
        internal const string ShortNameLP = "lpnorm";

        private const uint VerVectorNormalizerSupported = 0x00010002;

        public const string LoaderSignature = "GcnTransform";
        internal const string LoaderSignatureOld = "GcnFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GCNORMAF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added arguments for Lp-norm (vector/row-wise) normalizer
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld);
        }

        private const string RegistrationName = "LpNormNormalizer";

        // REVIEW: should this be an argument instead?
        private const Float MinScale = (Float)1e-8;

        private readonly ColInfoEx[] _exes;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public LpNormNormalizerTransform(IHostEnvironment env, GcnArguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column,
                input, TestIsFloatVector)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(args.Column[i], args);

            // REVIEW: for now check only global (default) values. Move to Bindings/ColInfoEx?
            if (!args.SubMean && args.UseStdDev)
            {
                using (var ch = Host.Start("Argument validation"))
                {
                    ch.Warning("subMean parameter is false while useStd is true. It is advisable to set subMean to true in case useStd is set to true.");
                    ch.Done();
                }
            }
            SetMetadata();
        }

        public LpNormNormalizerTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column,
                input, TestIsFloatVector)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(args.Column[i], args);
            SetMetadata();
        }

        private LpNormNormalizerTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsFloatItem)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // foreach added column
            //   ColInfoEx

            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(ctx, ctx.Header.ModelVerWritten >= VerVectorNormalizerSupported);
            SetMetadata();
        }

        public static LpNormNormalizerTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Float));
                    return new LpNormNormalizerTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // foreach added column
            //   ColInfoEx
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);
            Host.Assert(_exes.Length == Infos.Length);
            for (int i = 0; i < _exes.Length; i++)
                _exes[i].Save(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Infos.Length);
            return Infos[iinfo].TypeSrc;
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source, MetadataUtils.Kinds.SlotNames))
                    bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, DvBool.True);
            }
            md.Seal();
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            var ex = _exes[iinfo];
            Host.Assert(0 < ex.Scale && ex.Scale < Float.PositiveInfinity);
            Host.Assert(info.TypeSrc.IsVector);

            var getSrc = GetSrcGetter<VBuffer<Float>>(input, iinfo);
            var src = default(VBuffer<Float>);
            ValueGetter<VBuffer<Float>> del;
            Float scale = ex.Scale;

            if (ex.SubtractMean)
            {
                switch (ex.NormKind)
                {
                case NormalizerKind.StdDev:
                    del =
                        (ref VBuffer<Float> dst) =>
                        {
                            getSrc(ref src);
                            Float mean = Mean(src.Values, src.Count, src.Length);
                            Float divisor = StdDev(src.Values, src.Count, src.Length, mean);
                            FillValues(Host, ref src, ref dst, divisor, scale, mean);
                        };
                    return del;
                case NormalizerKind.L2Norm:
                    del =
                       (ref VBuffer<Float> dst) =>
                       {
                           getSrc(ref src);
                           Float mean = Mean(src.Values, src.Count, src.Length);
                           Float divisor = L2Norm(src.Values, src.Count, mean);
                           FillValues(Host, ref src, ref dst, divisor, scale, mean);
                       };
                    return del;
                case NormalizerKind.L1Norm:
                    del =
                       (ref VBuffer<Float> dst) =>
                       {
                           getSrc(ref src);
                           Float mean = Mean(src.Values, src.Count, src.Length);
                           Float divisor = L1Norm(src.Values, src.Count, mean);
                           FillValues(Host, ref src, ref dst, divisor, scale, mean);
                       };
                    return del;
                case NormalizerKind.LInf:
                    del =
                       (ref VBuffer<Float> dst) =>
                       {
                           getSrc(ref src);
                           Float mean = Mean(src.Values, src.Count, src.Length);
                           Float divisor = LInfNorm(src.Values, src.Count, mean);
                           FillValues(Host, ref src, ref dst, divisor, scale, mean);
                       };
                    return del;
                default:
                    Host.Assert(false, "Unsupported normalizer type");
                    goto case NormalizerKind.L2Norm;
                }
            }

            switch (ex.NormKind)
            {
            case NormalizerKind.StdDev:
                del =
                    (ref VBuffer<Float> dst) =>
                    {
                        getSrc(ref src);
                        Float divisor = StdDev(src.Values, src.Count, src.Length);
                        FillValues(Host, ref src, ref dst, divisor, scale);
                    };
                return del;
            case NormalizerKind.L2Norm:
                del =
                   (ref VBuffer<Float> dst) =>
                   {
                       getSrc(ref src);
                       Float divisor = L2Norm(src.Values, src.Count);
                       FillValues(Host, ref src, ref dst, divisor, scale);
                   };
                return del;
            case NormalizerKind.L1Norm:
                del =
                   (ref VBuffer<Float> dst) =>
                   {
                       getSrc(ref src);
                       Float divisor = L1Norm(src.Values, src.Count);
                       FillValues(Host, ref src, ref dst, divisor, scale);
                   };
                return del;
            case NormalizerKind.LInf:
                del =
                   (ref VBuffer<Float> dst) =>
                   {
                       getSrc(ref src);
                       Float divisor = LInfNorm(src.Values, src.Count);
                       FillValues(Host, ref src, ref dst, divisor, scale);
                   };
                return del;
            default:
                Host.Assert(false, "Unsupported normalizer type");
                goto case NormalizerKind.L2Norm;
            }
        }

        private static void FillValues(IExceptionContext ectx, ref VBuffer<Float> src, ref VBuffer<Float> dst, Float divisor, Float scale, Float offset = 0)
        {
            int count = src.Count;
            int length = src.Length;
            ectx.Assert(Utils.Size(src.Values) >= count);
            ectx.Assert(divisor >= 0);

            if (count == 0)
            {
                dst = new VBuffer<Float>(length, 0, dst.Values, dst.Indices);
                return;
            }
            ectx.Assert(count > 0);
            ectx.Assert(length > 0);

            Float normScale = scale;
            if (divisor > 0)
                normScale /= divisor;

            // Don't normalize small values.
            if (normScale < MinScale)
                normScale = 1;

            if (offset == 0)
            {
                var dstValues = dst.Values;
                if (Utils.Size(dstValues) < count)
                    dstValues = new Float[count];
                var dstIndices = dst.Indices;
                if (!src.IsDense)
                {
                    if (Utils.Size(dstIndices) < count)
                        dstIndices = new int[count];
                    Array.Copy(src.Indices, dstIndices, count);
                }

                SseUtils.Scale(normScale, src.Values, dstValues, count);
                dst = new VBuffer<Float>(length, count, dstValues, dstIndices);

                return;
            }

            // Subtracting the mean requires a dense representation.
            src.CopyToDense(ref dst);

            if (normScale != 1)
                SseUtils.ScaleAdd(normScale, -offset, dst.Values, length);
            else
                SseUtils.Add(-offset, dst.Values, length);
        }

        /// <summary>
        /// Compute Standard Deviation. In case of both subMean and useStd are true, we technically need to compute variance
        /// based on centered values (i.e. after subtracting the mean). But since the centered
        /// values mean is approximately zero, we can use variance of non-centered values.
        /// </summary>
        private static Float StdDev(Float[] values, int count, int length)
        {
            Contracts.Assert(0 <= count && count <= length);
            if (count == 0)
                return 0;
            // We need a mean to compute variance.
            Float tmpMean = SseUtils.Sum(values, 0, count) / length;
            Float sumSq = 0;
            if (count != length && tmpMean != 0)
            {
                // Sparse representation.
                Float meanSq = tmpMean * tmpMean;
                sumSq = (length - count) * meanSq;
            }
            sumSq += SseUtils.SumSq(tmpMean, values, 0, count);
            return MathUtils.Sqrt(sumSq / length);
        }

        /// <summary>
        /// Compute Standard Deviation.
        /// We have two overloads of StdDev instead of one with <see cref="Nullable{Float}"/> mean for perf reasons.
        /// </summary>
        private static Float StdDev(Float[] values, int count, int length, Float mean)
        {
            Contracts.Assert(0 <= count && count <= length);
            if (count == 0)
                return 0;
            Float sumSq = 0;
            if (count != length && mean != 0)
            {
                // Sparse representation.
                Float meanSq = mean * mean;
                sumSq = (length - count) * meanSq;
            }
            sumSq += SseUtils.SumSq(mean, values, 0, count);
            return MathUtils.Sqrt(sumSq / length);
        }

        /// <summary>
        /// Compute L2-norm. L2-norm computation doesn't subtract the mean from the source values.
        /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
        /// </summary>
        private static Float L2Norm(Float[] values, int count, Float mean = 0)
        {
            if (count == 0)
                return 0;
            return MathUtils.Sqrt(SseUtils.SumSq(mean, values, 0, count));
        }

        /// <summary>
        /// Compute L1-norm. L1-norm computation doesn't subtract the mean from the source values.
        /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
        /// </summary>
        private static Float L1Norm(Float[] values, int count, Float mean = 0)
        {
            if (count == 0)
                return 0;
            return SseUtils.SumAbs(mean, values, 0, count);
        }

        /// <summary>
        /// Compute LInf-norm. LInf-norm computation doesn't subtract the mean from the source values.
        /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
        /// </summary>
        private static Float LInfNorm(Float[] values, int count, Float mean = 0)
        {
            if (count == 0)
                return 0;
            return SseUtils.MaxAbsDiff(mean, values, count);
        }

        private static Float Mean(Float[] src, int count, int length)
        {
            if (length == 0 || count == 0)
                return 0;
            return SseUtils.Sum(src, 0, count) / length;
        }
    }

    public static class LpNormalization
    {
        [TlcModule.EntryPoint(Name = "Transforms.LpNormalizer", Desc = LpNormNormalizerTransform.Summary, UserName = LpNormNormalizerTransform.UserNameLP, ShortName = LpNormNormalizerTransform.ShortNameLP)]
        public static CommonOutputs.TransformOutput Normalize(IHostEnvironment env, LpNormNormalizerTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LpNormalize", input);
            var xf = new LpNormNormalizerTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.GlobalContrastNormalizer", Desc = LpNormNormalizerTransform.GcnSummary, UserName = LpNormNormalizerTransform.UserNameGn, ShortName = LpNormNormalizerTransform.ShortNameGn)]
        public static CommonOutputs.TransformOutput GcNormalize(IHostEnvironment env, LpNormNormalizerTransform.GcnArguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GcNormalize", input);
            var xf = new LpNormNormalizerTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}