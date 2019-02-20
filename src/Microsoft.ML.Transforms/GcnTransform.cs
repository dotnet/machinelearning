// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms.Projections;

[assembly: LoadableClass(LpNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormalizingTransformer), typeof(LpNormalizingTransformer.GcnOptions), typeof(SignatureDataTransform),
    LpNormalizingTransformer.UserNameGn, "GcnTransform", LpNormalizingTransformer.ShortNameGn)]

[assembly: LoadableClass(LpNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    LpNormalizingTransformer.UserNameGn, LpNormalizingTransformer.LoaderSignature, LpNormalizingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(LpNormalizingTransformer.Summary, typeof(IDataTransform), typeof(LpNormalizingTransformer), typeof(LpNormalizingTransformer.Options), typeof(SignatureDataTransform),
    LpNormalizingTransformer.UserNameLP, "LpNormNormalizer", LpNormalizingTransformer.ShortNameLP)]

[assembly: LoadableClass(LpNormalizingTransformer.Summary, typeof(LpNormalizingTransformer), null, typeof(SignatureLoadModel),
    LpNormalizingTransformer.UserNameGn, LpNormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(LpNormalizingTransformer), null, typeof(SignatureLoadRowMapper),
   LpNormalizingTransformer.UserNameGn, LpNormalizingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(LpNormalization))]

namespace Microsoft.ML.Transforms.Projections
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
    ///    <a href="https://www.cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf">https://www.cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf</a>.
    /// </summary>
    public sealed class LpNormalizingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public LpNormalizingEstimatorBase.NormalizerKind NormKind = LpNormalizingEstimatorBase.Defaults.NormKind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 2)]
            public bool SubMean = LpNormalizingEstimatorBase.Defaults.LpSubstractMean;
        }

        internal sealed class GcnOptions : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public GcnColumn[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 1)]
            public bool SubMean = LpNormalizingEstimatorBase.Defaults.GcnSubstractMean;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm", ShortName = "useStd")]
            public bool UseStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public float Scale = LpNormalizingEstimatorBase.Defaults.Scale;
        }

        internal abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing")]
            public bool? SubMean;

            private protected ColumnBase()
            {
            }

            private protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (SubMean != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        internal sealed class Column : ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public LpNormalizingEstimatorBase.NormalizerKind? NormKind;

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
                if (NormKind != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        internal sealed class GcnColumn : ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm")]
            public bool? UseStdDev;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public float? Scale;

            internal static GcnColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new GcnColumn();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (UseStdDev != null || Scale != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        private sealed class ColumnInfoLoaded : LpNormalizingEstimatorBase.ColumnInfoBase
        {
            internal ColumnInfoLoaded(ModelLoadContext ctx, string name, string inputColumnName, bool normKindSerialized)
                : base(ctx, name, inputColumnName, normKindSerialized)
            {

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

        internal const string LoaderSignature = "GcnTransform";
        internal const string LoaderSignatureOld = "GcnFunction";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GCNORMAF",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Added arguments for Lp-norm (vector/row-wise) normalizer
                verWrittenCur: 0x00010003, // Dropped sizeof(Float).
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(LpNormalizingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "LpNormNormalizer";

        // REVIEW: should this be an argument instead?
        private const float MinScale = 1e-8f;

        /// <summary>
        /// The objects describing how the transformation is applied on the input data.
        /// </summary>
        public IReadOnlyCollection<LpNormalizingEstimatorBase.ColumnInfoBase> Columns => _columns.AsReadOnly();
        private readonly LpNormalizingEstimatorBase.ColumnInfoBase[] _columns;

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(LpNormalizingEstimatorBase.ColumnInfoBase[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema[srcCol].Type;
            if (!LpNormalizingEstimatorBase.IsColumnTypeValid(inType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputSchema[srcCol].Name, LpNormalizingEstimatorBase.ExpectedColumnType, inType.ToString());
        }
        /// <summary>
        /// Create a <see cref="LpNormalizingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormalizingTransformer(IHostEnvironment env, params LpNormalizingEstimatorBase.ColumnInfoBase[] columns) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizingTransformer)), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        // Factory method for SignatureDataTransform for GcnArguments class.
        internal static IDataTransform Create(IHostEnvironment env, GcnOptions options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new GlobalContrastNormalizingEstimator.GcnColumnInfo[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = new GlobalContrastNormalizingEstimator.GcnColumnInfo(
                        item.Name,
                        item.Source ?? item.Name,
                        item.SubMean ?? options.SubMean,
                        item.UseStdDev ?? options.UseStdDev,
                        item.Scale ?? options.Scale);
                }
                if (!options.SubMean && options.UseStdDev)
                    ch.Warning("subMean parameter is false while useStd is true. It is advisable to set subMean to true in case useStd is set to true.");
            }
            return new LpNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureDataTransform for Arguments class.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new LpNormalizingEstimator.LpNormColumnInfo[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = new LpNormalizingEstimator.LpNormColumnInfo(
                        item.Name,
                        item.Source ?? item.Name,
                        item.SubMean ?? options.SubMean,
                        item.NormKind ?? options.NormKind);
                }
            }
            return new LpNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static LpNormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(LpNormalizingTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten <= 0x00010002)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new LpNormalizingTransformer(host, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private LpNormalizingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // <columns>
            var columnsLength = ColumnPairs.Length;
            _columns = new ColumnInfoLoaded[columnsLength];
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new ColumnInfoLoaded(ctx, ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx.Header.ModelVerWritten >= VerVectorNormalizerSupported);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // <columns>
            SaveColumns(ctx);

            Host.Assert(_columns.Length == ColumnPairs.Length);
            foreach (var col in _columns)
                col.Save(ctx);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly DataViewType[] _types;
            private readonly LpNormalizingTransformer _parent;

            public Mapper(LpNormalizingTransformer parent, DataViewSchema inputSchema)
                 : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                    _types[i] = srcCol.Type;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new MetadataBuilder();
                    builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.SlotNames);
                    ValueGetter<bool> getter = (ref bool dst) => dst = true;
                    builder.Add(MetadataUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, getter);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.GetMetadata());
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var ex = _parent._columns[iinfo];
                Host.Assert(0 < ex.Scale && ex.Scale < float.PositiveInfinity);
                Host.Assert(_srcTypes[iinfo] is VectorType);

                var getSrc = input.GetGetter<VBuffer<float>>(_srcCols[iinfo]);
                var src = default(VBuffer<float>);
                ValueGetter<VBuffer<float>> del;
                float scale = ex.Scale;

                if (ex.SubtractMean)
                {
                    switch (ex.NormKind)
                    {
                        case LpNormalizingEstimatorBase.NormalizerKind.StdDev:
                            del =
                                (ref VBuffer<float> dst) =>
                                {
                                    getSrc(ref src);
                                    var srcValues = src.GetValues();
                                    var mean = Mean(srcValues, src.Length);
                                    var divisor = StdDev(srcValues, src.Length, mean);
                                    FillValues(Host, in src, ref dst, divisor, scale, mean);
                                };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.L2Norm:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var srcValues = src.GetValues();
                                   var mean = Mean(srcValues, src.Length);
                                   var divisor = L2Norm(srcValues, mean);
                                   FillValues(Host, in src, ref dst, divisor, scale, mean);
                               };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.L1Norm:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var srcValues = src.GetValues();
                                   var mean = Mean(srcValues, src.Length);
                                   var divisor = L1Norm(srcValues, mean);
                                   FillValues(Host, in src, ref dst, divisor, scale, mean);
                               };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.LInf:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var srcValues = src.GetValues();
                                   var mean = Mean(srcValues, src.Length);
                                   var divisor = LInfNorm(srcValues, mean);
                                   FillValues(Host, in src, ref dst, divisor, scale, mean);
                               };
                            return del;
                        default:
                            Host.Assert(false, "Unsupported normalizer type");
                            goto case LpNormalizingEstimatorBase.NormalizerKind.L2Norm;
                    }
                }

                switch (ex.NormKind)
                {
                    case LpNormalizingEstimatorBase.NormalizerKind.StdDev:
                        del =
                            (ref VBuffer<float> dst) =>
                            {
                                getSrc(ref src);
                                var divisor = StdDev(src.GetValues(), src.Length);
                                FillValues(Host, in src, ref dst, divisor, scale);
                            };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.L2Norm:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L2Norm(src.GetValues());
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.L1Norm:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L1Norm(src.GetValues());
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.LInf:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = LInfNorm(src.GetValues());
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    default:
                        Host.Assert(false, "Unsupported normalizer type");
                        goto case LpNormalizingEstimatorBase.NormalizerKind.L2Norm;
                }
            }

            private static void FillValues(IExceptionContext ectx, in VBuffer<float> src, ref VBuffer<float> dst, float divisor, float scale, float offset = 0)
            {
                var srcValues = src.GetValues();
                int count = srcValues.Length;
                int length = src.Length;
                ectx.Assert(divisor >= 0);

                if (count == 0)
                {
                    VBufferUtils.Resize(ref dst, length, 0);
                    return;
                }
                ectx.Assert(count > 0);
                ectx.Assert(length > 0);

                float normScale = scale;
                if (divisor > 0)
                    normScale /= divisor;

                // Don't normalize small values.
                if (normScale < MinScale)
                    normScale = 1;

                VBufferEditor<float> editor;
                if (offset == 0)
                {
                    editor = VBufferEditor.Create(ref dst, length, count);
                    var dstValues = editor.Values;
                    if (!src.IsDense)
                    {
                        src.GetIndices().CopyTo(editor.Indices);
                    }

                    CpuMathUtils.Scale(normScale, src.GetValues(), dstValues, count);
                    dst = editor.Commit();

                    return;
                }

                // Subtracting the mean requires a dense representation.
                src.CopyToDense(ref dst);

                editor = VBufferEditor.CreateFromBuffer(ref dst);
                if (normScale != 1)
                    CpuMathUtils.ScaleAdd(normScale, -offset, editor.Values);
                else
                    CpuMathUtils.Add(-offset, editor.Values);
            }

            /// <summary>
            /// Compute Standard Deviation. In case of both subMean and useStd are true, we technically need to compute variance
            /// based on centered values (i.e. after subtracting the mean). But since the centered
            /// values mean is approximately zero, we can use variance of non-centered values.
            /// </summary>
            private static float StdDev(ReadOnlySpan<float> values, int length)
            {
                Contracts.Assert(0 <= values.Length && values.Length <= length);
                if (values.Length == 0)
                    return 0;
                // We need a mean to compute variance.
                var tmpMean = CpuMathUtils.Sum(values) / length;
                float sumSq = 0;
                if (values.Length != length && tmpMean != 0)
                {
                    // Sparse representation.
                    float meanSq = tmpMean * tmpMean;
                    sumSq = (length - values.Length) * meanSq;
                }
                sumSq += CpuMathUtils.SumSq(tmpMean, values);
                return MathUtils.Sqrt(sumSq / length);
            }

            /// <summary>
            /// Compute Standard Deviation.
            /// We have two overloads of StdDev instead of one with <see cref="Nullable{Float}"/> mean for perf reasons.
            /// </summary>
            private static float StdDev(ReadOnlySpan<float> values, int length, float mean)
            {
                Contracts.Assert(0 <= values.Length && values.Length <= length);
                if (values.Length == 0)
                    return 0;
                float sumSq = 0;
                if (values.Length != length && mean != 0)
                {
                    // Sparse representation.
                    float meanSq = mean * mean;
                    sumSq = (length - values.Length) * meanSq;
                }
                sumSq += CpuMathUtils.SumSq(mean, values);
                return MathUtils.Sqrt(sumSq / length);
            }

            /// <summary>
            /// Compute L2-norm. L2-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float L2Norm(ReadOnlySpan<float> values, float mean = 0)
            {
                if (values.Length == 0)
                    return 0;
                return MathUtils.Sqrt(CpuMathUtils.SumSq(mean, values));
            }

            /// <summary>
            /// Compute L1-norm. L1-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float L1Norm(ReadOnlySpan<float> values, float mean = 0)
            {
                if (values.Length == 0)
                    return 0;
                return CpuMathUtils.SumAbs(mean, values);
            }

            /// <summary>
            /// Compute LInf-norm. LInf-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float LInfNorm(ReadOnlySpan<float> values, float mean = 0)
            {
                if (values.Length == 0)
                    return 0;
                return CpuMathUtils.MaxAbsDiff(mean, values);
            }

            private static float Mean(ReadOnlySpan<float> src, int length)
            {
                if (length == 0 || src.Length == 0)
                    return 0;
                return CpuMathUtils.Sum(src) / length;
            }
        }
    }

    internal static class LpNormalization
    {
        [TlcModule.EntryPoint(Name = "Transforms.LpNormalizer",
            Desc = LpNormalizingTransformer.Summary,
            UserName = LpNormalizingTransformer.UserNameLP,
            ShortName = LpNormalizingTransformer.ShortNameLP)]
        public static CommonOutputs.TransformOutput Normalize(IHostEnvironment env, LpNormalizingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LpNormalize", input);
            var xf = LpNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.GlobalContrastNormalizer",
            Desc = LpNormalizingTransformer.GcnSummary,
            UserName = LpNormalizingTransformer.UserNameGn,
            ShortName = LpNormalizingTransformer.ShortNameGn)]
        public static CommonOutputs.TransformOutput GcNormalize(IHostEnvironment env, LpNormalizingTransformer.GcnOptions input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GcNormalize", input);
            var xf = LpNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }

    /// <summary>
    /// Base estimator class for LpNorm and Gcn normalizers.
    /// </summary>
    public abstract class LpNormalizingEstimatorBase : TrivialEstimator<LpNormalizingTransformer>
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

        /// <summary>
        /// Describes base class for one column pair.
        /// </summary>
        public abstract class ColumnInfoBase
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of column to transform.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// Subtract mean from each value before normalizing.
            /// </summary>
            public readonly bool SubtractMean;
            /// <summary>
            /// The norm to use to normalize each sample.
            /// </summary>
            public readonly NormalizerKind NormKind;
            /// <summary>
            /// Scale features by this value.
            /// </summary>
            public readonly float Scale;

            internal ColumnInfoBase(string name, string inputColumnName, bool substractMean, NormalizerKind normalizerKind, float scale)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Contracts.CheckNonWhiteSpace(inputColumnName, nameof(inputColumnName));
                Name = name;
                InputColumnName = inputColumnName;
                SubtractMean = substractMean;
                Contracts.CheckUserArg(0 < scale && scale < float.PositiveInfinity, nameof(scale), "scale must be a positive finite value");
                Scale = scale;
                NormKind = normalizerKind;
            }

            internal ColumnInfoBase(ModelLoadContext ctx, string name, string inputColumnName, bool normKindSerialized)
            {
                Contracts.AssertValue(ctx);
                Contracts.CheckNonWhiteSpace(inputColumnName, nameof(inputColumnName));
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Name = name;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // byte: SubtractMean
                // byte: NormKind
                // Float: Scale
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
                Contracts.CheckDecode(0 < Scale && Scale < float.PositiveInfinity);
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                // byte: SubtractMean
                // byte: NormKind
                // Float: Scale
                ctx.Writer.WriteBoolByte(SubtractMean);
                ctx.Writer.Write((byte)NormKind);
                Contracts.Assert(0 < Scale && Scale < float.PositiveInfinity);
                ctx.Writer.Write(Scale);
            }
        }

        [BestFriend]
        internal static class Defaults
        {
            public const NormalizerKind NormKind = NormalizerKind.L2Norm;
            public const bool LpSubstractMean = false;
            public const bool GcnSubstractMean = true;
            public const bool UseStdDev = false;
            public const float Scale = 1;
        }

        /// <summary>
        /// Create a <see cref="LpNormalizingEstimatorBase"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormalizingEstimatorBase(IHostEnvironment env, params ColumnInfoBase[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizingEstimator)), new LpNormalizingTransformer(env, columns))
        {
        }

        internal static bool IsColumnTypeValid(DataViewType type)
        {
            if (!(type is VectorType vectorType && vectorType.IsKnownSize))
                return false;
            return vectorType.ItemType == NumberDataViewType.Single;
        }

        internal static bool IsSchemaColumnValid(SchemaShape.Column col)
        {
            if (col.Kind != SchemaShape.Column.VectorKind.Vector)
                return false;
            return col.ItemType == NumberDataViewType.Single;
        }

        internal const string ExpectedColumnType = "Expected float or float vector of known size";

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colPair.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.InputColumnName);
                if (!IsSchemaColumnValid(col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.InputColumnName, ExpectedColumnType, col.GetTypeString());
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                result[colPair.Name] = new SchemaShape.Column(colPair.Name, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Lp Normalizing estimator takes columns and normalizes them individually by rescaling them to unit norm.
    /// </summary>
    public sealed class LpNormalizingEstimator : LpNormalizingEstimatorBase
    {
        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class LpNormColumnInfo : ColumnInfoBase
        {
            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
            /// <param name="normalizerKind">The norm to use to normalize each sample.</param>
            public LpNormColumnInfo(string name, string inputColumnName = null,
                bool substractMean = Defaults.LpSubstractMean,
                NormalizerKind normalizerKind = Defaults.NormKind)
                : base(name, inputColumnName ?? name, substractMean, normalizerKind, 1)
            {
            }
        }
        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
        internal LpNormalizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            NormalizerKind normKind = Defaults.NormKind, bool substractMean = Defaults.LpSubstractMean)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, normKind, substractMean)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
        internal LpNormalizingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns,
            NormalizerKind normKind = Defaults.NormKind, bool substractMean = Defaults.LpSubstractMean)
             : this(env, columns.Select(x => new LpNormColumnInfo(x.outputColumnName, x.inputColumnName, substractMean, normKind)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="LpNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormalizingEstimator(IHostEnvironment env, params LpNormColumnInfo[] columns)
            : base(env, columns)
        {
        }
    }

    /// <summary>
    /// Global contrast normalizing estimator takes columns and performs global constrast normalization.
    /// </summary>
    public sealed class GlobalContrastNormalizingEstimator : LpNormalizingEstimatorBase
    {
        /// <summary>
        /// Describes how the transformer handles one Gcn column pair.
        /// </summary>
        public sealed class GcnColumnInfo : ColumnInfoBase
        {
            /// <summary>
            /// Describes how the transformer handles one Gcn column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
            /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
            /// <param name="scale">Scale features by this value.</param>
            public GcnColumnInfo(string name, string inputColumnName = null,
                bool substractMean = Defaults.GcnSubstractMean,
                bool useStdDev = Defaults.UseStdDev,
                float scale = Defaults.Scale)
                : base(name, inputColumnName, substractMean, useStdDev ? NormalizerKind.StdDev : NormalizerKind.L2Norm, scale)
            {
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            bool substractMean = Defaults.GcnSubstractMean, bool useStdDev = Defaults.UseStdDev, float scale = Defaults.Scale)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, substractMean, useStdDev, scale)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns,
            bool substractMean = Defaults.GcnSubstractMean, bool useStdDev = Defaults.UseStdDev, float scale = Defaults.Scale)
            : this(env, columns.Select(x => new GcnColumnInfo(x.outputColumnName, x.inputColumnName, substractMean, useStdDev, scale)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="GlobalContrastNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, params GcnColumnInfo[] columns) :
            base(env, columns)
        {
        }
    }
}
