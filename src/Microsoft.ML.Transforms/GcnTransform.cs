// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(LpNormNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormNormalizingTransformer), typeof(LpNormNormalizingTransformer.GcnOptions), typeof(SignatureDataTransform),
    LpNormNormalizingTransformer.UserNameGn, "GcnTransform", LpNormNormalizingTransformer.ShortNameGn)]

[assembly: LoadableClass(LpNormNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormNormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    LpNormNormalizingTransformer.UserNameGn, LpNormNormalizingTransformer.LoaderSignature, LpNormNormalizingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(LpNormNormalizingTransformer.Summary, typeof(IDataTransform), typeof(LpNormNormalizingTransformer), typeof(LpNormNormalizingTransformer.Options), typeof(SignatureDataTransform),
    LpNormNormalizingTransformer.UserNameLP, "LpNormNormalizer", LpNormNormalizingTransformer.ShortNameLP)]

[assembly: LoadableClass(LpNormNormalizingTransformer.Summary, typeof(LpNormNormalizingTransformer), null, typeof(SignatureLoadModel),
    LpNormNormalizingTransformer.UserNameGn, LpNormNormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(LpNormNormalizingTransformer), null, typeof(SignatureLoadRowMapper),
   LpNormNormalizingTransformer.UserNameGn, LpNormNormalizingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(LpNormNormalization))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="LpNormNormalizingEstimator"/> or <see cref="GlobalContrastNormalizingEstimator"/>.
    /// </summary>
    public sealed class LpNormNormalizingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public LpNormNormalizingEstimatorBase.NormFunction Norm = LpNormNormalizingEstimatorBase.Defaults.Norm;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 2)]
            public bool SubMean = LpNormNormalizingEstimatorBase.Defaults.LpEnsureZeroMean;
        }

        internal sealed class GcnOptions : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public GcnColumn[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 1)]
            public bool SubMean = LpNormNormalizingEstimatorBase.Defaults.GcnEnsureZeroMean;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm", ShortName = "useStd")]
            public bool UseStdDev = LpNormNormalizingEstimatorBase.Defaults.EnsureUnitStdDev;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public float Scale = LpNormNormalizingEstimatorBase.Defaults.Scale;
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
            public LpNormNormalizingEstimatorBase.NormFunction? Norm;

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
                if (Norm != null)
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

        private sealed class ColumnOptionsLoaded : LpNormNormalizingEstimatorBase.ColumnOptionsBase
        {
            internal ColumnOptionsLoaded(ModelLoadContext ctx, string name, string inputColumnName, bool normKindSerialized)
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
                loaderAssemblyName: typeof(LpNormNormalizingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "LpNormNormalizer";

        // REVIEW: should this be an argument instead?
        private const float MinScale = 1e-8f;

        /// <summary>
        /// The objects describing how the transformation is applied on the input data.
        /// </summary>
        internal IReadOnlyCollection<LpNormNormalizingEstimatorBase.ColumnOptionsBase> Columns => _columns.AsReadOnly();
        private readonly LpNormNormalizingEstimatorBase.ColumnOptionsBase[] _columns;

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(LpNormNormalizingEstimatorBase.ColumnOptionsBase[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema[srcCol].Type;
            if (!LpNormNormalizingEstimatorBase.IsColumnTypeValid(inType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputSchema[srcCol].Name, LpNormNormalizingEstimatorBase.ExpectedColumnType, inType.ToString());
        }
        /// <summary>
        /// Create a <see cref="LpNormNormalizingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormNormalizingTransformer(IHostEnvironment env, params LpNormNormalizingEstimatorBase.ColumnOptionsBase[] columns) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormNormalizingTransformer)), GetColumnPairs(columns))
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
            var cols = new GlobalContrastNormalizingEstimator.ColumnOptions[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = new GlobalContrastNormalizingEstimator.ColumnOptions(
                        item.Name,
                        item.Source ?? item.Name,
                        item.SubMean ?? options.SubMean,
                        item.UseStdDev ?? options.UseStdDev,
                        item.Scale ?? options.Scale);
                }
                if (!options.SubMean && options.UseStdDev)
                    ch.Warning("subMean parameter is false while useStd is true. It is advisable to set subMean to true in case useStd is set to true.");
            }
            return new LpNormNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureDataTransform for Arguments class.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new LpNormNormalizingEstimator.ColumnOptions[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = new LpNormNormalizingEstimator.ColumnOptions(
                        item.Name,
                        item.Source ?? item.Name,
                        item.Norm ?? options.Norm,
                        item.SubMean ?? options.SubMean);
                }
            }
            return new LpNormNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static LpNormNormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(LpNormNormalizingTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten <= 0x00010002)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new LpNormNormalizingTransformer(host, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private LpNormNormalizingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // <columns>
            var columnsLength = ColumnPairs.Length;
            _columns = new ColumnOptionsLoaded[columnsLength];
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new ColumnOptionsLoaded(ctx, ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx.Header.ModelVerWritten >= VerVectorNormalizerSupported);
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

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly DataViewType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly DataViewType[] _types;
            private readonly LpNormNormalizingEstimatorBase.NormFunction[] _norms;
            private readonly bool[] _ensureZeroMeans;
            private readonly LpNormNormalizingTransformer _parent;

            public Mapper(LpNormNormalizingTransformer parent, DataViewSchema inputSchema)
                 : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                _norms = new LpNormNormalizingEstimatorBase.NormFunction[_parent.ColumnPairs.Length];
                _ensureZeroMeans = new bool[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                    _types[i] = srcCol.Type;
                    _norms[i] = _parent._columns[i].Norm;
                    _ensureZeroMeans[i] = _parent._columns[i].EnsureZeroMean;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    builder.Add(InputSchema[ColMapNewToOld[i]].Annotations, name => name == AnnotationUtils.Kinds.SlotNames);
                    ValueGetter<bool> getter = (ref bool dst) => dst = true;
                    builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, getter);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.ToAnnotations());
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
                Host.Assert(_srcTypes[iinfo] is VectorDataViewType);

                var getSrc = input.GetGetter<VBuffer<float>>(input.Schema[_srcCols[iinfo]]);
                var src = default(VBuffer<float>);
                ValueGetter<VBuffer<float>> del;
                float scale = ex.Scale;

                if (ex.EnsureZeroMean)
                {
                    switch (ex.Norm)
                    {
                        case LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation:
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
                        case LpNormNormalizingEstimatorBase.NormFunction.L2:
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
                        case LpNormNormalizingEstimatorBase.NormFunction.L1:
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
                        case LpNormNormalizingEstimatorBase.NormFunction.Infinity:
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
                            goto case LpNormNormalizingEstimatorBase.NormFunction.L2;
                    }
                }

                switch (ex.Norm)
                {
                    case LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation:
                        del =
                            (ref VBuffer<float> dst) =>
                            {
                                getSrc(ref src);
                                var divisor = StdDev(src.GetValues(), src.Length);
                                FillValues(Host, in src, ref dst, divisor, scale);
                            };
                        return del;
                    case LpNormNormalizingEstimatorBase.NormFunction.L2:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L2Norm(src.GetValues());
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormNormalizingEstimatorBase.NormFunction.L1:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L1Norm(src.GetValues());
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormNormalizingEstimatorBase.NormFunction.Infinity:
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
                        goto case LpNormNormalizingEstimatorBase.NormFunction.L2;
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

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _srcCols.Length; ++iinfo)
                {
                    string inputColumnName = InputSchema[_srcCols[iinfo]].Name;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(inputColumnName, false);
                        continue;
                    }

                    string outputColumnName = _parent.ColumnPairs[iinfo].outputColumnName;
                    if (!SaveAsOnnxCore(ctx, iinfo, ctx.GetVariableName(inputColumnName), ctx.AddIntermediateVariable(_srcTypes[iinfo], outputColumnName)))
                    {
                        ctx.RemoveColumn(inputColumnName, true);
                    }
                }
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, string srcVariableName, string dstVariableName)
            {
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoaderSignature);

                string opType;

                if ((_norms[iinfo] != LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation) && (_ensureZeroMeans[iinfo] == false))
                {
                    string strNorm;
                    if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.L1)
                        strNorm = "L1";
                    else if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.L2)
                        strNorm = "L2";
                    else
                        strNorm = "MAX";
                    opType = "Normalizer";
                    var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                    node.AddAttribute("norm", strNorm);
                    return true;
                }

                opType = "ReduceMean";
                string meanOfInput = ctx.AddIntermediateVariable(_types[iinfo], "MeanOfInput", true);
                var meanNode = ctx.CreateNode(opType, srcVariableName, meanOfInput, ctx.GetNodeName(opType), "");
                meanNode.AddAttribute("axes", new long[] { 1 });

                opType = "Sub";
                string inputMinusMean = ctx.AddIntermediateVariable(_types[iinfo], "InputMinusMean");
                var subtractNode = ctx.CreateNode(opType, new[] { srcVariableName, meanOfInput }, new[] { inputMinusMean }, ctx.GetNodeName(opType), "");

                if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.L1)
                {
                    opType = "Abs";
                    string absOfInput = ctx.AddIntermediateVariable(_types[iinfo], "AbsOfInput");
                    var absNode = ctx.CreateNode(opType, inputMinusMean, absOfInput, ctx.GetNodeName(opType), "");

                    opType = "ReduceSum";
                    string sumOfAbsOfInput = ctx.AddIntermediateVariable(_types[iinfo], "SumOfAbsOfInput", true);
                    var sumOfAbsNode = ctx.CreateNode(opType, absOfInput, sumOfAbsOfInput, ctx.GetNodeName(opType), "");
                    sumOfAbsNode.AddAttribute("axes", new long[] { 1 });

                    opType = "Div";
                    var l1Node = ctx.CreateNode(opType, new[] { inputMinusMean, sumOfAbsOfInput }, new[] { dstVariableName }, ctx.GetNodeName(opType), "");
                }
                else if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.L2)
                {
                    opType = "Pow";
                    string two = ctx.AddInitializer(2.0f);
                    string squareOfInput = ctx.AddIntermediateVariable(_types[iinfo], "SquareOfInput", true);
                    var squareNode = ctx.CreateNode(opType, new[] { inputMinusMean, two }, new[] { squareOfInput }, ctx.GetNodeName(opType), "");

                    opType = "ReduceSum";
                    string sumOfSquares = ctx.AddIntermediateVariable(_types[iinfo], "SumOfSquares", true);
                    var sumOfSquaresNode = ctx.CreateNode(opType, squareOfInput, sumOfSquares, ctx.GetNodeName(opType), "");
                    sumOfSquaresNode.AddAttribute("axes", new long[] { 1 });

                    opType = "Sqrt";
                    string squareRoot = ctx.AddIntermediateVariable(_types[iinfo], "SquareRoot", true);
                    var squareRootNode = ctx.CreateNode(opType, sumOfSquares, squareRoot, ctx.GetNodeName(opType), "");

                    opType = "Div";
                    var l2Node = ctx.CreateNode(opType, new[] { inputMinusMean, squareRoot }, new[] { dstVariableName }, ctx.GetNodeName(opType), "");
                }
                else if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.Infinity)
                {
                    opType = "ReduceMax";
                    string maxOfInput = ctx.AddIntermediateVariable(_types[iinfo], "MaxOfInput", true);
                    var maxNode = ctx.CreateNode(opType, inputMinusMean, maxOfInput, ctx.GetNodeName(opType), "");
                    maxNode.AddAttribute("axes", new long[] { 1 });

                    opType = "Div";
                    var lMaxNode = ctx.CreateNode(opType, new[] { inputMinusMean, maxOfInput }, new[] { dstVariableName }, ctx.GetNodeName(opType), "");
                }
                else if (_norms[iinfo] == LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation)
                {
                    // first calculate the standard deviation
                    opType = "Pow";
                    string two = ctx.AddInitializer(2.0f);
                    string squareOfInputMinusMean = ctx.AddIntermediateVariable(_types[iinfo], "SquareOfInputMinusMean", true);
                    var squareOfInputMinusMeanNode = ctx.CreateNode(opType, new[] { inputMinusMean, two }, new[] { squareOfInputMinusMean }, ctx.GetNodeName(opType), "");

                    opType = "ReduceMean";
                    string average = ctx.AddIntermediateVariable(_types[iinfo], "SumOfSquares", true);
                    var sumOfSquaresNode = ctx.CreateNode(opType, squareOfInputMinusMean, average, ctx.GetNodeName(opType), "");
                    sumOfSquaresNode.AddAttribute("axes", new long[] { 1 });

                    opType = "Sqrt";
                    string stdDev = ctx.AddIntermediateVariable(_types[iinfo], "SquareRoot", true);
                    var stdDevNode = ctx.CreateNode(opType, average, stdDev, ctx.GetNodeName(opType), "");

                    opType = "Div";
                    string input = _ensureZeroMeans[iinfo] ? inputMinusMean : srcVariableName;
                    var lStdDevNode = ctx.CreateNode(opType, new[] { input, stdDev }, new[] { dstVariableName }, ctx.GetNodeName(opType), "");
                }
                else
                {
                    Contracts.Assert(false);
                    return false;
                }
                return true;
            }
        }
    }

    internal static class LpNormNormalization
    {
        [TlcModule.EntryPoint(Name = "Transforms.LpNormalizer",
            Desc = LpNormNormalizingTransformer.Summary,
            UserName = LpNormNormalizingTransformer.UserNameLP,
            ShortName = LpNormNormalizingTransformer.ShortNameLP)]
        public static CommonOutputs.TransformOutput Normalize(IHostEnvironment env, LpNormNormalizingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LpNormalize", input);
            var xf = LpNormNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.GlobalContrastNormalizer",
            Desc = LpNormNormalizingTransformer.GcnSummary,
            UserName = LpNormNormalizingTransformer.UserNameGn,
            ShortName = LpNormNormalizingTransformer.ShortNameGn)]
        public static CommonOutputs.TransformOutput GcNormalize(IHostEnvironment env, LpNormNormalizingTransformer.GcnOptions input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GcNormalize", input);
            var xf = LpNormNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }

    /// <summary>
    /// Base estimator class for <see cref="LpNormNormalizingEstimator"/> and <see cref="GlobalContrastNormalizingEstimator"/> normalizers.
    /// </summary>
    public abstract class LpNormNormalizingEstimatorBase : TrivialEstimator<LpNormNormalizingTransformer>
    {
        /// <summary>
        /// The kind of unit norm vectors are rescaled to. This enumeration is serialized.
        /// </summary>
        public enum NormFunction : byte
        {
            /// <summary>
            /// L2-norm.
            /// </summary>
            L2 = 0,
            /// <summary>
            /// Standard deviation of a vector by viewing all its coordinates as a random variable.
            /// </summary>
            StandardDeviation = 1,
            /// <summary>
            /// L1-norm.
            /// </summary>
            L1 = 2,
            /// <summary>
            /// Infinity-norm.
            /// </summary>
            Infinity = 3
        }

        /// <summary>
        /// Describes base class for one column pair.
        /// </summary>
        internal abstract class ColumnOptionsBase
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
            /// The norm to use to normalize each sample.
            /// </summary>
            public readonly NormFunction Norm;
            /// <summary>
            /// Subtract mean from each value before normalizing.
            /// </summary>
            public readonly bool EnsureZeroMean;
            /// <summary>
            /// Scale features by this value.
            /// </summary>
            public readonly float Scale;

            internal ColumnOptionsBase(string name, string inputColumnName, NormFunction normalizerKind, bool substractMean, float scale)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Contracts.CheckNonWhiteSpace(inputColumnName, nameof(inputColumnName));
                Name = name;
                InputColumnName = inputColumnName;
                EnsureZeroMean = substractMean;
                Contracts.CheckUserArg(0 < scale && scale < float.PositiveInfinity, nameof(scale), "scale must be a positive finite value");
                Scale = scale;
                Norm = normalizerKind;
            }

            internal ColumnOptionsBase(ModelLoadContext ctx, string name, string inputColumnName, bool normKindSerialized)
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
                EnsureZeroMean = ctx.Reader.ReadBoolByte();
                byte normKindVal = ctx.Reader.ReadByte();
                Contracts.CheckDecode(Enum.IsDefined(typeof(NormFunction), normKindVal));
                Norm = (NormFunction)normKindVal;
                // Note: In early versions, a bool option (useStd) to whether to normalize by StdDev rather than
                // L2 norm was used. normKind was added in version=verVectorNormalizerSupported.
                // normKind was defined in a way such that the serialized boolean (0: use StdDev, 1: use L2) is
                // still valid.
                Contracts.CheckDecode(normKindSerialized ||
                        (Norm == NormFunction.L2 || Norm == NormFunction.StandardDeviation));
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
                ctx.Writer.WriteBoolByte(EnsureZeroMean);
                ctx.Writer.Write((byte)Norm);
                Contracts.Assert(0 < Scale && Scale < float.PositiveInfinity);
                ctx.Writer.Write(Scale);
            }
        }

        [BestFriend]
        internal static class Defaults
        {
            public const NormFunction Norm = NormFunction.L2;
            public const bool LpEnsureZeroMean = false;
            public const bool GcnEnsureZeroMean = true;
            public const bool EnsureUnitStdDev = false;
            public const float Scale = 1;
        }

        /// <summary>
        /// Create a <see cref="LpNormNormalizingEstimatorBase"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormNormalizingEstimatorBase(IHostEnvironment env, params ColumnOptionsBase[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormNormalizingEstimator)), new LpNormNormalizingTransformer(env, columns))
        {
        }

        internal static bool IsColumnTypeValid(DataViewType type)
        {
            if (!(type is VectorDataViewType vectorType && vectorType.IsKnownSize))
                return false;
            return vectorType.ItemType == NumberDataViewType.Single;
        }

        internal static bool IsSchemaColumnValid(SchemaShape.Column col)
        {
            if (col.Kind != SchemaShape.Column.VectorKind.Vector)
                return false;
            return col.ItemType == NumberDataViewType.Single;
        }

        internal const string ExpectedColumnType = "Expected known-size vector of Single";

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
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                result[colPair.Name] = new SchemaShape.Column(colPair.Name, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Normalizes (scales) vectors in the input column to the unit norm. The type of norm that is used can be specified by the user.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Vector of <xref:System.Single> |
    /// | Output column data type | Vector of <xref:System.Single> |
    /// | Exportable to ONNX | Yes |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.LpNormNormalizingTransformer> normalizes vectors in the input column individually
    /// by rescaling them to the unit norm.
    ///
    /// Let $x$ be the input vector, $n$ the size of the vector, $L(x)$ the norm function selected by the user.
    /// Let $\mu(x) = \sum_i x_i / n$ be the mean of the values of vector $x$. The <xref:Microsoft.ML.Transforms.LpNormNormalizingTransformer>
    /// performs the following operation on each input vector $x$:
    /// $y = \frac{x - \mu(x)}{L(x)}$
    /// if the user specifies that the mean should be zero, or otherwise:
    /// $y = \frac{x}{L(x)}$
    ///
    /// There are four types of norm that can be selected by the user to be applied on input vector $x$. They are defined as follows:
    /// - <xref:Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction.L1>: $L_1(x) = \sum_i |x_i|$
    /// - <xref:Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction.L2>: $L_2(x) = \sqrt{\sum_i x_i^2}$
    /// - <xref:Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction.Infinity>: $L_{\infty}(x) = \max_i\{|x_i|\}$
    /// - <xref:Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation>: $L_\sigma(x)$ is defined as the
    /// standard deviation of the elements of the input vector $x$
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="NormalizationCatalog.NormalizeLpNorm(TransformsCatalog, string, string, LpNormNormalizingEstimatorBase.NormFunction, bool)"/>
    public sealed class LpNormNormalizingEstimator : LpNormNormalizingEstimatorBase
    {
        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        internal sealed class ColumnOptions : ColumnOptionsBase
        {
            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="norm">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
            /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
            public ColumnOptions(string name, string inputColumnName = null,
                NormFunction norm = Defaults.Norm,
                bool ensureZeroMean = Defaults.LpEnsureZeroMean)
                : base(name, inputColumnName ?? name, norm, ensureZeroMean, 1)
            {
            }
        }
        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="norm">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        internal LpNormNormalizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            NormFunction norm = Defaults.Norm, bool ensureZeroMean = Defaults.LpEnsureZeroMean)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, norm, ensureZeroMean)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="norm">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        internal LpNormNormalizingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns,
            NormFunction norm = Defaults.Norm, bool ensureZeroMean = Defaults.LpEnsureZeroMean)
             : this(env, columns.Select(x => new ColumnOptions(x.outputColumnName, x.inputColumnName, norm, ensureZeroMean)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="LpNormNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        internal LpNormNormalizingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
            : base(env, columns)
        {
        }
    }

    /// <summary>
    /// Normalizes (scales) vectors in the input column applying the global contrast normalization.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Vector of <xref:System.Single> |
    /// | Output column data type | Vector of <xref:System.Single> |
    /// | Exportable to ONNX | Yes |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.LpNormNormalizingTransformer> normalizes vectors in the input column individually,
    /// rescaling them by applying global contrast normalization. The transform performs the following operation on each input vector $x$:
    /// $y = \frac{s * x - \mu(x)}{L(x)}$.
    /// Where $s$ is a user provided scaling factor, $\mu(x)$ is the mean of the elements of vector $x$, and $L(x)$ is the $L_2$ norm or the
    /// standard deviation of the elements of vector $x$. These settings can be specified by the user when the
    /// <xref:Microsoft.ML.Transforms.GlobalContrastNormalizingEstimator> is initialized.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="NormalizationCatalog.NormalizeGlobalContrast(TransformsCatalog, string, string, bool, bool, float)"/>
    public sealed class GlobalContrastNormalizingEstimator : LpNormNormalizingEstimatorBase
    {
        /// <summary>
        /// Describes how the transformer handles one Gcn column pair.
        /// </summary>
        internal sealed class ColumnOptions : ColumnOptionsBase
        {
            /// <summary>
            /// Describes how the transformer handles one Gcn column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
            /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
            /// <param name="scale">Scale features by this value.</param>
            public ColumnOptions(string name, string inputColumnName = null,
                bool ensureZeroMean = Defaults.GcnEnsureZeroMean,
                bool ensureUnitStandardDeviation = Defaults.EnsureUnitStdDev,
                float scale = Defaults.Scale)
                : base(name, inputColumnName, ensureUnitStandardDeviation ? NormFunction.StandardDeviation : NormFunction.L2, ensureZeroMean, scale)
            {
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
        /// <param name="scale">Scale features by this value.</param>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            bool ensureZeroMean = Defaults.GcnEnsureZeroMean, bool ensureUnitStandardDeviation = Defaults.EnsureUnitStdDev, float scale = Defaults.Scale)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, ensureZeroMean, ensureUnitStandardDeviation, scale)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
        /// <param name="scale">Scale features by this value.</param>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns,
            bool ensureZeroMean = Defaults.GcnEnsureZeroMean, bool ensureUnitStandardDeviation = Defaults.EnsureUnitStdDev, float scale = Defaults.Scale)
            : this(env, columns.Select(x => new ColumnOptions(x.outputColumnName, x.inputColumnName, ensureZeroMean, ensureUnitStandardDeviation, scale)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="GlobalContrastNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        internal GlobalContrastNormalizingEstimator(IHostEnvironment env, params ColumnOptions[] columns) :
            base(env, columns)
        {
        }
    }
}
