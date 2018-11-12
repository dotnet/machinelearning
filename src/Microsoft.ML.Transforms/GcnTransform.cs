// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Projections;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(LpNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormalizingTransformer), typeof(LpNormalizingTransformer.GcnArguments), typeof(SignatureDataTransform),
    LpNormalizingTransformer.UserNameGn, "GcnTransform", LpNormalizingTransformer.ShortNameGn)]

[assembly: LoadableClass(LpNormalizingTransformer.GcnSummary, typeof(IDataTransform), typeof(LpNormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    LpNormalizingTransformer.UserNameGn, LpNormalizingTransformer.LoaderSignature, LpNormalizingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(LpNormalizingTransformer.Summary, typeof(IDataTransform), typeof(LpNormalizingTransformer), typeof(LpNormalizingTransformer.Arguments), typeof(SignatureDataTransform),
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
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The norm to use to normalize each sample", ShortName = "norm", SortOrder = 1)]
            public LpNormalizingEstimatorBase.NormalizerKind NormKind = LpNormalizingEstimatorBase.Defaults.NormKind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 2)]
            public bool SubMean = LpNormalizingEstimatorBase.Defaults.LpSubMean;
        }

        public sealed class GcnArguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public GcnColumn[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Subtract mean from each value before normalizing", SortOrder = 1)]
            public bool SubMean = LpNormalizingEstimatorBase.Defaults.GcnSubMean;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize by standard deviation rather than L2 norm", ShortName = "useStd")]
            public bool UseStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale features by this value")]
            public float Scale = LpNormalizingEstimatorBase.Defaults.Scale;
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
            public LpNormalizingEstimatorBase.NormalizerKind? NormKind;

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
            public float? Scale;

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

        /// <summary>
        /// Describes how the transformer handles one Gcn column pair.
        /// </summary>
        public sealed class GcnColumnInfo : ColumnInfoBase
        {
            /// <summary>
            /// Describes how the transformer handles one Gcn column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="subMean">Subtract mean from each value before normalizing.</param>
            /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
            /// <param name="scale">Scale features by this value.</param>
            public GcnColumnInfo(string input, string output,
                bool subMean = LpNormalizingEstimatorBase.Defaults.GcnSubMean,
                bool useStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev,
                float scale = LpNormalizingEstimatorBase.Defaults.Scale)
                : base(input, output, subMean, useStdDev ? LpNormalizingEstimatorBase.NormalizerKind.StdDev : LpNormalizingEstimatorBase.NormalizerKind.L2Norm, scale)
            {
            }
        }

        /// <summary>
        /// Describes how the transformer handles one LpNorm column pair.
        /// </summary>
        public sealed class LpNormColumnInfo : ColumnInfoBase
        {
            /// <summary>
            /// Describes how the transformer handles one LpNorm column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="subMean">Subtract mean from each value before normalizing.</param>
            /// <param name="normalizerKind">The norm to use to normalize each sample.</param>
            public LpNormColumnInfo(string input, string output,
                bool subMean = LpNormalizingEstimatorBase.Defaults.LpSubMean,
                LpNormalizingEstimatorBase.NormalizerKind normalizerKind = LpNormalizingEstimatorBase.Defaults.NormKind)
                : base(input, output, subMean, normalizerKind, 1)
            {
            }
        }

        private sealed class ColumnInfoLoaded : ColumnInfoBase
        {
            internal ColumnInfoLoaded(ModelLoadContext ctx, string input, string output, bool normKindSerialized)
                : base(ctx, input, output, normKindSerialized)
            {

            }
        }

        /// <summary>
        /// Describes base class for one column pair.
        /// </summary>
        public abstract class ColumnInfoBase
        {
            public readonly string Input;
            public readonly string Output;
            public readonly bool SubtractMean;
            public readonly LpNormalizingEstimatorBase.NormalizerKind NormKind;
            public readonly float Scale;

            internal ColumnInfoBase(string input, string output, bool subMean, LpNormalizingEstimatorBase.NormalizerKind normalizerKind, float scale)
            {
                Contracts.CheckNonWhiteSpace(input, nameof(input));
                Contracts.CheckNonWhiteSpace(output, nameof(output));
                Input = input;
                Output = output;
                SubtractMean = subMean;
                Contracts.CheckUserArg(0 < scale && scale < float.PositiveInfinity, nameof(scale), "scale must be a positive finite value");
                Scale = scale;
                NormKind = normalizerKind;
            }

            internal ColumnInfoBase(ModelLoadContext ctx, string input, string output, bool normKindSerialized)
            {
                Contracts.AssertValue(ctx);
                Contracts.CheckNonWhiteSpace(input, nameof(input));
                Contracts.CheckNonWhiteSpace(output, nameof(output));
                Input = input;
                Output = output;

                // *** Binary format ***
                // byte: subMean
                // byte: NormKind
                // Float: scale
                SubtractMean = ctx.Reader.ReadBoolByte();
                byte normKindVal = ctx.Reader.ReadByte();
                Contracts.CheckDecode(Enum.IsDefined(typeof(LpNormalizingEstimatorBase.NormalizerKind), normKindVal));
                NormKind = (LpNormalizingEstimatorBase.NormalizerKind)normKindVal;
                // Note: In early versions, a bool option (useStd) to whether to normalize by StdDev rather than
                // L2 norm was used. normKind was added in version=verVectorNormalizerSupported.
                // normKind was defined in a way such that the serialized boolean (0: use StdDev, 1: use L2) is
                // still valid.
                Contracts.CheckDecode(normKindSerialized ||
                        (NormKind == LpNormalizingEstimatorBase.NormalizerKind.L2Norm || NormKind == LpNormalizingEstimatorBase.NormalizerKind.StdDev));
                Scale = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(0 < Scale && Scale < float.PositiveInfinity);
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                // byte: subMean
                // byte: NormKind
                // Float: scale
                ctx.Writer.WriteBoolByte(SubtractMean);
                ctx.Writer.Write((byte)NormKind);
                Contracts.Assert(0 < Scale && Scale < float.PositiveInfinity);
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
        private const float MinScale = (float)1e-8;

        public IReadOnlyCollection<ColumnInfoBase> Columns => _columns.AsReadOnly();
        private readonly ColumnInfoBase[] _columns;

        private static (string input, string output)[] GetColumnPairs(ColumnInfoBase[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema.GetColumnType(srcCol);
            var reason = TestColumn(inType);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        // Check if the input column's type is supported. Note that only float vector with a known shape is allowed.
        internal static string TestColumn(ColumnType type)
        {
            if ((type.IsVector && !type.IsKnownSizeVector && (type.AsVector.Dimensions.Length > 1)) || type.ItemType != NumberType.R4)
                return "Expected float or float vector of known size";

            if ((long)type.ValueCount * type.ValueCount > Utils.ArrayMaxSize)
                return "Vector size exceeds limit";

            return null;
        }

        /// <summary>
        /// Create a <see cref="LpNormalizingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        public LpNormalizingTransformer(IHostEnvironment env, params ColumnInfoBase[] columns) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizingTransformer)), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        // Factory method for SignatureDataTransform for GcnArguments class/>
        internal static IDataTransform Create(IHostEnvironment env, GcnArguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new GcnColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    cols[i] = new GcnColumnInfo(item.Source ?? item.Name,
                        item.Name,
                        item.SubMean ?? args.SubMean,
                        item.UseStdDev ?? args.UseStdDev,
                        item.Scale ?? args.Scale);
                }
                if (!args.SubMean && args.UseStdDev)
                    ch.Warning("subMean parameter is false while useStd is true. It is advisable to set subMean to true in case useStd is set to true.");
            }
            return new LpNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureDataTransform for Arguments class/>
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new LpNormColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    cols[i] = new LpNormColumnInfo(item.Source ?? item.Name,
                        item.Name,
                        item.SubMean ?? args.SubMean,
                        item.NormKind ?? args.NormKind);
                }
            }
            return new LpNormalizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static LpNormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(RffTransform));

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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

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
                _columns[i] = new ColumnInfoLoaded(ctx, ColumnPairs[i].input, ColumnPairs[i].output, ctx.Header.ModelVerWritten >= VerVectorNormalizerSupported);
        }

        public override void Save(ModelSaveContext ctx)
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

        protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly ColumnType[] _types;
            private readonly LpNormalizingTransformer _parent;

            public Mapper(LpNormalizingTransformer parent, Schema inputSchema)
                 : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                    _types[i] = srcCol.Type;
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new Schema.Metadata.Builder();
                    builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.SlotNames);
                    ValueGetter<bool> getter = (ref bool dst) => dst = true;
                    builder.Add(new Schema.Column(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, null), getter);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var ex = _parent._columns[iinfo];
                Host.Assert(0 < ex.Scale && ex.Scale < float.PositiveInfinity);
                Host.Assert(_srcTypes[iinfo].IsVector);

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
                                    var mean = Mean(src.Values, src.Count, src.Length);
                                    var divisor = StdDev(src.Values, src.Count, src.Length, mean);
                                    FillValues(Host, in src, ref dst, divisor, scale, mean);
                                };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.L2Norm:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var mean = Mean(src.Values, src.Count, src.Length);
                                   var divisor = L2Norm(src.Values, src.Count, mean);
                                   FillValues(Host, in src, ref dst, divisor, scale, mean);
                               };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.L1Norm:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var mean = Mean(src.Values, src.Count, src.Length);
                                   var divisor = L1Norm(src.Values, src.Count, mean);
                                   FillValues(Host, in src, ref dst, divisor, scale, mean);
                               };
                            return del;
                        case LpNormalizingEstimatorBase.NormalizerKind.LInf:
                            del =
                               (ref VBuffer<float> dst) =>
                               {
                                   getSrc(ref src);
                                   var mean = Mean(src.Values, src.Count, src.Length);
                                   var divisor = LInfNorm(src.Values, src.Count, mean);
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
                                var divisor = StdDev(src.Values, src.Count, src.Length);
                                FillValues(Host, in src, ref dst, divisor, scale);
                            };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.L2Norm:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L2Norm(src.Values, src.Count);
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.L1Norm:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = L1Norm(src.Values, src.Count);
                               FillValues(Host, in src, ref dst, divisor, scale);
                           };
                        return del;
                    case LpNormalizingEstimatorBase.NormalizerKind.LInf:
                        del =
                           (ref VBuffer<float> dst) =>
                           {
                               getSrc(ref src);
                               var divisor = LInfNorm(src.Values, src.Count);
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
                int count = src.Count;
                int length = src.Length;
                ectx.Assert(Utils.Size(src.Values) >= count);
                ectx.Assert(divisor >= 0);

                if (count == 0)
                {
                    dst = new VBuffer<float>(length, 0, dst.Values, dst.Indices);
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

                if (offset == 0)
                {
                    var dstValues = dst.Values;
                    if (Utils.Size(dstValues) < count)
                        dstValues = new float[count];
                    var dstIndices = dst.Indices;
                    if (!src.IsDense)
                    {
                        if (Utils.Size(dstIndices) < count)
                            dstIndices = new int[count];
                        Array.Copy(src.Indices, dstIndices, count);
                    }

                    CpuMathUtils.Scale(normScale, src.Values, dstValues, count);
                    dst = new VBuffer<float>(length, count, dstValues, dstIndices);

                    return;
                }

                // Subtracting the mean requires a dense representation.
                src.CopyToDense(ref dst);

                if (normScale != 1)
                    CpuMathUtils.ScaleAdd(normScale, -offset, dst.Values.AsSpan(0, length));
                else
                    CpuMathUtils.Add(-offset, dst.Values.AsSpan(0, length));
            }

            /// <summary>
            /// Compute Standard Deviation. In case of both subMean and useStd are true, we technically need to compute variance
            /// based on centered values (i.e. after subtracting the mean). But since the centered
            /// values mean is approximately zero, we can use variance of non-centered values.
            /// </summary>
            private static float StdDev(float[] values, int count, int length)
            {
                Contracts.Assert(0 <= count && count <= length);
                if (count == 0)
                    return 0;
                // We need a mean to compute variance.
                var tmpMean = CpuMathUtils.Sum(values.AsSpan(0, count)) / length;
                float sumSq = 0;
                if (count != length && tmpMean != 0)
                {
                    // Sparse representation.
                    float meanSq = tmpMean * tmpMean;
                    sumSq = (length - count) * meanSq;
                }
                sumSq += CpuMathUtils.SumSq(tmpMean, values.AsSpan(0, count));
                return MathUtils.Sqrt(sumSq / length);
            }

            /// <summary>
            /// Compute Standard Deviation.
            /// We have two overloads of StdDev instead of one with <see cref="Nullable{Float}"/> mean for perf reasons.
            /// </summary>
            private static float StdDev(float[] values, int count, int length, float mean)
            {
                Contracts.Assert(0 <= count && count <= length);
                if (count == 0)
                    return 0;
                float sumSq = 0;
                if (count != length && mean != 0)
                {
                    // Sparse representation.
                    float meanSq = mean * mean;
                    sumSq = (length - count) * meanSq;
                }
                sumSq += CpuMathUtils.SumSq(mean, values.AsSpan(0, count));
                return MathUtils.Sqrt(sumSq / length);
            }

            /// <summary>
            /// Compute L2-norm. L2-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float L2Norm(float[] values, int count, float mean = 0)
            {
                if (count == 0)
                    return 0;
                return MathUtils.Sqrt(CpuMathUtils.SumSq(mean, values.AsSpan(0, count)));
            }

            /// <summary>
            /// Compute L1-norm. L1-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float L1Norm(float[] values, int count, float mean = 0)
            {
                if (count == 0)
                    return 0;
                return CpuMathUtils.SumAbs(mean, values.AsSpan(0, count));
            }

            /// <summary>
            /// Compute LInf-norm. LInf-norm computation doesn't subtract the mean from the source values.
            /// However, we substract the mean here in case subMean is true (if subMean is false, mean is zero).
            /// </summary>
            private static float LInfNorm(float[] values, int count, float mean = 0)
            {
                if (count == 0)
                    return 0;
                return CpuMathUtils.MaxAbsDiff(mean, values.AsSpan(0, count));
            }

            private static float Mean(float[] src, int count, int length)
            {
                if (length == 0 || count == 0)
                    return 0;
                return CpuMathUtils.Sum(src.AsSpan(0, count)) / length;
            }
        }
    }

    public static class LpNormalization
    {
        [TlcModule.EntryPoint(Name = "Transforms.LpNormalizer",
            Desc = LpNormalizingTransformer.Summary,
            UserName = LpNormalizingTransformer.UserNameLP,
            ShortName = LpNormalizingTransformer.ShortNameLP,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""LpNormalize""]/*' />" })]
        public static CommonOutputs.TransformOutput Normalize(IHostEnvironment env, LpNormalizingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LpNormalize", input);
            var xf = LpNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.GlobalContrastNormalizer",
            Desc = LpNormalizingTransformer.GcnSummary,
            UserName = LpNormalizingTransformer.UserNameGn,
            ShortName = LpNormalizingTransformer.ShortNameGn,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""GcNormalize""]/*' />" })]
        public static CommonOutputs.TransformOutput GcNormalize(IHostEnvironment env, LpNormalizingTransformer.GcnArguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "GcNormalize", input);
            var xf = LpNormalizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
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

        internal static class Defaults
        {
            public const NormalizerKind NormKind = NormalizerKind.L2Norm;
            public const bool LpSubMean = false;
            public const bool GcnSubMean = true;
            public const bool UseStdDev = false;
            public const float Scale = 1;
        }

        /// <summary>
        /// Create a <see cref="LpNormalizingEstimatorBase"/> that takes multiple pairs of columns.
        /// </summary>
        public LpNormalizingEstimatorBase(IHostEnvironment env, params LpNormalizingTransformer.ColumnInfoBase[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizingEstimator)), new LpNormalizingTransformer(env, columns))
        {

        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colPair in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colPair.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.Input);
                var reason = LpNormalizingTransformer.TestColumn(col.ItemType);
                if (reason != null)
                    throw Host.ExceptUserArg(nameof(inputSchema), reason);
                if (col.Kind != SchemaShape.Column.VectorKind.Vector)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.Input);
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
                result[colPair.Output] = new SchemaShape.Column(colPair.Output, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Lp Normalizing estimator allow you take columns and normalize them individually by rescaling them to unit norm.
    /// </summary>
    public sealed class LpNormalizingEstimator : LpNormalizingEstimatorBase
    {
        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public LpNormalizingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null,
            NormalizerKind normKind = Defaults.NormKind, bool subMean = Defaults.LpSubMean)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, normKind, subMean)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public LpNormalizingEstimator(IHostEnvironment env, (string input, string output)[] columns,
            NormalizerKind normKind = Defaults.NormKind, bool subMean = Defaults.LpSubMean)
             : this(env, columns.Select(x => new LpNormalizingTransformer.LpNormColumnInfo(x.input, x.output, subMean, normKind)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="LpNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        public LpNormalizingEstimator(IHostEnvironment env, params LpNormalizingTransformer.LpNormColumnInfo[] columns)
            : base(env, columns)
        {
        }
    }

    /// <summary>
    /// Gcn Normalizing estimator allow you take columns and performs global constrast normalization on them.
    /// </summary>
    public sealed class GcnNormalizingEstimator : LpNormalizingEstimatorBase
    {
        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public GcnNormalizingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null,
            bool subMean = Defaults.GcnSubMean, bool useStdDev = Defaults.UseStdDev, float scale = Defaults.Scale)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, subMean, useStdDev, scale)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the normalization on.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public GcnNormalizingEstimator(IHostEnvironment env, (string input, string output)[] columns,
            bool subMean = Defaults.GcnSubMean, bool useStdDev = Defaults.UseStdDev, float scale = Defaults.Scale)
            : this(env, columns.Select(x => new LpNormalizingTransformer.GcnColumnInfo(x.input, x.output, subMean, useStdDev, scale)).ToArray())
        {
        }

        /// <summary>
        /// Create a <see cref="GcnNormalizingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        public GcnNormalizingEstimator(IHostEnvironment env, params LpNormalizingTransformer.GcnColumnInfo[] columns) :
            base(env, columns)
        {

        }
    }

    /// <summary>
    /// Extensions for statically typed <see cref="LpNormalizingEstimator"/>.
    /// </summary>
    public static class LpNormNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, LpNormalizingEstimatorBase.NormalizerKind normKind, bool subMean)
                : base(new Reconciler(normKind, subMean), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly LpNormalizingEstimatorBase.NormalizerKind _normKind;
            private readonly bool _subMean;

            public Reconciler(LpNormalizingEstimatorBase.NormalizerKind normKind, bool subMean)
            {
                _normKind = normKind;
                _subMean = subMean;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new LpNormalizingEstimator(env, pairs.ToArray(), _normKind, _subMean);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public static Vector<float> LpNormalize(this Vector<float> input,
            LpNormalizingEstimatorBase.NormalizerKind normKind = LpNormalizingEstimatorBase.Defaults.NormKind,
            bool subMean = LpNormalizingEstimatorBase.Defaults.LpSubMean) => new OutPipelineColumn(input, normKind, subMean);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="GcnNormalizingEstimator"/>.
    /// </summary>
    public static class GcNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, bool subMean, bool useStdDev, float scale)
                : base(new Reconciler(subMean, useStdDev, scale), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly bool _subMean;
            private readonly bool _useStdDev;
            private readonly float _scale;

            public Reconciler(bool subMean, bool useStdDev, float scale)
            {
                _subMean = subMean;
                _useStdDev = useStdDev;
                _scale = scale;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new GcnNormalizingEstimator(env, pairs.ToArray(), _subMean, _useStdDev, _scale);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public static Vector<float> GlobalContrastNormalize(this Vector<float> input,
            bool subMean = LpNormalizingEstimatorBase.Defaults.GcnSubMean,
            bool useStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev,
            float scale = LpNormalizingEstimatorBase.Defaults.Scale) => new OutPipelineColumn(input, subMean, useStdDev, scale);
    }
}
