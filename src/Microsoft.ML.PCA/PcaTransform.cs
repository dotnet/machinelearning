// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(PcaTransform.Summary, typeof(PcaTransform), typeof(PcaTransform.Arguments), typeof(SignatureDataTransform),
    PcaTransform.UserName, PcaTransform.LoaderSignature, PcaTransform.ShortName)]

[assembly: LoadableClass(PcaTransform.Summary, typeof(PcaTransform), null, typeof(SignatureLoadDataTransform),
    PcaTransform.UserName, PcaTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(PcaTransform), null, typeof(SignatureEntryPointModule), PcaTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='./doc.xml' path='docs/members/member[@name="PCA"]/*' />
    public sealed class PcaTransform : OneToOneTransformBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Multiple, HelpText = "The name of the weight column", ShortName = "weight", Purpose = SpecialPurpose.ColumnName)]
            public string WeightColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of components in the PCA", ShortName = "k")]
            public int Rank = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Oversampling parameter for randomized PCA training", ShortName = "over")]
            public int Oversampling = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If enabled, data is centered to be zero mean")]
            public bool Center = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed for random number generation")]
            public int Seed = 0;
        }

        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "The name of the weight column", ShortName = "weight")]
            public string WeightColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of components in the PCA", ShortName = "k")]
            public int? Rank;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Oversampling parameter for randomized PCA training", ShortName = "over")]
            public int? Oversampling;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If enabled, data is centered to be zero mean", ShortName = "center")]
            public bool? Center;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed for random number generation", ShortName = "seed")]
            public int? Seed;

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
                if (!string.IsNullOrEmpty(WeightColumn) || Rank != null || Oversampling != null ||
                    Center != null || Seed != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        private sealed class TransformInfo
        {
            public readonly int Dimension;
            public readonly int Rank;

            public Float[][] Eigenvectors;
            public Float[] MeanProjected;

            public TransformInfo(Column item, Arguments args, int d)
            {
                Dimension = d;
                Rank = item.Rank ?? args.Rank;
                Contracts.CheckUserArg(0 < Rank && Rank <= Dimension, nameof(item.Rank), "Rank must be positive, and at most the dimension of untransformed data");
            }

            public TransformInfo(ModelLoadContext ctx, int colValueCount)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: Dimension
                // int: Rank
                // for i=0,..,Rank-1:
                //   Float[]: the i'th eigenvector
                // int: the size of MeanProjected (0 if it is null)
                // Float[]: MeanProjected

                Dimension = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Dimension == colValueCount);

                Rank = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < Rank && Rank <= Dimension);

                Eigenvectors = new Float[Rank][];
                for (int i = 0; i < Rank; i++)
                {
                    Eigenvectors[i] = ctx.Reader.ReadFloatArray(Dimension);
                    Contracts.CheckDecode(FloatUtils.IsFinite(Eigenvectors[i], Eigenvectors[i].Length));
                }

                MeanProjected = ctx.Reader.ReadFloatArray();
                Contracts.CheckDecode(MeanProjected == null || (MeanProjected.Length == Rank && FloatUtils.IsFinite(MeanProjected, MeanProjected.Length)));
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: Dimension
                // int: Rank
                // for i=0,..,Rank-1:
                //   Float[]: the i'th eigenvector
                // int: the size of MeanProjected (0 if it is null)
                // Float[]: MeanProjected

                Contracts.Assert(0 < Rank && Rank <= Dimension);
                ctx.Writer.Write(Dimension);
                ctx.Writer.Write(Rank);
                for (int i = 0; i < Rank; i++)
                {
                    Contracts.Assert(FloatUtils.IsFinite(Eigenvectors[i], Eigenvectors[i].Length));
                    ctx.Writer.WriteFloatsNoCount(Eigenvectors[i], Dimension);
                }
                Contracts.Assert(MeanProjected == null || (MeanProjected.Length == Rank && FloatUtils.IsFinite(MeanProjected, Rank)));
                ctx.Writer.WriteFloatArray(MeanProjected);
            }

            internal void ProjectMean(Float[] mean)
            {
                Contracts.AssertValue(Eigenvectors);
                if (mean == null)
                {
                    MeanProjected = null;
                    return;
                }

                MeanProjected = new Float[Rank];
                for (var i = 0; i < Rank; ++i)
                    MeanProjected[i] = VectorUtils.DotProduct(Eigenvectors[i], mean);
            }
        }

        internal const string Summary = "PCA is a dimensionality-reduction transform which computes the projection of a numeric vector onto a low-rank subspace.";
        internal const string UserName = "Principal Component Analysis Transform";
        internal const string ShortName = "Pca";

        public const string LoaderSignature = "PcaTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCA FUNC",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        // These are parallel to Infos.
        private readonly ColumnType[] _types;
        private readonly TransformInfo[] _transformInfos;

        private readonly int[] _oversampling;
        private readonly bool[] _center;
        private readonly int[] _weightColumnIndex;

        private const string RegistrationName = "Pca";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public PcaTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsFloatItem)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _transformInfos = new TransformInfo[args.Column.Length];
            _oversampling = new int[args.Column.Length];
            _center = new bool[args.Column.Length];
            _weightColumnIndex = new int[args.Column.Length];
            for (int i = 0; i < _transformInfos.Length; i++)
            {
                Host.Check(Infos[i].TypeSrc.VectorSize > 1, "Pca transform can only be applied to columns with known dimensionality greater than 1");
                _transformInfos[i] = new TransformInfo(args.Column[i], args, Infos[i].TypeSrc.ValueCount);
                _center[i] = args.Column[i].Center ?? args.Center;
                _oversampling[i] = args.Column[i].Oversampling ?? args.Oversampling;
                Host.CheckUserArg(_oversampling[i] >= 0, nameof(args.Oversampling), "Oversampling must be non-negative");
                _weightColumnIndex[i] = -1;
                var weightColumn = args.Column[i].WeightColumn ?? args.WeightColumn;
                if (weightColumn != null)
                {
                    if (!Source.Schema.TryGetColumnIndex(weightColumn, out _weightColumnIndex[i]))
                        throw Host.Except("weight column '{0}' does not exist", weightColumn);
                    var type = Source.Schema.GetColumnType(_weightColumnIndex[i]);
                    Host.CheckUserArg(type == NumberType.Float, nameof(args.WeightColumn));
                }
            }

            Train(args, _transformInfos, input);

            _types = InitColumnTypes();
        }

        private PcaTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsFloatItem)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // transformInfos
            Host.AssertNonEmpty(Infos);
            _transformInfos = new TransformInfo[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
                _transformInfos[i] = new TransformInfo(ctx, Infos[i].TypeSrc.ValueCount);
            _types = InitColumnTypes();
        }

        public static PcaTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <remainder handled in ctors>
            int cbFloat = ctx.Reader.ReadInt32();
            h.CheckDecode(cbFloat == sizeof(Float));
            return h.Apply("Loading Model", ch => new PcaTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // transformInfos
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
                _transformInfos[i].Save(ctx);
        }

        private void Train(Arguments args, TransformInfo[] transformInfos, IDataView trainingData)
        {
            var y = new Float[transformInfos.Length][][];
            var omega = new Float[transformInfos.Length][][];
            var mean = new Float[transformInfos.Length][];

            var oversampledRank = new int[transformInfos.Length];
            var rnd = Host.Rand;
            Double totalMemoryUsageEstimate = 0;
            for (int iinfo = 0; iinfo < transformInfos.Length; iinfo++)
            {
                oversampledRank[iinfo] = Math.Min(transformInfos[iinfo].Rank + _oversampling[iinfo], transformInfos[iinfo].Dimension);

                //exact: (size of the 2 big matrices + other minor allocations) / (2^30)
                Double colMemoryUsageEstimate = 2.0 * transformInfos[iinfo].Dimension * oversampledRank[iinfo] * sizeof(Float) / 1e9;
                totalMemoryUsageEstimate += colMemoryUsageEstimate;
                if (colMemoryUsageEstimate > 2)
                {
                    using (var ch = Host.Start("Memory usage"))
                    {
                        ch.Info("Estimate memory usage for transforming column {1}: {0:G2} GB. If running out of memory, reduce rank and oversampling factor.",
                            colMemoryUsageEstimate, Infos[iinfo].Name);
                        ch.Done();
                    }
                }

                y[iinfo] = new Float[oversampledRank[iinfo]][];
                omega[iinfo] = new Float[oversampledRank[iinfo]][];
                for (int i = 0; i < oversampledRank[iinfo]; i++)
                {
                    y[iinfo][i] = new Float[transformInfos[iinfo].Dimension];
                    omega[iinfo][i] = new Float[transformInfos[iinfo].Dimension];
                    for (int j = 0; j < transformInfos[iinfo].Dimension; j++)
                    {
                        omega[iinfo][i][j] = (Float)Stats.SampleFromGaussian(rnd);
                    }
                }

                if (_center[iinfo])
                    mean[iinfo] = new Float[transformInfos[iinfo].Dimension];
            }
            if (totalMemoryUsageEstimate > 2)
            {
                using (var ch = Host.Start("Memory usage"))
                {
                    ch.Info("Estimate memory usage for all PCA transforms: {0:G2} GB. If running out of memory, reduce ranks and oversampling factors.",
                        totalMemoryUsageEstimate);
                    ch.Done();
                }
            }

            Project(trainingData, mean, omega, y, transformInfos);

            for (int iinfo = 0; iinfo < transformInfos.Length; iinfo++)
            {
                //Orthonormalize Y in-place using stabilized Gram Schmidt algorithm 
                //Ref: http://en.wikipedia.org/wiki/Gram-Schmidt#Algorithm
                for (var i = 0; i < oversampledRank[iinfo]; ++i)
                {
                    var v = y[iinfo][i];
                    VectorUtils.ScaleBy(v, 1 / VectorUtils.Norm(y[iinfo][i])); // normalize

                    // Make the next vectors in the queue orthogonal to the orthonormalized vectors
                    for (var j = i + 1; j < oversampledRank[iinfo]; ++j)
                        VectorUtils.AddMult(v, y[iinfo][j], -VectorUtils.DotProduct(v, y[iinfo][j])); //subtract the projection of y[j] on v
                }
            }
            var q = y; // q in QR decomposition

            var b = omega; // reuse the memory allocated by Omega
            Project(trainingData, mean, q, b, transformInfos);

            for (int iinfo = 0; iinfo < transformInfos.Length; iinfo++)
            {
                //Compute B2 = B' * B
                var b2 = new Float[oversampledRank[iinfo] * oversampledRank[iinfo]];
                for (var i = 0; i < oversampledRank[iinfo]; ++i)
                {
                    for (var j = i; j < oversampledRank[iinfo]; ++j)
                        b2[i * oversampledRank[iinfo] + j] = b2[j * oversampledRank[iinfo] + i] = VectorUtils.DotProduct(b[iinfo][i], b[iinfo][j]);
                }

                Float[] smallEigenvalues; // eigenvectors and eigenvalues of the small matrix B2.
                Float[] smallEigenvectors;

                EigenUtils.EigenDecomposition(b2, out smallEigenvalues, out smallEigenvectors);
                transformInfos[iinfo].Eigenvectors = PostProcess(b[iinfo], smallEigenvalues, smallEigenvectors, transformInfos[iinfo].Dimension, oversampledRank[iinfo]);
                transformInfos[iinfo].ProjectMean(mean[iinfo]);
            }
        }

        //Project the covariance matrix A on to Omega: Y <- A * Omega
        //A = X' * X / n, where X = data - mean
        //Note that the covariance matrix is not computed explicitly
        private void Project(IDataView trainingData, Float[][] mean, Float[][][] omega, Float[][][] y, TransformInfo[] transformInfos)
        {
            Host.Assert(mean.Length == omega.Length && omega.Length == y.Length && y.Length == Infos.Length);
            for (int i = 0; i < omega.Length; i++)
                Contracts.Assert(omega[i].Length == y[i].Length);

            // set y to be all zeros
            for (int iinfo = 0; iinfo < y.Length; iinfo++)
            {
                for (int i = 0; i < y[iinfo].Length; i++)
                    Array.Clear(y[iinfo][i], 0, y[iinfo][i].Length);
            }

            bool[] center = Enumerable.Range(0, mean.Length).Select(i => mean[i] != null).ToArray();

            Double[] totalColWeight = new Double[Infos.Length];

            bool[] activeColumns = new bool[Source.Schema.ColumnCount];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                activeColumns[Infos[iinfo].Source] = true;
                if (_weightColumnIndex[iinfo] >= 0)
                    activeColumns[_weightColumnIndex[iinfo]] = true;
            }
            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var weightGetters = new ValueGetter<Float>[Infos.Length];
                var columnGetters = new ValueGetter<VBuffer<Float>>[Infos.Length];
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    if (_weightColumnIndex[iinfo] >= 0)
                        weightGetters[iinfo] = cursor.GetGetter<Float>(_weightColumnIndex[iinfo]);
                    columnGetters[iinfo] = cursor.GetGetter<VBuffer<Float>>(Infos[iinfo].Source);
                }

                var features = default(VBuffer<Float>);
                while (cursor.MoveNext())
                {
                    for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    {
                        Contracts.Check(Infos[iinfo].TypeSrc.IsVector && Infos[iinfo].TypeSrc.ItemType.IsNumber,
                            "PCA transform can only be performed on numeric columns of dimension > 1");

                        Float weight = 1;
                        if (weightGetters[iinfo] != null)
                            weightGetters[iinfo](ref weight);
                        columnGetters[iinfo](ref features);

                        if (FloatUtils.IsFinite(weight) && weight >= 0 && (features.Count == 0 || FloatUtils.IsFinite(features.Values, features.Count)))
                        {
                            totalColWeight[iinfo] += weight;

                            if (center[iinfo])
                                VectorUtils.AddMult(ref features, mean[iinfo], weight);

                            for (int i = 0; i < omega[iinfo].Length; i++)
                                VectorUtils.AddMult(ref features, y[iinfo][i], weight * VectorUtils.DotProductWithOffset(omega[iinfo][i], 0, ref features));
                        }
                    }
                }

                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    if (totalColWeight[iinfo] <= 0)
                        throw Host.Except("Empty data in column '{0}'", Source.Schema.GetColumnName(Infos[iinfo].Source));
                }

                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    var invn = (Float)(1 / totalColWeight[iinfo]);

                    for (var i = 0; i < omega[iinfo].Length; ++i)
                        VectorUtils.ScaleBy(y[iinfo][i], invn);

                    if (center[iinfo])
                    {
                        VectorUtils.ScaleBy(mean[iinfo], invn);
                        for (int i = 0; i < omega[iinfo].Length; i++)
                            VectorUtils.AddMult(mean[iinfo], y[iinfo][i], -VectorUtils.DotProduct(omega[iinfo][i], mean[iinfo]));
                    }
                }
            }
        }

        //return Y * eigenvectors / eigenvalues
        // REVIEW: improve
        private Float[][] PostProcess(Float[][] y, Float[] sigma, Float[] z, int d, int k)
        {
            var pinv = new Float[k];
            var tmp = new Float[k];

            for (int i = 0; i < k; i++)
                pinv[i] = (Float)(1.0) / ((Float)(1e-6) + sigma[i]);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    tmp[j] = 0;
                    for (int l = 0; l < k; l++)
                        tmp[j] += y[l][i] * z[j * k + l];
                }
                for (int j = 0; j < k; j++)
                    y[j][i] = pinv[j] * tmp[j];
            }

            return y;
        }

        private ColumnType[] InitColumnTypes()
        {
            Host.Assert(Infos.Length == _transformInfos.Length);
            var types = new ColumnType[Infos.Length];
            for (int i = 0; i < _transformInfos.Length; i++)
                types[i] = new VectorType(NumberType.Float, _transformInfos[i].Rank);
            Metadata.Seal();
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Utils.Size(_types));
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var getSrc = GetSrcGetter<VBuffer<Float>>(input, iinfo);
            var src = default(VBuffer<Float>);
            var trInfo = _transformInfos[iinfo];
            ValueGetter<VBuffer<Float>> del =
                (ref VBuffer<Float> dst) =>
                {
                    getSrc(ref src);
                    TransformFeatures(Host, ref src, ref dst, trInfo);
                };
            return del;
        }

        private static void TransformFeatures(IExceptionContext ectx, ref VBuffer<Float> src, ref VBuffer<Float> dst, TransformInfo transformInfo)
        {
            ectx.Check(src.Length == transformInfo.Dimension);

            var values = dst.Values;
            if (Utils.Size(values) < transformInfo.Rank)
                values = new Float[transformInfo.Rank];

            for (int i = 0; i < transformInfo.Rank; i++)
            {
                values[i] = VectorUtils.DotProductWithOffset(transformInfo.Eigenvectors[i], 0, ref src) -
                    (transformInfo.MeanProjected == null ? 0 : transformInfo.MeanProjected[i]);
            }

            dst = new VBuffer<Float>(transformInfo.Rank, values, dst.Indices);
        }

        [TlcModule.EntryPoint(Name = "Transforms.PcaCalculator", 
            Desc = Summary,
            UserName = UserName, 
            ShortName = ShortName, 
            XmlInclude = new[] { @"<include file='../Microsoft.ML.PCA/doc.xml' path='docs/members/member[@name=""PCA""]/*' />" })]
        public static CommonOutputs.TransformOutput Calculate(IHostEnvironment env, Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Pca", input);
            var view = new PcaTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
