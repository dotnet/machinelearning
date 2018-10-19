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
using Microsoft.ML.Core.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System.Collections.Generic;

[assembly: LoadableClass(PcaTransformer.Summary, typeof(IDataTransform), typeof(PcaTransformer), typeof(PcaTransformer.Arguments), typeof(SignatureDataTransform),
    PcaTransformer.UserName, PcaTransformer.LoaderSignature, PcaTransformer.ShortName)]

[assembly: LoadableClass(PcaTransformer.Summary, typeof(IDataTransform), typeof(PcaTransformer), null, typeof(SignatureLoadDataTransform),
    PcaTransformer.UserName, PcaTransformer.LoaderSignature)]

[assembly: LoadableClass(PcaTransformer.Summary, typeof(PcaTransformer), null, typeof(SignatureLoadModel),
    PcaTransformer.UserName, PcaTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(PcaTransformer), null, typeof(SignatureLoadRowMapper),
    PcaTransformer.UserName, PcaTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(PcaTransformer), null, typeof(SignatureEntryPointModule), PcaTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
    public sealed class PcaTransformer : OneToOneTransformerBase
    {
        internal static class Defaults
        {
            public const string WeightColumn = null;
            public const int Rank = 20;
            public const int Oversampling = 20;
            public const bool Center = true;
            public const int Seed = 0;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Multiple, HelpText = "The name of the weight column", ShortName = "weight", Purpose = SpecialPurpose.ColumnName)]
            public string WeightColumn = Defaults.WeightColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of components in the PCA", ShortName = "k")]
            public int Rank = Defaults.Rank;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Oversampling parameter for randomized PCA training", ShortName = "over")]
            public int Oversampling = Defaults.Oversampling;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If enabled, data is centered to be zero mean")]
            public bool Center = Defaults.Center;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed for random number generation")]
            public int Seed = Defaults.Seed;
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

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly string WeightColumn;
            public readonly int Rank;
            public readonly int Oversampling;
            public readonly bool Center;
            public readonly int? Seed;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            public ColumnInfo(string input,
                              string output,
                              string weightColumn = PcaTransformer.Defaults.WeightColumn,
                              int rank = PcaTransformer.Defaults.Rank,
                              int overSampling = PcaTransformer.Defaults.Oversampling,
                              bool center = PcaTransformer.Defaults.Center,
                              int? seed = null)
            {
                Input = input;
                Output = output;
                WeightColumn = weightColumn;
                Rank = rank;
                Oversampling = overSampling;
                Center = center;
                Seed = seed;
            }
        }

        private sealed class TransformInfo
        {
            public readonly int Dimension;
            public readonly int Rank;

            public Float[][] Eigenvectors;
            public Float[] MeanProjected;

            public TransformInfo(int rank, int dim)
            {
                Dimension = dim;
                Rank = rank;
                Contracts.CheckUserArg(0 < Rank && Rank <= Dimension, nameof(Rank), "Rank must be positive, and at most the dimension of untransformed data");
            }

            public TransformInfo(ModelLoadContext ctx)
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
        internal const string ShortName = "Pca2";

        public const string LoaderSignature = "PcaTransformer";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCA FUN2",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PcaTransformer).Assembly.FullName);
        }

        // These are parallel to Infos.
        private readonly ColumnType[] _outputColumnTypes;
        private readonly TransformInfo[] _transformInfos;
        private readonly int[] _weightColumnIndices;
        private readonly int[] _inputColumnIndices;
        private readonly ColumnType[] _inputColumnTypes;
        private readonly int _numColumns;

        private const string RegistrationName = "Pca2";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public PcaTransformer(IHostEnvironment env, IDataView input, ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(PcaTransformer)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);

            _numColumns = ColumnPairs.Length;
            _transformInfos = new TransformInfo[_numColumns];
            _weightColumnIndices = new int[_numColumns];
            _inputColumnIndices = new int[_numColumns];
            _inputColumnTypes = new ColumnType[_numColumns];

            for (int i = 0; i < _numColumns; i++)
            {
                var col = columns[i];
                if (!input.Schema.TryGetColumnIndex(col.Input, out _inputColumnIndices[i]))
                    throw Host.ExceptSchemaMismatch(nameof(col.Input), "input", col.Input);
                _inputColumnTypes[i] = input.Schema[_inputColumnIndices[i]].Type;
                Host.Check(_inputColumnTypes[i].IsKnownSizeVector && _inputColumnTypes[i].VectorSize > 1,
                    "Pca transform can only be applied to columns with known dimensionality greater than 1");
                _transformInfos[i] = new TransformInfo(col.Rank, _inputColumnTypes[i].ValueCount);
                Host.CheckUserArg(col.Oversampling >= 0, nameof(col.Oversampling), "Oversampling must be non-negative");
                _weightColumnIndices[i] = -1;
                var weightColumn = col.WeightColumn;
                if (weightColumn != null)
                {
                    if (!input.Schema.TryGetColumnIndex(weightColumn, out _weightColumnIndices[i]))
                        throw Host.Except("weight column '{0}' does not exist", weightColumn);
                    var type = input.Schema.GetColumnType(_weightColumnIndices[i]);
                    Host.CheckUserArg(type == NumberType.Float, nameof(weightColumn));
                }
            }

            Train(columns, _transformInfos, input);
            _outputColumnTypes = InitColumnTypes();
        }

        private PcaTransformer(IHost host, ModelLoadContext ctx)
         : base(host, ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // transformInfos
            Host.AssertNonEmpty(ColumnPairs);
            _numColumns = ColumnPairs.Length;
            _transformInfos = new TransformInfo[_numColumns];
            for (int i = 0; i < _numColumns; i++)
                _transformInfos[i] = new TransformInfo(ctx);
            _outputColumnTypes = InitColumnTypes();
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.Column, nameof(args.Column));
            var cols = args.Column.Select(item => new ColumnInfo(
                        item.Source,
                        item.Name,
                        item.WeightColumn,
                        item.Rank ?? args.Rank,
                        item.Oversampling ?? args.Oversampling,
                        item.Center ?? args.Center,
                        item.Seed ?? args.Seed)).ToArray();
            return new PcaTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static PcaTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(PcaTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new PcaTransformer(host, ctx);
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
            SaveColumns(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
                _transformInfos[i].Save(ctx);
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            //Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private void Train(ColumnInfo[] columns, TransformInfo[] transformInfos, IDataView trainingData)
        {
            var y = new Float[_numColumns][][];
            var omega = new Float[_numColumns][][];
            var mean = new Float[_numColumns][];
            var oversampledRank = new int[_numColumns];
            var rnd = Host.Rand;
            Double totalMemoryUsageEstimate = 0;
            for (int iinfo = 0; iinfo < _numColumns; iinfo++)
            {
                oversampledRank[iinfo] = Math.Min(transformInfos[iinfo].Rank + columns[iinfo].Oversampling, transformInfos[iinfo].Dimension);

                //exact: (size of the 2 big matrices + other minor allocations) / (2^30)
                Double colMemoryUsageEstimate = 2.0 * transformInfos[iinfo].Dimension * oversampledRank[iinfo] * sizeof(Float) / 1e9;
                totalMemoryUsageEstimate += colMemoryUsageEstimate;
                if (colMemoryUsageEstimate > 2)
                {
                    using (var ch = Host.Start("Memory usage"))
                    {
                        ch.Info("Estimate memory usage for transforming column {1}: {0:G2} GB. If running out of memory, reduce rank and oversampling factor.",
                            colMemoryUsageEstimate, ColumnPairs[iinfo].input);
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

                if (columns[iinfo].Center)
                    mean[iinfo] = new Float[transformInfos[iinfo].Dimension];
            }
            if (totalMemoryUsageEstimate > 2)
            {
                using (var ch = Host.Start("Memory usage"))
                {
                    ch.Info("Estimate memory usage for all PCA transforms: {0:G2} GB. If running out of memory, reduce ranks and oversampling factors.",
                        totalMemoryUsageEstimate);
                }
            }

            Project(trainingData, mean, omega, y, transformInfos);

            for (int iinfo = 0; iinfo < transformInfos.Length; iinfo++)
            {
                //Orthonormalize Y in-place using stabilized Gram Schmidt algorithm
                //Ref: https://en.wikipedia.org/wiki/Gram-Schmidt#Algorithm
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
            Host.Assert(mean.Length == omega.Length && omega.Length == y.Length && y.Length == _numColumns);
            for (int i = 0; i < omega.Length; i++)
                Contracts.Assert(omega[i].Length == y[i].Length);

            // set y to be all zeros
            for (int iinfo = 0; iinfo < y.Length; iinfo++)
            {
                for (int i = 0; i < y[iinfo].Length; i++)
                    Array.Clear(y[iinfo][i], 0, y[iinfo][i].Length);
            }

            bool[] center = Enumerable.Range(0, mean.Length).Select(i => mean[i] != null).ToArray();

            Double[] totalColWeight = new Double[_numColumns];

            bool[] activeColumns = new bool[trainingData.Schema.ColumnCount];
            for (int iinfo = 0; iinfo < _numColumns; iinfo++)
            {
                activeColumns[_inputColumnIndices[iinfo]] = true;
                if (_weightColumnIndices[iinfo] >= 0)
                    activeColumns[_weightColumnIndices[iinfo]] = true;
            }

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var weightGetters = new ValueGetter<Float>[_numColumns];
                var columnGetters = new ValueGetter<VBuffer<Float>>[_numColumns];
                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    if (_weightColumnIndices[iinfo] >= 0)
                        weightGetters[iinfo] = cursor.GetGetter<Float>(_weightColumnIndices[iinfo]);
                    columnGetters[iinfo] = cursor.GetGetter<VBuffer<Float>>(_inputColumnIndices[iinfo]);
                }

                var features = default(VBuffer<Float>);
                while (cursor.MoveNext())
                {
                    for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                    {
                        Contracts.Check(_inputColumnTypes[iinfo].IsVector && _inputColumnTypes[iinfo].ItemType.IsNumber,
                            "PCA transform can only be performed on numeric columns of dimension > 1");

                        Float weight = 1;
                        weightGetters[iinfo]?.Invoke(ref weight);
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

                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    if (totalColWeight[iinfo] <= 0)
                        throw Host.Except("Empty data in column '{0}'", ColumnPairs[iinfo].input);
                }

                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
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
            Host.Assert(ColumnPairs.Length == _transformInfos.Length);
            var types = _transformInfos.Select(tInfo => new VectorType(NumberType.Float, tInfo.Rank)).ToArray();
            return types;
        }

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _outputColumnTypes;
            // Todo: replace with ColMapNewToOld
            private readonly ColumnType[] _inputColumnTypes;
            private readonly int[] _inputColumnIndices;
            private readonly PcaTransformer _parent;
            private readonly int _numColumns;

            public Mapper(PcaTransformer parent, Schema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _numColumns = parent._numColumns;
                _outputColumnTypes = parent.InitColumnTypes();
                _inputColumnTypes = new ColumnType[_numColumns];
                _inputColumnIndices = new int[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                {
                    var inputColName = _parent.ColumnPairs[i].input;
                    if (!inputSchema.TryGetColumnIndex(inputColName, out _inputColumnIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputColName), "input", inputColName);
                    _inputColumnTypes[i] = inputSchema[_inputColumnIndices[i]].Type;
                    Host.Check(_inputColumnTypes[i].IsKnownSizeVector && _inputColumnTypes[i].VectorSize > 1,
                        "Pca transform can only be applied to columns with known dimensionality greater than 1");
                    if (_inputColumnTypes[i].VectorSize != _parent._transformInfos[i].Dimension)
                    {
                        var msg = $"Dimension of column ${inputColName} is ${_inputColumnTypes[i].VectorSize}, which doesn't match the expected size ${_parent._transformInfos[i].Dimension}";
                        throw Host.Except(msg);
                    }
                }
                // Ivan't comment:
                //var getSrc = input.GetGetter<ReadOnlyMemory<char>>(ColMapNewToOld[iinfo]);
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _outputColumnTypes[i], null);
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _numColumns);
                disposer = null;

                var srcGetter = input.GetGetter<VBuffer<float>>(_inputColumnIndices[iinfo]);
                var src = default(VBuffer<float>);

                ValueGetter<VBuffer<float>> dstGetter = (ref VBuffer<float> dst) =>
                    {
                        srcGetter(ref src);
                        TransformFeatures(Host, ref src, ref dst, _parent._transformInfos[iinfo]);
                    };

                return dstGetter;
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

            //protected virtual void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
            //{
            //    // By default, there are no extra checks.
            //}

        }

        [TlcModule.EntryPoint(Name = "Transforms.PcaCalculator2",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/member[@name=""PCA""]/*' />",
                                 @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/example[@name=""PcaCalculator""]/*' />"})]
        public static CommonOutputs.TransformOutput Calculate(IHostEnvironment env, Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Pca", input);
            var view = PcaTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    public sealed class PcaEstimator2 : IEstimator<PcaTransformer>
    {
        private readonly IHost _host;
        private readonly PcaTransformer.ColumnInfo[] _columns;

        public PcaEstimator2(IHostEnvironment env, string inputColumn, string outputColumn = null,
            string weightColumn = PcaTransformer.Defaults.WeightColumn, int rank = PcaTransformer.Defaults.Rank,
            int overSampling = PcaTransformer.Defaults.Oversampling, bool center = PcaTransformer.Defaults.Center,
            int? seed = null)
            : this(env, new PcaTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn, weightColumn, rank, overSampling, center, seed))
        {
        }

        public PcaEstimator2(IHostEnvironment env, params PcaTransformer.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(PcaEstimator2));
            _columns = columns;
        }

        //TODO: move the dosctrings above
        ///// <summary>
        ///// Convinence constructor for simple one column case
        ///// </summary>
        ///// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
        ///// <param name="env">The environment.</param>
        ///// <param name="inputColumn">Input column to apply PCA on.</param>
        ///// <param name="outputColumn">Output column. Null means <paramref name="inputColumn"/> is replaced.</param>
        ///// <param name="rank">The number of components in the PCA.</param>
        ///// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        //public PcaEstimator2(IHostEnvironment env, string inputColumn, string outputColumn = null,
        //    int rank = PcaTransformer.Defaults.Rank,
        //    Action<PcaTransformer.Arguments> advancedSettings = null)
        //    : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, rank, advancedSettings)
        //{
        //}

        ///// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
        ///// <param name="env">The environment.</param>
        ///// <param name="columns">Pairs of columns to run the PCA on.</param>
        ///// <param name="rank">The number of components in the PCA.</param>
        ///// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        //public PcaEstimator2(IHostEnvironment env, (string input, string output)[] columns,
        //    int rank = PcaTransformer.Defaults.Rank,
        //    Action<PcaTransformer.Arguments> advancedSettings = null)
        //{
        //    Contracts.CheckValue(env, nameof(env));
        //    _host = env.Register(nameof(PcaEstimator2));

        //    foreach (var (input, output) in columns)
        //    {
        //        _host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
        //        _host.CheckValue(output, nameof(output));
        //    }

        //    var args = new PcaTransformer.Arguments();
        //    args.Column = columns.Select(x => new PcaTransformer.Column { Source = x.input, Name = x.output }).ToArray();
        //    args.Rank = rank;
        //    advancedSettings?.Invoke(args);
        //    _columns = PcaTransformer.ArgumentsToColumnInfos(args);
        //}

        public PcaTransformer Fit(IDataView input) => new PcaTransformer(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.ItemType.RawKind != DataKind.R4 || col.Kind != SchemaShape.Column.VectorKind.Vector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output,
                    SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
            }

            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Extensions for statically typed <see cref="PcaEstimator"/>.
    /// </summary>
    public static class PcaEstimatorExtensions2
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, string weightColumn, int rank,
                                     int overSampling, bool center, int? seed = null)
                : base(new Reconciler(weightColumn, rank, overSampling, center, seed))
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly PcaTransformer.ColumnInfo _colInfo;

            public Reconciler(string weightColumn, int rank, int overSampling, bool center, int? seed = null)
            {
                _colInfo = new PcaTransformer.ColumnInfo(
                    null, null, weightColumn, rank, overSampling, center, seed);
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                // Only one column is allowed.
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutPipelineColumn)toOutput[0];
                var inputColName = inputNames[outCol.Input];
                var outputColName = outputNames[outCol];
                return new PcaEstimator2(env, inputColName, outputColName,
                                         _colInfo.WeightColumn, _colInfo.Rank, _colInfo.Oversampling,
                                         _colInfo.Center, _colInfo.Seed);
            }
        }

        // TODO: fix docstrings
        // /// <summary>Replace current vector with its principal components. Can significantly reduce size of vector.</summary>
        // /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        // /// <param name="input">The column to apply PCA to.</param>
        // /// <param name="rank">The number of components in the PCA.</param>
        // /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static Vector<float> ToPrincipalComponents(this Vector<float> input,
            string weightColumn = PcaTransformer.Defaults.WeightColumn,
            int rank = PcaTransformer.Defaults.Rank,
            int overSampling = PcaTransformer.Defaults.Oversampling,
            bool center = PcaTransformer.Defaults.Center,
            int? seed = null) => new OutPipelineColumn(input, weightColumn, rank, overSampling, center, seed);
    }
}
