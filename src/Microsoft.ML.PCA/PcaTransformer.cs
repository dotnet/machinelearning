// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(PrincipalComponentAnalysisTransformer.Summary, typeof(IDataTransform), typeof(PrincipalComponentAnalysisTransformer), typeof(PrincipalComponentAnalysisTransformer.Options), typeof(SignatureDataTransform),
    PrincipalComponentAnalysisTransformer.UserName, PrincipalComponentAnalysisTransformer.LoaderSignature, PrincipalComponentAnalysisTransformer.ShortName)]

[assembly: LoadableClass(PrincipalComponentAnalysisTransformer.Summary, typeof(IDataTransform), typeof(PrincipalComponentAnalysisTransformer), null, typeof(SignatureLoadDataTransform),
    PrincipalComponentAnalysisTransformer.UserName, PrincipalComponentAnalysisTransformer.LoaderSignature)]

[assembly: LoadableClass(PrincipalComponentAnalysisTransformer.Summary, typeof(PrincipalComponentAnalysisTransformer), null, typeof(SignatureLoadModel),
    PrincipalComponentAnalysisTransformer.UserName, PrincipalComponentAnalysisTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(PrincipalComponentAnalysisTransformer), null, typeof(SignatureLoadRowMapper),
    PrincipalComponentAnalysisTransformer.UserName, PrincipalComponentAnalysisTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(PrincipalComponentAnalysisTransformer), null, typeof(SignatureEntryPointModule), PrincipalComponentAnalysisTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
    public sealed class PrincipalComponentAnalysisTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.Multiple, HelpText = "The name of the weight column", ShortName = "weight", Purpose = SpecialPurpose.ColumnName)]
            public string ExampleWeightColumnName = PrincipalComponentAnalyzer.Defaults.WeightColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of components in the PCA", ShortName = "k")]
            public int Rank = PrincipalComponentAnalyzer.Defaults.Rank;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Oversampling parameter for randomized PCA training", ShortName = "over")]
            public int Oversampling = PrincipalComponentAnalyzer.Defaults.Oversampling;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If enabled, data is centered to be zero mean")]
            public bool Center = PrincipalComponentAnalyzer.Defaults.EnsureZeroMean;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed for random number generation")]
            public int Seed = PrincipalComponentAnalyzer.Defaults.Seed;
        }

        internal class Column : OneToOneColumn
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

            public float[][] Eigenvectors;
            public float[] MeanProjected;

            public DataViewType OutputType => new VectorType(NumberDataViewType.Single, Rank);

            public TransformInfo(int rank, int dim)
            {
                Dimension = dim;
                Rank = rank;
                Contracts.CheckParam(0 < Rank && Rank <= Dimension, nameof(Rank), "Rank must be positive, and at most the dimension of untransformed data");
            }

            public TransformInfo(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: Dimension
                // int: Rank
                // for i=0,..,Rank-1:
                //   float[]: the i'th eigenvector
                // int: the size of MeanProjected (0 if it is null)
                // float[]: MeanProjected

                Dimension = ctx.Reader.ReadInt32();
                Rank = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < Rank && Rank <= Dimension);

                Eigenvectors = new float[Rank][];
                for (int i = 0; i < Rank; i++)
                {
                    Eigenvectors[i] = ctx.Reader.ReadFloatArray(Dimension);
                    Contracts.CheckDecode(FloatUtils.IsFinite(Eigenvectors[i]));
                }

                MeanProjected = ctx.Reader.ReadFloatArray();
                Contracts.CheckDecode(MeanProjected == null || (MeanProjected.Length == Rank && FloatUtils.IsFinite(MeanProjected)));
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: Dimension
                // int: Rank
                // for i=0,..,Rank-1:
                //   float[]: the i'th eigenvector
                // int: the size of MeanProjected (0 if it is null)
                // float[]: MeanProjected

                Contracts.Assert(0 < Rank && Rank <= Dimension);
                ctx.Writer.Write(Dimension);
                ctx.Writer.Write(Rank);
                for (int i = 0; i < Rank; i++)
                {
                    Contracts.Assert(FloatUtils.IsFinite(Eigenvectors[i]));
                    ctx.Writer.WriteSinglesNoCount(Eigenvectors[i].AsSpan(0, Dimension));
                }
                Contracts.Assert(MeanProjected == null || (MeanProjected.Length == Rank && FloatUtils.IsFinite(MeanProjected)));
                ctx.Writer.WriteSingleArray(MeanProjected);
            }

            public void ProjectMean(float[] mean)
            {
                Contracts.AssertValue(Eigenvectors);
                if (mean == null)
                {
                    MeanProjected = null;
                    return;
                }

                MeanProjected = new float[Rank];
                for (var i = 0; i < Rank; ++i)
                    MeanProjected[i] = VectorUtils.DotProduct(Eigenvectors[i], mean);
            }
        }

        internal const string Summary = "PCA is a dimensionality-reduction transform which computes the projection of a numeric vector onto a low-rank subspace.";
        internal const string UserName = "Principal Component Analysis Transform";
        internal const string ShortName = "Pca";

        internal const string LoaderSignature = "PcaTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCA FUNC",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Got rid of writing float size in model context
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PrincipalComponentAnalysisTransformer).Assembly.FullName);
        }

        private readonly int _numColumns;
        private readonly Mapper.ColumnSchemaInfo[] _schemaInfos;
        private readonly TransformInfo[] _transformInfos;

        private const string RegistrationName = "Pca";

        internal PrincipalComponentAnalysisTransformer(IHostEnvironment env, IDataView input, PrincipalComponentAnalyzer.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(PrincipalComponentAnalysisTransformer)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            _numColumns = columns.Length;
            _transformInfos = new TransformInfo[_numColumns];
            _schemaInfos = new Mapper.ColumnSchemaInfo[_numColumns];

            for (int i = 0; i < _numColumns; i++)
            {
                var colInfo = columns[i];
                var sInfo = _schemaInfos[i] = new Mapper.ColumnSchemaInfo(ColumnPairs[i], input.Schema, colInfo.WeightColumn);
                ValidatePcaInput(Host, colInfo.InputColumnName, sInfo.InputType);
                _transformInfos[i] = new TransformInfo(colInfo.Rank, sInfo.InputType.GetValueCount());
            }

            Train(columns, _transformInfos, input);
        }

        private PrincipalComponentAnalysisTransformer(IHost host, ModelLoadContext ctx)
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
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = options.Columns.Select(item => new PrincipalComponentAnalyzer.ColumnOptions(
                        item.Name,
                        item.Source,
                        item.WeightColumn,
                        item.Rank ?? options.Rank,
                        item.Oversampling ?? options.Oversampling,
                        item.Center ?? options.Center,
                        item.Seed ?? options.Seed)).ToArray();
            return new PrincipalComponentAnalysisTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static PrincipalComponentAnalysisTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(PrincipalComponentAnalysisTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new PrincipalComponentAnalysisTransformer(host, ctx);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // transformInfos
            SaveColumns(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
                _transformInfos[i].Save(ctx);
        }
        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(PrincipalComponentAnalyzer.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private void Train(PrincipalComponentAnalyzer.ColumnOptions[] columns, TransformInfo[] transformInfos, IDataView trainingData)
        {
            var y = new float[_numColumns][][];
            var omega = new float[_numColumns][][];
            var mean = new float[_numColumns][];
            var oversampledRank = new int[_numColumns];
            double totalMemoryUsageEstimate = 0;
            for (int iinfo = 0; iinfo < _numColumns; iinfo++)
            {
                var rnd = columns[iinfo].Seed == null ? Host.Rand : new Random(columns[iinfo].Seed.Value);
                oversampledRank[iinfo] = Math.Min(transformInfos[iinfo].Rank + columns[iinfo].Oversampling, transformInfos[iinfo].Dimension);

                //exact: (size of the 2 big matrices + other minor allocations) / (2^30)
                double colMemoryUsageEstimate = 2.0 * transformInfos[iinfo].Dimension * oversampledRank[iinfo] * sizeof(float) / 1e9;
                totalMemoryUsageEstimate += colMemoryUsageEstimate;
                if (colMemoryUsageEstimate > 2)
                {
                    using (var ch = Host.Start("Memory usage"))
                    {
                        ch.Info("Estimate memory usage for transforming column {1}: {0:G2} GB. If running out of memory, reduce rank and oversampling factor.",
                            colMemoryUsageEstimate, ColumnPairs[iinfo].inputColumnName);
                    }
                }

                y[iinfo] = new float[oversampledRank[iinfo]][];
                omega[iinfo] = new float[oversampledRank[iinfo]][];
                for (int i = 0; i < oversampledRank[iinfo]; i++)
                {
                    y[iinfo][i] = new float[transformInfos[iinfo].Dimension];
                    omega[iinfo][i] = new float[transformInfos[iinfo].Dimension];
                    for (int j = 0; j < transformInfos[iinfo].Dimension; j++)
                    {
                        omega[iinfo][i][j] = (float)Stats.SampleFromGaussian(rnd);
                    }
                }

                if (columns[iinfo].EnsureZeroMean)
                    mean[iinfo] = new float[transformInfos[iinfo].Dimension];
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
                var b2 = new float[oversampledRank[iinfo] * oversampledRank[iinfo]];
                for (var i = 0; i < oversampledRank[iinfo]; ++i)
                {
                    for (var j = i; j < oversampledRank[iinfo]; ++j)
                        b2[i * oversampledRank[iinfo] + j] = b2[j * oversampledRank[iinfo] + i] = VectorUtils.DotProduct(b[iinfo][i], b[iinfo][j]);
                }

                float[] smallEigenvalues; // eigenvectors and eigenvalues of the small matrix B2.
                float[] smallEigenvectors;

                EigenUtils.EigenDecomposition(b2, out smallEigenvalues, out smallEigenvectors);
                transformInfos[iinfo].Eigenvectors = PostProcess(b[iinfo], smallEigenvalues, smallEigenvectors, transformInfos[iinfo].Dimension, oversampledRank[iinfo]);
                transformInfos[iinfo].ProjectMean(mean[iinfo]);
            }
        }

        //Project the covariance matrix A on to Omega: Y <- A * Omega
        //A = X' * X / n, where X = data - mean
        //Note that the covariance matrix is not computed explicitly
        private void Project(IDataView trainingData, float[][] mean, float[][][] omega, float[][][] y, TransformInfo[] transformInfos)
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

            bool[] activeColumns = new bool[trainingData.Schema.Count];
            foreach (var sInfo in _schemaInfos)
            {
                activeColumns[sInfo.InputIndex] = true;
                if (sInfo.WeightColumnIndex >= 0)
                    activeColumns[sInfo.WeightColumnIndex] = true;
            }

            var inputCols = trainingData.Schema.Where(x => activeColumns[x.Index]);
            using (var cursor = trainingData.GetRowCursor(inputCols))
            {
                var weightGetters = new ValueGetter<float>[_numColumns];
                var columnGetters = new ValueGetter<VBuffer<float>>[_numColumns];
                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    var sInfo = _schemaInfos[iinfo];
                    if (sInfo.WeightColumnIndex >= 0)
                        weightGetters[iinfo] = cursor.GetGetter<float>(cursor.Schema[sInfo.WeightColumnIndex]);
                    columnGetters[iinfo] = cursor.GetGetter<VBuffer<float>>(cursor.Schema[sInfo.InputIndex]);
                }

                var features = default(VBuffer<float>);
                while (cursor.MoveNext())
                {
                    for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                    {
                        float weight = 1;
                        weightGetters[iinfo]?.Invoke(ref weight);
                        columnGetters[iinfo](ref features);

                        var featureValues = features.GetValues();
                        if (FloatUtils.IsFinite(weight) && weight >= 0 && (featureValues.Length == 0 || FloatUtils.IsFinite(featureValues)))
                        {
                            totalColWeight[iinfo] += weight;

                            if (center[iinfo])
                                VectorUtils.AddMult(in features, mean[iinfo], weight);

                            for (int i = 0; i < omega[iinfo].Length; i++)
                                VectorUtils.AddMult(in features, y[iinfo][i], weight * VectorUtils.DotProductWithOffset(omega[iinfo][i], 0, in features));
                        }
                    }
                }

                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    if (totalColWeight[iinfo] <= 0)
                        throw Host.Except("Empty data in column '{0}'", ColumnPairs[iinfo].inputColumnName);
                }

                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    var invn = (float)(1 / totalColWeight[iinfo]);

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
        private float[][] PostProcess(float[][] y, float[] sigma, float[] z, int d, int k)
        {
            var pinv = new float[k];
            var tmp = new float[k];

            for (int i = 0; i < k; i++)
                pinv[i] = (float)(1.0) / ((float)(1e-6) + sigma[i]);

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

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            ValidatePcaInput(Host, inputSchema[srcCol].Name, inputSchema[srcCol].Type);
        }

        internal static void ValidatePcaInput(IExceptionContext ectx, string name, DataViewType type)
        {
            string inputSchema; // just used for the excpections

            if (!(type is VectorType vectorType && vectorType.Size > 1 && vectorType.ItemType.Equals(NumberDataViewType.Single)))
                throw ectx.ExceptSchemaMismatch(nameof(inputSchema), "input", name, "known-size vector of float of two or more items", type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            public sealed class ColumnSchemaInfo
            {
                public DataViewType InputType { get; }
                public int InputIndex { get; }
                public int WeightColumnIndex { get; }

                public ColumnSchemaInfo((string outputColumnName, string inputColumnName) columnPair, DataViewSchema schema, string weightColumn = null)
                {
                    schema.TryGetColumnIndex(columnPair.inputColumnName, out int inputIndex);
                    InputIndex = inputIndex;
                    InputType = schema[columnPair.inputColumnName].Type;

                    var weightIndex = -1;
                    if (weightColumn != null)
                    {
                        if (!schema.TryGetColumnIndex(weightColumn, out weightIndex))
                            throw Contracts.Except("Weight column '{0}' does not exist.", weightColumn);
                        Contracts.CheckParam(schema[weightIndex].Type == NumberDataViewType.Single, nameof(weightColumn));
                    }
                    WeightColumnIndex = weightIndex;
                }
            }

            private readonly PrincipalComponentAnalysisTransformer _parent;
            private readonly int _numColumns;

            public Mapper(PrincipalComponentAnalysisTransformer parent, DataViewSchema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _numColumns = parent._numColumns;
                for (int i = 0; i < _numColumns; i++)
                {
                    var colPair = _parent.ColumnPairs[i];
                    var colSchemaInfo = new ColumnSchemaInfo(colPair, inputSchema);
                    ValidatePcaInput(Host, colPair.inputColumnName, colSchemaInfo.InputType);
                    if (colSchemaInfo.InputType.GetVectorSize() != _parent._transformInfos[i].Dimension)
                    {
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.inputColumnName,
                            new VectorType(NumberDataViewType.Single, _parent._transformInfos[i].Dimension).ToString(), colSchemaInfo.InputType.ToString());
                    }
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _parent._transformInfos[i].OutputType, null);
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _numColumns);
                disposer = null;

                var srcGetter = input.GetGetter<VBuffer<float>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(VBuffer<float>);

                ValueGetter<VBuffer<float>> dstGetter = (ref VBuffer<float> dst) =>
                    {
                        srcGetter(ref src);
                        TransformFeatures(Host, in src, ref dst, _parent._transformInfos[iinfo]);
                    };

                return dstGetter;
            }

            private static void TransformFeatures(IExceptionContext ectx, in VBuffer<float> src, ref VBuffer<float> dst, TransformInfo transformInfo)
            {
                ectx.Check(src.Length == transformInfo.Dimension);

                var editor = VBufferEditor.Create(ref dst, transformInfo.Rank);
                for (int i = 0; i < transformInfo.Rank; i++)
                {
                    editor.Values[i] = VectorUtils.DotProductWithOffset(transformInfo.Eigenvectors[i], 0, in src) -
                        (transformInfo.MeanProjected == null ? 0 : transformInfo.MeanProjected[i]);
                }

                dst = editor.Commit();
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.PcaCalculator",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName)]
        internal static CommonOutputs.TransformOutput Calculate(IHostEnvironment env, Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Pca", input);
            var view = PrincipalComponentAnalysisTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
    public sealed class PrincipalComponentAnalyzer : IEstimator<PrincipalComponentAnalysisTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const string WeightColumn = null;
            public const int Rank = 20;
            public const int Oversampling = 20;
            public const bool EnsureZeroMean = true;
            public const int Seed = 0;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
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
            /// The name of the weight column.
            /// </summary>
            public readonly string WeightColumn;
            /// <summary>
            /// The number of components in the PCA.
            /// </summary>
            public readonly int Rank;
            /// <summary>
            /// Oversampling parameter for randomized PCA training.
            /// </summary>
            public readonly int Oversampling;
            /// <summary>
            /// If enabled, data is centered to be zero mean.
            /// </summary>
            public readonly bool EnsureZeroMean;
            /// <summary>
            /// The seed for random number generation.
            /// </summary>
            public readonly int? Seed;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform.
            /// If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="weightColumn">The name of the weight column.</param>
            /// <param name="rank">The number of components in the PCA.</param>
            /// <param name="overSampling">Oversampling parameter for randomized PCA training.</param>
            /// <param name="ensureZeroMean">If enabled, data is centered to be zero mean.</param>
            /// <param name="seed">The random seed. If unspecified random state will be instead derived from the <see cref="MLContext"/>.</param>
            public ColumnOptions(string name,
                              string inputColumnName = null,
                              string weightColumn = Defaults.WeightColumn,
                              int rank = Defaults.Rank,
                              int overSampling = Defaults.Oversampling,
                              bool ensureZeroMean = Defaults.EnsureZeroMean,
                              int? seed = null)
            {
                Name = name;
                InputColumnName = inputColumnName ?? name;
                WeightColumn = weightColumn;
                Rank = rank;
                Oversampling = overSampling;
                EnsureZeroMean = ensureZeroMean;
                Seed = seed;
                Contracts.CheckParam(Oversampling >= 0, nameof(Oversampling), "Oversampling must be non-negative.");
                Contracts.CheckParam(Rank > 0, nameof(Rank), "Rank must be positive.");
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="ensureZeroMean">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        internal PrincipalComponentAnalyzer(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            string weightColumn = Defaults.WeightColumn, int rank = Defaults.Rank,
            int overSampling = Defaults.Oversampling, bool ensureZeroMean = Defaults.EnsureZeroMean,
            int? seed = null)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, weightColumn, rank, overSampling, ensureZeroMean, seed))
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The dataset columns to use, and their specific settings.</param>
        internal PrincipalComponentAnalyzer(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(PrincipalComponentAnalyzer));
            _columns = columns;
        }

        /// <summary>
        /// Trains and returns a <see cref="PrincipalComponentAnalysisTransformer"/>.
        /// </summary>
        public PrincipalComponentAnalysisTransformer Fit(IDataView input) => new PrincipalComponentAnalysisTransformer(_host, input, _columns);

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);

                if (col.Kind != SchemaShape.Column.VectorKind.Vector || !col.ItemType.Equals(NumberDataViewType.Single))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name,
                    SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
