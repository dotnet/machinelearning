// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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

[assembly: LoadableClass(PcaTransform.Summary, typeof(IDataTransform), typeof(PcaTransform), typeof(PcaTransform.Arguments), typeof(SignatureDataTransform),
    PcaTransform.UserName, PcaTransform.LoaderSignature, PcaTransform.ShortName)]

[assembly: LoadableClass(PcaTransform.Summary, typeof(IDataTransform), typeof(PcaTransform), null, typeof(SignatureLoadDataTransform),
    PcaTransform.UserName, PcaTransform.LoaderSignature)]

[assembly: LoadableClass(PcaTransform.Summary, typeof(PcaTransform), null, typeof(SignatureLoadModel),
    PcaTransform.UserName, PcaTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(PcaTransform), null, typeof(SignatureLoadRowMapper),
    PcaTransform.UserName, PcaTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(PcaTransform), null, typeof(SignatureEntryPointModule), PcaTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
    public sealed class PcaTransform : OneToOneTransformerBase
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
                              string weightColumn = Defaults.WeightColumn,
                              int rank = Defaults.Rank,
                              int overSampling = Defaults.Oversampling,
                              bool center = Defaults.Center,
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

            // The following functions and properties are all internal and used for simplifying the
            // Transformer and Mapper code.

            internal ColumnInfo((string input, string output) columnPair)
            {
                Input = columnPair.input;
                Output = columnPair.output;
            }

            internal void SetSchema(Schema schema)
            {
                _schema = schema;
            }

            private Schema _schema;

            internal int InputIndex
            {
                get
                {
                    Contracts.AssertValue(_schema);
                    // Column names are already checked by PcaTransform
                    _schema.TryGetColumnIndex(Input, out int index);
                    return index;
                }
            }

            internal ColumnType InputType
            {
                get
                {
                    Contracts.AssertValue(_schema);
                    return _schema[Input].Type;
                }
            }
        }

        private sealed class TransformInfo
        {
            public readonly int Dimension;
            public readonly int Rank;

            public float[][] Eigenvectors;
            public float[] MeanProjected;

            internal ColumnType OutputType => new VectorType(NumberType.Float, Rank);

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
                //   float[]: the i'th eigenvector
                // int: the size of MeanProjected (0 if it is null)
                // float[]: MeanProjected

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

            internal void ProjectMean(float[] mean)
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

        public const string LoaderSignature = "PcaTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCA FUNC",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Get rid of writing float size in model context
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PcaTransform).Assembly.FullName);
        }

        private readonly int _numColumns;
        private readonly ColumnInfo[] _columns;
        private readonly TransformInfo[] _transformInfos;
        private readonly int[] _weightColumnIndices;

        private const string RegistrationName = "Pca";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public PcaTransform(IHostEnvironment env, IDataView input, ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(PcaTransform)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);

            _numColumns = ColumnPairs.Length;
            _columns = columns;
            _transformInfos = new TransformInfo[_numColumns];
            _weightColumnIndices = new int[_numColumns];

            for (int i = 0; i < _numColumns; i++)
            {
                var col = columns[i];
                col.SetSchema(input.Schema);

                ValidatePcaInput(Host, col.Input, col.InputType);
                Host.CheckUserArg(col.Oversampling >= 0, nameof(col.Oversampling), "Oversampling must be non-negative");

                _transformInfos[i] = new TransformInfo(col.Rank, col.InputType.ValueCount);
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
        }

        private PcaTransform(IHost host, ModelLoadContext ctx)
         : base(host, ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // transformInfos
            Host.AssertNonEmpty(ColumnPairs);
            _numColumns = ColumnPairs.Length;
            _columns = new ColumnInfo[_numColumns];
            _transformInfos = new TransformInfo[_numColumns];
            for (int i = 0; i < _numColumns; i++)
            {
                _columns[i] = new ColumnInfo(ColumnPairs[i]);
                _transformInfos[i] = new TransformInfo(ctx);
            }
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
            return new PcaTransform(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static PcaTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(PcaTransform));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new PcaTransform(host, ctx);
        }

        public override void Save(ModelSaveContext ctx)
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
        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private void Train(ColumnInfo[] columns, TransformInfo[] transformInfos, IDataView trainingData)
        {
            var y = new float[_numColumns][][];
            var omega = new float[_numColumns][][];
            var mean = new float[_numColumns][];
            var oversampledRank = new int[_numColumns];
            var rnd = Host.Rand;
            Double totalMemoryUsageEstimate = 0;
            for (int iinfo = 0; iinfo < _numColumns; iinfo++)
            {
                oversampledRank[iinfo] = Math.Min(transformInfos[iinfo].Rank + columns[iinfo].Oversampling, transformInfos[iinfo].Dimension);

                //exact: (size of the 2 big matrices + other minor allocations) / (2^30)
                Double colMemoryUsageEstimate = 2.0 * transformInfos[iinfo].Dimension * oversampledRank[iinfo] * sizeof(float) / 1e9;
                totalMemoryUsageEstimate += colMemoryUsageEstimate;
                if (colMemoryUsageEstimate > 2)
                {
                    using (var ch = Host.Start("Memory usage"))
                    {
                        ch.Info("Estimate memory usage for transforming column {1}: {0:G2} GB. If running out of memory, reduce rank and oversampling factor.",
                            colMemoryUsageEstimate, ColumnPairs[iinfo].input);
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

                if (columns[iinfo].Center)
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

            bool[] activeColumns = new bool[trainingData.Schema.ColumnCount];
            for (int iinfo = 0; iinfo < _numColumns; iinfo++)
            {
                activeColumns[_columns[iinfo].InputIndex] = true;
                if (_weightColumnIndices[iinfo] >= 0)
                    activeColumns[_weightColumnIndices[iinfo]] = true;
            }

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var weightGetters = new ValueGetter<float>[_numColumns];
                var columnGetters = new ValueGetter<VBuffer<float>>[_numColumns];
                for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                {
                    if (_weightColumnIndices[iinfo] >= 0)
                        weightGetters[iinfo] = cursor.GetGetter<float>(_weightColumnIndices[iinfo]);
                    columnGetters[iinfo] = cursor.GetGetter<VBuffer<float>>(_columns[iinfo].InputIndex);
                }

                var features = default(VBuffer<float>);
                while (cursor.MoveNext())
                {
                    for (int iinfo = 0; iinfo < _numColumns; iinfo++)
                    {
                        float weight = 1;
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

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            ValidatePcaInput(Host, inputSchema.GetColumnName(srcCol), inputSchema.GetColumnType(srcCol));
        }

        internal static void ValidatePcaInput(IHost host, string name, ColumnType type)
        {
            if (!type.IsVector)
                throw host.Except($"Pca transform can only be applied to vector columns. Column ${name} is of type ${type}");

            if (!(type.IsKnownSizeVector && type.VectorSize > 1))
                throw host.Except($"Pca transform can only be applied to vector columns. Column ${name} is of size ${type.VectorSize}");

            var itemType = type.ItemType;
            if (!itemType.IsNumber)
                throw host.Except($"Pca transform can only be applied to vector of numeric items. Column ${name} contains type ${itemType}");
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnInfo[] _columns;
            private readonly PcaTransform _parent;
            private readonly int _numColumns;

            public Mapper(PcaTransform parent, Schema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _numColumns = parent._numColumns;
                _columns = new ColumnInfo[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                {
                    var col = _columns[i] = new ColumnInfo(_parent.ColumnPairs[i]);
                    col.SetSchema(inputSchema);
                    ValidatePcaInput(Host, col.Input, col.InputType);
                    if (col.InputType.VectorSize != _parent._transformInfos[i].Dimension)
                    {
                        var msg = $"Dimension of column ${col.Input} is ${col.InputType.VectorSize}, which doesn't match the expected size ${_parent._transformInfos[i].Dimension}";
                        throw Host.Except(msg);
                    }
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                    result[i] = new Schema.Column(_columns[i].Output, _parent._transformInfos[i].OutputType, null);
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _numColumns);
                disposer = null;

                var srcGetter = input.GetGetter<VBuffer<float>>(ColMapNewToOld[iinfo]);
                var src = default(VBuffer<float>);

                ValueGetter<VBuffer<float>> dstGetter = (ref VBuffer<float> dst) =>
                    {
                        srcGetter(ref src);
                        TransformFeatures(Host, ref src, ref dst, _parent._transformInfos[iinfo]);
                    };

                return dstGetter;
            }

            private static void TransformFeatures(IExceptionContext ectx, ref VBuffer<float> src, ref VBuffer<float> dst, TransformInfo transformInfo)
            {
                ectx.Check(src.Length == transformInfo.Dimension);

                var values = dst.Values;
                if (Utils.Size(values) < transformInfo.Rank)
                    values = new float[transformInfo.Rank];

                for (int i = 0; i < transformInfo.Rank; i++)
                {
                    values[i] = VectorUtils.DotProductWithOffset(transformInfo.Eigenvectors[i], 0, ref src) -
                        (transformInfo.MeanProjected == null ? 0 : transformInfo.MeanProjected[i]);
                }

                dst = new VBuffer<float>(transformInfo.Rank, values, dst.Indices);
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.PcaCalculator",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/member[@name=""PCA""]/*' />",
                                 @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/example[@name=""PcaCalculator""]/*' />"})]
        public static CommonOutputs.TransformOutput Calculate(IHostEnvironment env, Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Pca", input);
            var view = PcaTransform.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    public sealed class PcaEstimator : IEstimator<PcaTransform>
    {
        private readonly IHost _host;
        private readonly PcaTransform.ColumnInfo[] _columns;

        /// <summary>Convinence constructor for simple one column case.</summary>
        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Input column to apply PCA on.</param>
        /// <param name="outputColumn">Output column. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation</param>
        public PcaEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null,
            string weightColumn = PcaTransform.Defaults.WeightColumn, int rank = PcaTransform.Defaults.Rank,
            int overSampling = PcaTransform.Defaults.Oversampling, bool center = PcaTransform.Defaults.Center,
            int? seed = null)
            : this(env, new PcaTransform.ColumnInfo(inputColumn, outputColumn ?? inputColumn, weightColumn, rank, overSampling, center, seed))
        {
        }

        public PcaEstimator(IHostEnvironment env, params PcaTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(PcaEstimator));
            _columns = columns;
        }

        public PcaTransform Fit(IDataView input) => new PcaTransform(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector && col.ItemType.IsNumber))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output,
                    SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
            }

            return new SchemaShape(result.Values);
        }
    }

    public static class PcaEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, string weightColumn, int rank,
                                     int overSampling, bool center, int? seed = null)
                : base(new Reconciler(weightColumn, rank, overSampling, center, seed), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly PcaTransform.ColumnInfo _colInfo;

            public Reconciler(string weightColumn, int rank, int overSampling, bool center, int? seed = null)
            {
                _colInfo = new PcaTransform.ColumnInfo(
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
                return new PcaEstimator(env, inputColName, outputColName,
                                         _colInfo.WeightColumn, _colInfo.Rank, _colInfo.Oversampling,
                                         _colInfo.Center, _colInfo.Seed);
            }
        }

        /// <summary>Compute the principal components of the input column. Can significantly reduce size of vector.</summary>
        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        /// <param name="input">The column to apply PCA to.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation</param>
        /// <returns>Vector containing the principal components.</returns>
        public static Vector<float> ToPrincipalComponents(this Vector<float> input,
            string weightColumn = PcaTransform.Defaults.WeightColumn,
            int rank = PcaTransform.Defaults.Rank,
            int overSampling = PcaTransform.Defaults.Oversampling,
            bool center = PcaTransform.Defaults.Center,
            int? seed = null) => new OutPipelineColumn(input, weightColumn, rank, overSampling, center, seed);
    }
}
