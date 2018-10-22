// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(RffTransform.Summary, typeof(IDataTransform), typeof(RffTransform), typeof(RffTransform.Arguments), typeof(SignatureDataTransform),
    "Random Fourier Features Transform", "RffTransform", "Rff")]

[assembly: LoadableClass(RffTransform.Summary, typeof(IDataTransform), typeof(RffTransform), null, typeof(SignatureLoadDataTransform),
    "Random Fourier Features Transform", RffTransform.LoaderSignature)]

[assembly: LoadableClass(RffTransform.Summary, typeof(RffTransform), null, typeof(SignatureLoadModel),
    "Random Fourier Features Transform", RffTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(RffTransform), null, typeof(SignatureLoadRowMapper),
    "Random Fourier Features Transform", RffTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class RffTransform : OneToOneTransformerBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of random Fourier features to create", ShortName = "dim")]
            public int NewDim = RandomFourierFeaturizingEstimator.Defaults.NewDim;

            [Argument(ArgumentType.Multiple, HelpText = "Which kernel to use?", ShortName = "kernel", SignatureType = typeof(SignatureFourierDistributionSampler))]
            public IComponentFactory<float, IFourierDistributionSampler> MatrixGenerator = new GaussianFourierSampler.Arguments();
            [Argument(ArgumentType.AtMostOnce, HelpText = "Create two features for every random Fourier frequency? (one for cos and one for sin)")]
            public bool UseSin = RandomFourierFeaturizingEstimator.Defaults.UseSin;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The seed of the random number generator for generating the new features (if unspecified, " +
                "the global random is used)")]
            public int? Seed;
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of random Fourier features to create", ShortName = "dim")]
            public int? NewDim;

            [Argument(ArgumentType.Multiple, HelpText = "which kernel to use?", ShortName = "kernel", SignatureType = typeof(SignatureFourierDistributionSampler))]
            public IComponentFactory<float, IFourierDistributionSampler> MatrixGenerator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "create two features for every random Fourier frequency? (one for cos and one for sin)")]
            public bool? UseSin;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The seed of the random number generator for generating the new features (if unspecified, " +
                           "the global random is used)")]
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
                if (NewDim != null || MatrixGenerator != null || UseSin != null || Seed != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        private sealed class TransformInfo
        {
            public readonly int NewDim;
            public readonly int SrcDim;

            // the matrix containing the random fourier vectors
            public readonly AlignedArray RndFourierVectors;

            // the random rotations
            public readonly AlignedArray RotationTerms;

            private readonly IFourierDistributionSampler _matrixGenerator;
            private readonly bool _useSin;
            private readonly TauswortheHybrid _rand;
            private readonly TauswortheHybrid.State _state;

            public TransformInfo(IHost host, ColumnInfo column, int d, float avgDist)
            {
                Contracts.AssertValue(host);

                SrcDim = d;
                NewDim = column.NewDim;
                host.CheckUserArg(NewDim > 0, nameof(column.NewDim));
                _useSin = column.UseSin;
                var seed = column.Seed;
                _rand = seed.HasValue ? RandomUtils.Create(seed) : RandomUtils.Create(host.Rand);
                _state = _rand.GetState();

                var generator = column.Generator;
                _matrixGenerator = generator.CreateComponent(host, avgDist);

                int roundedUpD = RoundUp(NewDim, _cfltAlign);
                int roundedUpNumFeatures = RoundUp(SrcDim, _cfltAlign);
                RndFourierVectors = new AlignedArray(roundedUpD * roundedUpNumFeatures, CpuMathUtils.GetVectorAlignment());
                RotationTerms = _useSin ? null : new AlignedArray(roundedUpD, CpuMathUtils.GetVectorAlignment());

                InitializeFourierCoefficients(roundedUpNumFeatures, roundedUpD);
            }

            public TransformInfo(IHostEnvironment env, ModelLoadContext ctx, string directoryName)
            {
                env.AssertValue(env);

                // *** Binary format ***
                // int: d (number of untransformed features)
                // int: NewDim (number of transformed features)
                // bool: UseSin
                // uint[4]: the seeds for the pseudo random number generator.

                SrcDim = ctx.Reader.ReadInt32();

                NewDim = ctx.Reader.ReadInt32();
                env.CheckDecode(NewDim > 0);

                _useSin = ctx.Reader.ReadBoolByte();

                var length = ctx.Reader.ReadInt32();
                env.CheckDecode(length == 4);
                _state = TauswortheHybrid.State.Load(ctx.Reader);
                _rand = new TauswortheHybrid(_state);

                env.CheckDecode(ctx.Repository != null &&
                    ctx.LoadModelOrNull<IFourierDistributionSampler, SignatureLoadModel>(env, out _matrixGenerator, directoryName));

                // initialize the transform matrix
                int roundedUpD = RoundUp(NewDim, _cfltAlign);
                int roundedUpNumFeatures = RoundUp(SrcDim, _cfltAlign);
                RndFourierVectors = new AlignedArray(roundedUpD * roundedUpNumFeatures, CpuMathUtils.GetVectorAlignment());
                RotationTerms = _useSin ? null : new AlignedArray(roundedUpD, CpuMathUtils.GetVectorAlignment());
                InitializeFourierCoefficients(roundedUpNumFeatures, roundedUpD);
            }

            public void Save(ModelSaveContext ctx, string directoryName)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: d (number of untransformed features)
                // int: NewDim (number of transformed features)
                // bool: UseSin
                // uint[4]: the seeds for the pseudo random number generator.

                ctx.Writer.Write(SrcDim);
                ctx.Writer.Write(NewDim);
                ctx.Writer.WriteBoolByte(_useSin);
                ctx.Writer.Write(4); // fake array length
                _state.Save(ctx.Writer);
                ctx.SaveModel(_matrixGenerator, directoryName);
            }

            private void GetDDimensionalFeatureMapping(int rowSize)
            {
                Contracts.Assert(rowSize >= SrcDim);

                for (int i = 0; i < NewDim; i++)
                {
                    for (int j = 0; j < SrcDim; j++)
                        RndFourierVectors[i * rowSize + j] = _matrixGenerator.Next(_rand);
                }
            }

            private void GetDRotationTerms(int colSize)
            {
                for (int i = 0; i < colSize; ++i)
                    RotationTerms[i] = (_rand.NextSingle() - (float)0.5) * (float)Math.PI;
            }

            private void InitializeFourierCoefficients(int rowSize, int colSize)
            {
                GetDDimensionalFeatureMapping(rowSize);

                if (!_useSin)
                    GetDRotationTerms(NewDim);
            }
        }

        internal const string Summary = "This transform maps numeric vectors to a random low-dimensional feature space. It is useful when data has non-linear features, "
            + "since the transform is designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified "
            + "shift-invariant kernel.";

        public const string LoaderSignature = "RffTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RFF FUNC",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Get rid of writing float size in model context
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RffTransform).Assembly.FullName);
        }

        private readonly TransformInfo[] _transformInfos;

        private static readonly int _cfltAlign = CpuMathUtils.GetVectorAlignment() / sizeof(float);

        private static string TestColumnType(ColumnType type)
        {
            if (type.ItemType == NumberType.Float && type.IsKnownSizeVector)
                return null;
            return "Expected vector of floats with known size";
        }

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly IComponentFactory<float, IFourierDistributionSampler> Generator;
            public readonly int NewDim;
            public readonly bool UseSin;
            public readonly int? Seed;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="generator">Which fourier generator to use.</param>
            /// <param name="newDim">The number of random Fourier features to create.</param>
            /// <param name="useSin">Create two features for every random Fourier frequency? (one for cos and one for sin).</param>
            /// <param name="seed">The seed of the random number generator for generating the new features (if unspecified, the global random is used.</param>
            public ColumnInfo(string input, string output, int newDim, bool useSin, IComponentFactory<float, IFourierDistributionSampler> generator = null, int? seed = null)
            {
                Contracts.CheckUserArg(newDim > 0, nameof(newDim), "must be positive.");
                Input = input;
                Output = output;
                Generator = generator ?? new GaussianFourierSampler.Arguments();
                NewDim = newDim;
                UseSin = useSin;
                Seed = seed;
            }
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            string reason = TestColumnType(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, reason, type.ToString());
            if (_transformInfos[col].SrcDim != type.VectorSize)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input,
                    new VectorType(NumberType.Float, _transformInfos[col].SrcDim).ToString(), type.ToString());
        }

        public RffTransform(IHostEnvironment env, IDataView input, ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RffTransform)), GetColumnPairs(columns))
        {
            var avgDistances = GetAvgDistances(columns, input);
            _transformInfos = new TransformInfo[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                input.Schema.TryGetColumnIndex(columns[i].Input, out int srcCol);
                var typeSrc = input.Schema.GetColumnType(srcCol);
                _transformInfos[i] = new TransformInfo(Host.Register(string.Format("column{0}", i)), columns[i],
                    typeSrc.ValueCount, avgDistances[i]);
            }
        }

        // Round cflt up to a multiple of cfltAlign.
        private static int RoundUp(int cflt, int cfltAlign)
        {
            Contracts.Assert(0 < cflt);
            // cfltAlign should be a power of two.
            Contracts.Assert(0 < cfltAlign && (cfltAlign & (cfltAlign - 1)) == 0);

            // Determine the number of "blobs" of size cfltAlign.
            int cblob = (cflt + cfltAlign - 1) / cfltAlign;
            return cblob * cfltAlign;
        }

        private float[] GetAvgDistances(ColumnInfo[] columns, IDataView input)
        {
            var avgDistances = new float[columns.Length];
            const int reservoirSize = 5000;
            bool[] activeColumns = new bool[input.Schema.ColumnCount];
            int[] srcCols = new int[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input);
                var type = input.Schema.GetColumnType(srcCol);
                string reason = TestColumnType(type);
                if (reason != null)
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input, reason, type.ToString());
                srcCols[i] = srcCol;
                activeColumns[srcCol] = true;
            }
            var reservoirSamplers = new ReservoirSamplerWithReplacement<VBuffer<float>>[columns.Length];
            using (var cursor = input.GetRowCursor(col => activeColumns[col]))
            {
                for (int i = 0; i < columns.Length; i++)
                {
                    var rng = columns[i].Seed.HasValue ? RandomUtils.Create(columns[i].Seed.Value) : Host.Rand;
                    var srcType = input.Schema.GetColumnType(srcCols[i]);
                    if (srcType.IsVector)
                    {
                        var get = cursor.GetGetter<VBuffer<float>>(srcCols[i]);
                        reservoirSamplers[i] = new ReservoirSamplerWithReplacement<VBuffer<float>>(rng, reservoirSize, get);
                    }
                    else
                    {
                        var getOne = cursor.GetGetter<float>(srcCols[i]);
                        float val = 0;
                        ValueGetter<VBuffer<float>> get =
                            (ref VBuffer<float> dst) =>
                            {
                                getOne(ref val);
                                dst = new VBuffer<float>(1, new[] { val });
                            };
                        reservoirSamplers[i] = new ReservoirSamplerWithReplacement<VBuffer<float>>(rng, reservoirSize, get);
                    }
                }

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < columns.Length; i++)
                        reservoirSamplers[i].Sample();
                }
                for (int i = 0; i < columns.Length; i++)
                    reservoirSamplers[i].Lock();

                for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                {
                    var instanceCount = reservoirSamplers[iinfo].NumSampled;

                    // If the number of pairs is at most the maximum reservoir size / 2, we go over all the pairs,
                    // so we get all the examples. Otherwise, get a sample with replacement.
                    VBuffer<float>[] res;
                    int resLength;
                    if (instanceCount < reservoirSize && instanceCount * (instanceCount - 1) <= reservoirSize)
                    {
                        res = reservoirSamplers[iinfo].GetCache();
                        resLength = reservoirSamplers[iinfo].Size;
                        Contracts.Assert(resLength == instanceCount);
                    }
                    else
                    {
                        res = reservoirSamplers[iinfo].GetSample().ToArray();
                        resLength = res.Length;
                    }

                    // If the dataset contains only one valid Instance, then we can't learn anything anyway, so just return 1.
                    if (instanceCount <= 1)
                        avgDistances[iinfo] = 1;
                    else
                    {
                        float[] distances;
                        // create a dummy generator in order to get its type.
                        // REVIEW this should be refactored. See https://github.com/dotnet/machinelearning/issues/699
                        var matrixGenerator = columns[iinfo].Generator.CreateComponent(Host, 1);
                        bool gaussian = matrixGenerator is GaussianFourierSampler;

                        // If the number of pairs is at most the maximum reservoir size / 2, go over all the pairs.
                        if (resLength < reservoirSize)
                        {
                            distances = new float[instanceCount * (instanceCount - 1) / 2];
                            int count = 0;
                            for (int i = 0; i < instanceCount; i++)
                            {
                                for (int j = i + 1; j < instanceCount; j++)
                                {
                                    distances[count++] = gaussian ? VectorUtils.L2DistSquared(ref res[i], ref res[j])
                                        : VectorUtils.L1Distance(ref res[i], ref res[j]);
                                }
                            }
                            Host.Assert(count == distances.Length);
                        }
                        else
                        {
                            distances = new float[reservoirSize / 2];
                            for (int i = 0; i < reservoirSize - 1; i += 2)
                            {
                                // For Gaussian kernels, we scale by the L2 distance squared, since the kernel function is exp(-gamma ||x-y||^2).
                                // For Laplacian kernels, we scale by the L1 distance, since the kernel function is exp(-gamma ||x-y||_1).
                                distances[i / 2] = gaussian ? VectorUtils.L2DistSquared(ref res[i], ref res[i + 1]) :
                                    VectorUtils.L1Distance(ref res[i], ref res[i + 1]);
                            }
                        }

                        // If by chance, in the random permutation all the pairs are the same instance we return 1.
                        float median = MathUtils.GetMedianInPlace(distances, distances.Length);
                        avgDistances[iinfo] = median == 0 ? 1 : median;
                    }
                }
                return avgDistances;
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private RffTransform(IHost host, ModelLoadContext ctx)
         : base(host, ctx)
        {
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // transformInfos
            var columnsLength = ColumnPairs.Length;
            _transformInfos = new TransformInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
            {
                _transformInfos[i] = new TransformInfo(Host, ctx,
                    string.Format("MatrixGenerator{0}", i));
            }
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    cols[i] = new ColumnInfo(item.Source,
                        item.Name,
                        item.NewDim ?? args.NewDim,
                        item.UseSin ?? args.UseSin,
                        item.MatrixGenerator ?? args.MatrixGenerator,
                        item.Seed ?? args.Seed);
                };
            }
            return new RffTransform(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static RffTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(RffTransform));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new RffTransform(host, ctx);
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
                _transformInfos[i].Save(ctx, string.Format("MatrixGenerator{0}", i));
        }

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly ColumnType[] _types;
            private readonly RffTransform _parent;

            public Mapper(RffTransform parent, Schema inputSchema)
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
                    //validate typeSrc.ValueCount and transformInfo.SrcDim
                    _types[i] = new VectorType(NumberType.Float, _parent._transformInfos[i].RotationTerms == null ?
                    _parent._transformInfos[i].NewDim * 2 : _parent._transformInfos[i].NewDim);
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], null);
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                if (_srcTypes[iinfo].IsVector)
                    return GetterFromVectorType(input, iinfo);
                return GetterFromFloatType(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> GetterFromVectorType(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<float>>(_srcCols[iinfo]);
                var src = default(VBuffer<float>);

                var featuresAligned = new AlignedArray(RoundUp(_srcTypes[iinfo].ValueCount, _cfltAlign), CpuMathUtils.GetVectorAlignment());
                var productAligned = new AlignedArray(RoundUp(_parent._transformInfos[iinfo].NewDim, _cfltAlign), CpuMathUtils.GetVectorAlignment());

                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        TransformFeatures(ref src, ref dst, _parent._transformInfos[iinfo], featuresAligned, productAligned);
                    };
            }

            private ValueGetter<VBuffer<float>> GetterFromFloatType(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<float>(_srcCols[iinfo]);
                var src = default(float);

                var featuresAligned = new AlignedArray(RoundUp(1, _cfltAlign), CpuMathUtils.GetVectorAlignment());
                var productAligned = new AlignedArray(RoundUp(_parent._transformInfos[iinfo].NewDim, _cfltAlign), CpuMathUtils.GetVectorAlignment());

                var oneDimensionalVector = new VBuffer<float>(1, new float[] { 0 });

                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        oneDimensionalVector.Values[0] = src;
                        TransformFeatures(ref oneDimensionalVector, ref dst, _parent._transformInfos[iinfo], featuresAligned, productAligned);
                    };
            }

            private void TransformFeatures(ref VBuffer<float> src, ref VBuffer<float> dst, TransformInfo transformInfo,
                AlignedArray featuresAligned, AlignedArray productAligned)
            {
                Host.Check(src.Length == transformInfo.SrcDim, "column does not have the expected dimensionality.");

                var values = dst.Values;
                float scale;
                if (transformInfo.RotationTerms != null)
                {
                    if (Utils.Size(values) < transformInfo.NewDim)
                        values = new float[transformInfo.NewDim];
                    scale = MathUtils.Sqrt(2.0f / transformInfo.NewDim);
                }
                else
                {
                    if (Utils.Size(values) < 2 * transformInfo.NewDim)
                        values = new float[2 * transformInfo.NewDim];
                    scale = MathUtils.Sqrt(1.0f / transformInfo.NewDim);
                }

                if (src.IsDense)
                {
                    featuresAligned.CopyFrom(src.Values, 0, src.Length);
                    CpuMathUtils.MatTimesSrc(false, transformInfo.RndFourierVectors, featuresAligned, productAligned,
                        transformInfo.NewDim);
                }
                else
                {
                    // This overload of MatTimesSrc ignores the values in slots that are not in src.Indices, so there is
                    // no need to zero them out.
                    featuresAligned.CopyFrom(src.Indices, src.Values, 0, 0, src.Count, zeroItems: false);
                    CpuMathUtils.MatTimesSrc(transformInfo.RndFourierVectors, src.Indices, featuresAligned, 0, 0,
                        src.Count, productAligned, transformInfo.NewDim);
                }

                for (int i = 0; i < transformInfo.NewDim; i++)
                {
                    var dotProduct = productAligned[i];
                    if (transformInfo.RotationTerms != null)
                        values[i] = (float)MathUtils.Cos(dotProduct + transformInfo.RotationTerms[i]) * scale;
                    else
                    {
                        values[2 * i] = (float)MathUtils.Cos(dotProduct) * scale;
                        values[2 * i + 1] = (float)MathUtils.Sin(dotProduct) * scale;
                    }
                }

                dst = new VBuffer<float>(transformInfo.RotationTerms == null ? 2 * transformInfo.NewDim : transformInfo.NewDim,
                    values, dst.Indices);
            }
        }
    }

    /// <summary>
    /// Estimator which takes set of vector columns and maps its input to a random low-dimensional feature space.
    /// </summary>
    public sealed class RandomFourierFeaturizingEstimator : IEstimator<RffTransform>
    {
        internal static class Defaults
        {
            public const int NewDim = 1000;
            public const bool UseSin = false;
        }

        private readonly IHost _host;
        private readonly RffTransform.ColumnInfo[] _columns;

        /// <summary>
        /// Convinence constructor for simple one column case
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="newDim">The number of random Fourier features to create.</param>
        /// <param name="useSin">Create two features for every random Fourier frequency? (one for cos and one for sin).</param>
        public RandomFourierFeaturizingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, int newDim = Defaults.NewDim, bool useSin = Defaults.UseSin)
            : this(env, new RffTransform.ColumnInfo(inputColumn, outputColumn ?? inputColumn, newDim, useSin))
        {
        }

        public RandomFourierFeaturizingEstimator(IHostEnvironment env, params RffTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(RandomFourierFeaturizingEstimator));
            _columns = columns;
        }

        public RffTransform Fit(IDataView input) => new RffTransform(_host, input, _columns);

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

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
            }

            return new SchemaShape(result.Values);
        }
    }

    public static class RffExtenensions
    {
        private struct Config
        {
            public readonly int NewDim;
            public readonly bool UseSin;
            public readonly int? Seed;
            public readonly IComponentFactory<float, IFourierDistributionSampler> Generator;

            public Config(int newDim, bool useSin, IComponentFactory<float, IFourierDistributionSampler> generator, int? seed = null)
            {
                NewDim = newDim;
                UseSin = useSin;
                Generator = generator;
                Seed = seed;
            }
        }
        private interface IColInput
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplVector<T> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Reconciler.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static readonly Reconciler Inst = new Reconciler();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new RffTransform.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (IColInput)toOutput[i];
                    infos[i] = new RffTransform.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], tcol.Config.NewDim, tcol.Config.UseSin, tcol.Config.Generator, tcol.Config.Seed);
                }
                return new RandomFourierFeaturizingEstimator(env, infos);
            }
        }

        /// <summary>
        /// It maps input to a random low-dimensional feature space. It is useful when data has non-linear features, since the transform
        /// is designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user
        /// speciﬁed shift-invariant kernel. With this transform, we are able to use linear methods (which are scalable) to approximate more complex kernel SVM models.
        /// </summary>
        /// <param name="input">The column to apply Random Fourier transfomration.</param>
        /// <param name="newDim">Expected size of new vector.</param>
        /// <param name="useSin">Create two features for every random Fourier frequency? (one for cos and one for sin) </param>
        /// <param name="generator">Which kernel to use. (<see cref="GaussianFourierSampler"/> by default)</param>
        /// <param name="seed">The seed of the random number generator for generating the new features. If not specified global random would be used.</param>
        public static Vector<float> LowerVectorSizeWithRandomFourierTransformation(this Vector<float> input,
            int newDim = RandomFourierFeaturizingEstimator.Defaults.NewDim, bool useSin = RandomFourierFeaturizingEstimator.Defaults.UseSin,
            IComponentFactory<float, IFourierDistributionSampler> generator = null, int? seed = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(newDim, useSin, generator, seed));
        }
    }
}
