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
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ApproximatedKernelTransformer.Summary, typeof(IDataTransform), typeof(ApproximatedKernelTransformer), typeof(ApproximatedKernelTransformer.Options), typeof(SignatureDataTransform),
    "Random Fourier Features Transform", "RffTransform", "Rff")]

[assembly: LoadableClass(ApproximatedKernelTransformer.Summary, typeof(IDataTransform), typeof(ApproximatedKernelTransformer), null, typeof(SignatureLoadDataTransform),
    "Random Fourier Features Transform", ApproximatedKernelTransformer.LoaderSignature)]

[assembly: LoadableClass(ApproximatedKernelTransformer.Summary, typeof(ApproximatedKernelTransformer), null, typeof(SignatureLoadModel),
    "Random Fourier Features Transform", ApproximatedKernelTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ApproximatedKernelTransformer), null, typeof(SignatureLoadRowMapper),
    "Random Fourier Features Transform", ApproximatedKernelTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Maps vector columns to a feature space where the inner products approximate a user specified shift-invariant kernel.
    /// The kernel is indicated by specifying a <see cref="KernelBase"/> instance. The available implementations
    /// are <see cref="GaussianKernel"/> and <see cref="LaplacianKernel"/>.
    /// This transformation is based on this paper by
    /// <a href="http://pages.cs.wisc.edu/~brecht/papers/07.rah.rec.nips.pdf">Rahimi and Recht</a>.
    /// </summary>
    public sealed class ApproximatedKernelTransformer : OneToOneTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of random Fourier features to create", ShortName = "dim")]
            public int NewDim = ApproximatedKernelMappingEstimator.Defaults.Rank;

            [Argument(ArgumentType.Multiple, HelpText = "Which kernel to use?", ShortName = "kernel", SignatureType = typeof(SignatureKernelBase))]
            public IComponentFactory<KernelBase> MatrixGenerator = new GaussianKernel.Options();

            [Argument(ArgumentType.AtMostOnce, HelpText = "Create two features for every random Fourier frequency? (one for cos and one for sin)")]
            public bool UseSin = ApproximatedKernelMappingEstimator.Defaults.UseCosAndSinBases;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The seed of the random number generator for generating the new features (if unspecified, " +
                "the global random is used)")]
            public int? Seed;
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of random Fourier features to create", ShortName = "dim")]
            public int? NewDim;

            [Argument(ArgumentType.Multiple, HelpText = "which kernel to use?", ShortName = "kernel", SignatureType = typeof(SignatureKernelBase))]
            public IComponentFactory<KernelBase> MatrixGenerator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "create two features for every random Fourier frequency? (one for cos and one for sin)")]
            public bool? UseSin;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The seed of the random number generator for generating the new features (if unspecified, " +
                           "the global random is used)")]
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

            private readonly FourierRandomNumberGeneratorBase _matrixGenerator;
            private readonly bool _useSin;
            private readonly TauswortheHybrid _rand;
            private readonly TauswortheHybrid.State _state;

            public TransformInfo(IHost host, ApproximatedKernelMappingEstimator.ColumnOptions column, int d, float avgDist)
            {
                Contracts.AssertValue(host);

                SrcDim = d;
                NewDim = column.Rank;
                host.CheckUserArg(NewDim > 0, nameof(column.Rank));
                _useSin = column.UseCosAndSinBases;
                var seed = column.Seed;
                _rand = seed.HasValue ? RandomUtils.Create(seed) : RandomUtils.Create(host.Rand);
                _state = _rand.GetState();

                var generator = column.Generator;
                _matrixGenerator = generator.GetRandomNumberGenerator(avgDist);

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
                    ctx.LoadModelOrNull<FourierRandomNumberGeneratorBase, SignatureLoadModel>(env, out _matrixGenerator, directoryName));

                // initialize the transform matrix
                int roundedUpD = RoundUp(NewDim, _cfltAlign);
                int roundedUpNumFeatures = RoundUp(SrcDim, _cfltAlign);
                RndFourierVectors = new AlignedArray(roundedUpD * roundedUpNumFeatures, CpuMathUtils.GetVectorAlignment());
                RotationTerms = _useSin ? null : new AlignedArray(roundedUpD, CpuMathUtils.GetVectorAlignment());
                InitializeFourierCoefficients(roundedUpNumFeatures, roundedUpD);
            }

            internal void Save(ModelSaveContext ctx, string directoryName)
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

        internal const string LoaderSignature = "RffTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RFF FUNC",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Get rid of writing float size in model context
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ApproximatedKernelTransformer).Assembly.FullName);
        }

        private readonly TransformInfo[] _transformInfos;

        private static readonly int _cfltAlign = CpuMathUtils.GetVectorAlignment() / sizeof(float);

        private static string TestColumnType(DataViewType type)
        {
            if (type is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType == NumberDataViewType.Single)
                return null;
            return "Expected vector of floats with known size";
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ApproximatedKernelMappingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            string reason = TestColumnType(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, reason, type.ToString());
            if (_transformInfos[col].SrcDim != type.GetVectorSize())
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName,
                    new VectorType(NumberDataViewType.Single, _transformInfos[col].SrcDim).ToString(), type.ToString());
        }

        internal ApproximatedKernelTransformer(IHostEnvironment env, IDataView input, ApproximatedKernelMappingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ApproximatedKernelTransformer)), GetColumnPairs(columns))
        {
            var avgDistances = GetAvgDistances(columns, input);
            _transformInfos = new TransformInfo[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                input.Schema.TryGetColumnIndex(columns[i].InputColumnName, out int srcCol);
                var typeSrc = input.Schema[srcCol].Type;
                _transformInfos[i] = new TransformInfo(Host.Register(string.Format("column{0}", i)), columns[i],
                    typeSrc.GetValueCount(), avgDistances[i]);
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

        private float[] GetAvgDistances(ApproximatedKernelMappingEstimator.ColumnOptions[] columns, IDataView input)
        {
            var avgDistances = new float[columns.Length];
            const int reservoirSize = 5000;
            var activeColumns = new List<DataViewSchema.Column>();
            int[] srcCols = new int[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].inputColumnName, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].inputColumnName);
                var type = input.Schema[srcCol].Type;
                string reason = TestColumnType(type);
                if (reason != null)
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].inputColumnName, reason, type.ToString());
                srcCols[i] = srcCol;
                activeColumns.Add(input.Schema[srcCol]);
            }
            var reservoirSamplers = new ReservoirSamplerWithReplacement<VBuffer<float>>[columns.Length];
            using (var cursor = input.GetRowCursor(activeColumns))
            {
                for (int i = 0; i < columns.Length; i++)
                {
                    var rng = columns[i].Seed.HasValue ? RandomUtils.Create(columns[i].Seed.Value) : Host.Rand;
                    var srcType = input.Schema[srcCols[i]].Type;
                    if (srcType is VectorType)
                    {
                        var get = cursor.GetGetter<VBuffer<float>>(cursor.Schema[srcCols[i]]);
                        reservoirSamplers[i] = new ReservoirSamplerWithReplacement<VBuffer<float>>(rng, reservoirSize, get);
                    }
                    else
                    {
                        var getOne = cursor.GetGetter<float>(cursor.Schema[srcCols[i]]);
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
                        var generator = columns[iinfo].Generator;
                        // If the number of pairs is at most the maximum reservoir size / 2, go over all the pairs.
                        if (resLength < reservoirSize)
                        {
                            distances = new float[instanceCount * (instanceCount - 1) / 2];
                            int count = 0;
                            for (int i = 0; i < instanceCount; i++)
                            {
                                for (int j = i + 1; j < instanceCount; j++)
                                    distances[count++] = generator.Distance(in res[i], in res[j]);
                            }
                            Host.Assert(count == distances.Length);
                        }
                        else
                        {
                            distances = new float[reservoirSize / 2];
                            for (int i = 0; i < reservoirSize - 1; i += 2)
                                distances[i / 2] = generator.Distance(in res[i], in res[i + 1]);
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private ApproximatedKernelTransformer(IHost host, ModelLoadContext ctx)
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
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(options.MatrixGenerator, nameof(options.MatrixGenerator));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new ApproximatedKernelMappingEstimator.ColumnOptions[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = new ApproximatedKernelMappingEstimator.ColumnOptions(
                        item.Name,
                        item.NewDim ?? options.NewDim,
                        item.UseSin ?? options.UseSin,
                        item.Source ?? item.Name,
                        (item.MatrixGenerator ?? options.MatrixGenerator).CreateComponent(env),
                        item.Seed ?? options.Seed);
                };
            }
            return new ApproximatedKernelTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ApproximatedKernelTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(ApproximatedKernelTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new ApproximatedKernelTransformer(host, ctx);
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
                _transformInfos[i].Save(ctx, string.Format("MatrixGenerator{0}", i));
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly DataViewType[] _types;
            private readonly ApproximatedKernelTransformer _parent;

            public Mapper(ApproximatedKernelTransformer parent, DataViewSchema inputSchema)
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
                    //validate typeSrc.ValueCount and transformInfo.SrcDim
                    _types[i] = new VectorType(NumberDataViewType.Single, _parent._transformInfos[i].RotationTerms == null ?
                    _parent._transformInfos[i].NewDim * 2 : _parent._transformInfos[i].NewDim);
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], null);
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                if (_srcTypes[iinfo] is VectorType)
                    return GetterFromVectorType(input, iinfo);
                return GetterFromFloatType(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> GetterFromVectorType(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<float>>(input.Schema[_srcCols[iinfo]]);
                var src = default(VBuffer<float>);

                var featuresAligned = new AlignedArray(RoundUp(_srcTypes[iinfo].GetValueCount(), _cfltAlign), CpuMathUtils.GetVectorAlignment());
                var productAligned = new AlignedArray(RoundUp(_parent._transformInfos[iinfo].NewDim, _cfltAlign), CpuMathUtils.GetVectorAlignment());

                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        TransformFeatures(in src, ref dst, _parent._transformInfos[iinfo], featuresAligned, productAligned);
                    };

            }

            private ValueGetter<VBuffer<float>> GetterFromFloatType(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<float>(input.Schema[_srcCols[iinfo]]);
                var src = default(float);

                var featuresAligned = new AlignedArray(RoundUp(1, _cfltAlign), CpuMathUtils.GetVectorAlignment());
                var productAligned = new AlignedArray(RoundUp(_parent._transformInfos[iinfo].NewDim, _cfltAlign), CpuMathUtils.GetVectorAlignment());

                var oneDimensionalVector = new VBuffer<float>(1, new float[] { 0 });

                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        VBufferEditor.CreateFromBuffer(ref oneDimensionalVector).Values[0] = src;
                        TransformFeatures(in oneDimensionalVector, ref dst, _parent._transformInfos[iinfo], featuresAligned, productAligned);
                    };
            }

            private void TransformFeatures(in VBuffer<float> src, ref VBuffer<float> dst, TransformInfo transformInfo,
                AlignedArray featuresAligned, AlignedArray productAligned)
            {
                Host.Check(src.Length == transformInfo.SrcDim, "column does not have the expected dimensionality.");

                float scale;
                int newDstLength;
                if (transformInfo.RotationTerms != null)
                {
                    newDstLength = transformInfo.NewDim;
                    scale = MathUtils.Sqrt(2.0f / transformInfo.NewDim);
                }
                else
                {
                    newDstLength = 2 * transformInfo.NewDim;
                    scale = MathUtils.Sqrt(1.0f / transformInfo.NewDim);
                }

                if (src.IsDense)
                {
                    featuresAligned.CopyFrom(src.GetValues());
                    CpuMathUtils.MatrixTimesSource(false, transformInfo.RndFourierVectors, featuresAligned, productAligned,
                        transformInfo.NewDim);
                }
                else
                {
                    // This overload of MatTimesSrc ignores the values in slots that are not in src.Indices, so there is
                    // no need to zero them out.
                    var srcValues = src.GetValues();
                    var srcIndices = src.GetIndices();
                    featuresAligned.CopyFrom(srcIndices, srcValues, 0, 0, srcValues.Length, zeroItems: false);
                    CpuMathUtils.MatrixTimesSource(transformInfo.RndFourierVectors, srcIndices, featuresAligned, 0, 0,
                        srcValues.Length, productAligned, transformInfo.NewDim);
                }

                var dstEditor = VBufferEditor.Create(ref dst, newDstLength);
                for (int i = 0; i < transformInfo.NewDim; i++)
                {
                    var dotProduct = productAligned[i];
                    if (transformInfo.RotationTerms != null)
                        dstEditor.Values[i] = (float)MathUtils.Cos(dotProduct + transformInfo.RotationTerms[i]) * scale;
                    else
                    {
                        dstEditor.Values[2 * i] = (float)MathUtils.Cos(dotProduct) * scale;
                        dstEditor.Values[2 * i + 1] = (float)MathUtils.Sin(dotProduct) * scale;
                    }
                }

                dst = dstEditor.Commit();
            }
        }
    }

    /// <summary>
    /// Maps vector columns to a low -dimensional feature space.
    /// </summary>
    public sealed class ApproximatedKernelMappingEstimator : IEstimator<ApproximatedKernelTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const int Rank = 1000;
            public const bool UseCosAndSinBases = false;
        }

        /// <summary>
        /// Describes how the transformer handles one Gcn column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of the column to transform.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// Which fourier generator to use.
            /// </summary>
            public readonly KernelBase Generator;
            /// <summary>
            /// The number of random Fourier features to create.
            /// </summary>
            public readonly int Rank;
            /// <summary>
            /// If <see langword="true"/>, use both of cos and sin basis functions to create two features for every random Fourier frequency.
            /// Otherwise, only cos bases would be used.
            /// </summary>
            public readonly bool UseCosAndSinBases;
            /// <summary>
            /// The seed of the random number generator for generating the new features (if unspecified, the global random is used).
            /// </summary>
            public readonly int? Seed;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="rank">The number of random Fourier features to create.</param>
            /// <param name="useCosAndSinBases">If <see langword="true"/>, use both of cos and sin basis functions to create two features for
            /// every random Fourier frequency. Otherwise, only cos bases would be used.</param>
            /// <param name="inputColumnName">Name of column to transform. </param>
            /// <param name="generator">Which fourier generator to use.</param>
            /// <param name="seed">The seed of the random number generator for generating the new features (if unspecified, the global random is used).</param>
            public ColumnOptions(string name, int rank, bool useCosAndSinBases, string inputColumnName = null, KernelBase generator = null, int? seed = null)
            {
                Contracts.CheckUserArg(rank > 0, nameof(rank), "must be positive.");
                InputColumnName = inputColumnName ?? name;
                Name = name;
                Generator = generator ?? new GaussianKernel();
                Rank = rank;
                UseCosAndSinBases = useCosAndSinBases;
                Seed = seed;
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        /// <summary>
        /// Convinence constructor for simple one column case.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="rank">The number of random Fourier features to create.</param>
        /// <param name="useCosAndSinBases">Create two features for every random Fourier frequency? (one for cos and one for sin).</param>
        internal ApproximatedKernelMappingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, int rank = Defaults.Rank, bool useCosAndSinBases = Defaults.UseCosAndSinBases)
            : this(env, new ColumnOptions(outputColumnName, rank, useCosAndSinBases, inputColumnName ?? outputColumnName))
        {
        }

        internal ApproximatedKernelMappingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ApproximatedKernelMappingEstimator));
            _columns = columns;
        }

        /// <summary>
        /// Trains and returns a <see cref="ApproximatedKernelTransformer"/>.
        /// </summary>
        public ApproximatedKernelTransformer Fit(IDataView input) => new ApproximatedKernelTransformer(_host, input, _columns);

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
                if (col.ItemType.RawType != typeof(float) || col.Kind != SchemaShape.Column.VectorKind.Vector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
