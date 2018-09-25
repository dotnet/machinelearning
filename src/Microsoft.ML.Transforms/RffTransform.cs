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
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(RffTransform.Summary, typeof(RffTransform), typeof(RffTransform.Arguments), typeof(SignatureDataTransform),
    "Random Fourier Features Transform", "RffTransform", "Rff")]

[assembly: LoadableClass(RffTransform.Summary, typeof(RffTransform), null, typeof(SignatureLoadDataTransform),
    "Random Fourier Features Transform", RffTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class RffTransform : OneToOneTransformBase
    {
        private static class Defaults
        {
            public const int NewDim = 1000;
            public const bool UseSin = false;
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of random Fourier features to create", ShortName = "dim")]
            public int NewDim = Defaults.NewDim;

            [Argument(ArgumentType.Multiple, HelpText = "Which kernel to use?", ShortName = "kernel", SignatureType = typeof(SignatureFourierDistributionSampler))]
            public IComponentFactory<Float, IFourierDistributionSampler> MatrixGenerator =
                ComponentFactoryUtils.CreateFromFunction<Float, IFourierDistributionSampler>(
                    (env, avgDist) => new GaussianFourierSampler(env, new GaussianFourierSampler.Arguments(), avgDist));

            [Argument(ArgumentType.AtMostOnce, HelpText = "Create two features for every random Fourier frequency? (one for cos and one for sin)")]
            public bool UseSin = Defaults.UseSin;

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
            public IComponentFactory<Float, IFourierDistributionSampler> MatrixGenerator;

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
            public readonly float[] RndFourierVectors;

            // the random rotations
            public readonly float[] RotationTerms;

            private readonly IFourierDistributionSampler _matrixGenerator;
            private readonly bool _useSin;
            private readonly TauswortheHybrid _rand;
            private readonly TauswortheHybrid.State _state;

            public TransformInfo(IHost host, Column item, Arguments args, int d, Float avgDist)
            {
                Contracts.AssertValue(host);

                SrcDim = d;
                NewDim = item.NewDim ?? args.NewDim;
                host.CheckUserArg(NewDim > 0, nameof(item.NewDim));
                _useSin = item.UseSin ?? args.UseSin;
                var seed = item.Seed ?? args.Seed;
                _rand = seed.HasValue ? RandomUtils.Create(seed) : RandomUtils.Create(host.Rand);
                _state = _rand.GetState();

                var generator = item.MatrixGenerator;
                if (generator == null)
                    generator = args.MatrixGenerator;
                _matrixGenerator = generator.CreateComponent(host, avgDist);

                int roundedUpD = RoundUp(NewDim, _cfltAlign);
                int roundedUpNumFeatures = RoundUp(SrcDim, _cfltAlign);
                RndFourierVectors = new float[roundedUpD * roundedUpNumFeatures];
                RotationTerms = _useSin ? null : new float[roundedUpD];

                InitializeFourierCoefficients(roundedUpNumFeatures, roundedUpD);
            }

            public TransformInfo(IHostEnvironment env, ModelLoadContext ctx, int colValueCount, string directoryName)
            {
                env.AssertValue(env);
                env.Assert(colValueCount > 0);

                // *** Binary format ***
                // int: d (number of untransformed features)
                // int: NewDim (number of transformed features)
                // bool: UseSin
                // uint[4]: the seeds for the pseudo random number generator.

                SrcDim = ctx.Reader.ReadInt32();
                env.CheckDecode(SrcDim == colValueCount);

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
                RndFourierVectors = new float[roundedUpD * roundedUpNumFeatures];
                RotationTerms = _useSin ? null : new float[roundedUpD];
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
                    RotationTerms[i] = (_rand.NextSingle() - (Float)0.5) * (Float)Math.PI;
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
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RffTransform).Assembly.FullName);
        }

        // These are parallel to Infos.
        private readonly ColumnType[] _types;
        private readonly TransformInfo[] _transformInfos;

        private const string RegistrationName = "Rff";
        private static readonly int _cfltAlign = CpuMathUtils.GetVectorAlignment() / sizeof(float);

        private static string TestColumnType(ColumnType type)
        {
            if (type.ItemType == NumberType.Float && type.ValueCount > 0)
                return null;
            return "Expected R4 or vector of R4 with known size";
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="newDim">The number of random Fourier features to create.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        public RffTransform(IHostEnvironment env,
            IDataView input,
            int newDim,
            string name,
            string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, NewDim = newDim }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to <see cref="SignatureDataTransform"/>.
        /// </summary>
        public RffTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestColumnType)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));
            Host.CheckUserArg(
                args.Column.All(c => c.Seed.HasValue) ||
                args.Column.All(c => !c.Seed.HasValue) || args.Seed.HasValue, nameof(args.Seed),
                "If any column specific seeds are non-zero, the global transform seed must also be non-zero, to make results deterministic");

            _transformInfos = new TransformInfo[args.Column.Length];

            var avgDistances = Train(Host, Infos, args, input);

            for (int i = 0; i < _transformInfos.Length; i++)
            {
                _transformInfos[i] = new TransformInfo(Host.Register(string.Format("column{0}", i)), args.Column[i], args,
                    Infos[i].TypeSrc.ValueCount, avgDistances[i]);
            }

            _types = InitColumnTypes();
        }

        private RffTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestColumnType)
        {
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // transformInfos
            Host.AssertNonEmpty(Infos);
            _transformInfos = new TransformInfo[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                _transformInfos[i] = new TransformInfo(Host, ctx, Infos[i].TypeSrc.ValueCount,
                    string.Format("MatrixGenerator{0}", i));
            }
            _types = InitColumnTypes();
        }

        public static RffTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    h.CheckDecode(cbFloat == sizeof(Float));
                    return new RffTransform(h, ctx, input);
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
            // transformInfos
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
                _transformInfos[i].Save(ctx, string.Format("MatrixGenerator{0}", i));
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

        private static Float[] Train(IHost host, ColInfo[] infos, Arguments args, IDataView trainingData)
        {
            Contracts.AssertValue(host, "host");
            host.AssertNonEmpty(infos);

            var avgDistances = new Float[infos.Length];
            const int reservoirSize = 5000;

            bool[] activeColumns = new bool[trainingData.Schema.ColumnCount];
            for (int i = 0; i < infos.Length; i++)
                activeColumns[infos[i].Source] = true;

            var reservoirSamplers = new ReservoirSamplerWithReplacement<VBuffer<Float>>[infos.Length];
            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var rng = args.Seed.HasValue ? RandomUtils.Create(args.Seed) : host.Rand;
                for (int i = 0; i < infos.Length; i++)
                {
                    if (infos[i].TypeSrc.IsVector)
                    {
                        var get = cursor.GetGetter<VBuffer<Float>>(infos[i].Source);
                        reservoirSamplers[i] = new ReservoirSamplerWithReplacement<VBuffer<Float>>(rng, reservoirSize, get);
                    }
                    else
                    {
                        var getOne = cursor.GetGetter<Float>(infos[i].Source);
                        Float val = 0;
                        ValueGetter<VBuffer<Float>> get =
                            (ref VBuffer<Float> dst) =>
                            {
                                getOne(ref val);
                                dst = new VBuffer<float>(1, new[] { val });
                            };
                        reservoirSamplers[i] = new ReservoirSamplerWithReplacement<VBuffer<Float>>(rng, reservoirSize, get);
                    }
                }

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < infos.Length; i++)
                        reservoirSamplers[i].Sample();
                }
                for (int i = 0; i < infos.Length; i++)
                    reservoirSamplers[i].Lock();
            }

            for (int iinfo = 0; iinfo < infos.Length; iinfo++)
            {
                var instanceCount = reservoirSamplers[iinfo].NumSampled;

                // If the number of pairs is at most the maximum reservoir size / 2, we go over all the pairs,
                // so we get all the examples. Otherwise, get a sample with replacement.
                VBuffer<Float>[] res;
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
                    Float[] distances;
                    var sub = args.Column[iinfo].MatrixGenerator;
                    if (sub == null)
                        sub = args.MatrixGenerator;
                    // create a dummy generator in order to get its type.
                    // REVIEW this should be refactored. See https://github.com/dotnet/machinelearning/issues/699
                    var matrixGenerator = sub.CreateComponent(host, 1);
                    bool gaussian = matrixGenerator is GaussianFourierSampler;

                    // If the number of pairs is at most the maximum reservoir size / 2, go over all the pairs.
                    if (resLength < reservoirSize)
                    {
                        distances = new Float[instanceCount * (instanceCount - 1) / 2];
                        int count = 0;
                        for (int i = 0; i < instanceCount; i++)
                        {
                            for (int j = i + 1; j < instanceCount; j++)
                            {
                                distances[count++] = gaussian ? VectorUtils.L2DistSquared(ref res[i], ref res[j])
                                    : VectorUtils.L1Distance(ref res[i], ref res[j]);
                            }
                        }
                        host.Assert(count == distances.Length);
                    }
                    else
                    {
                        distances = new Float[reservoirSize / 2];
                        for (int i = 0; i < reservoirSize - 1; i += 2)
                        {
                            // For Gaussian kernels, we scale by the L2 distance squared, since the kernel function is exp(-gamma ||x-y||^2).
                            // For Laplacian kernels, we scale by the L1 distance, since the kernel function is exp(-gamma ||x-y||_1).
                            distances[i / 2] = gaussian ? VectorUtils.L2DistSquared(ref res[i], ref res[i + 1]) :
                                VectorUtils.L1Distance(ref res[i], ref res[i + 1]);
                        }
                    }

                    // If by chance, in the random permutation all the pairs are the same instance we return 1.
                    Float median = MathUtils.GetMedianInPlace(distances, distances.Length);
                    avgDistances[iinfo] = median == 0 ? 1 : median;
                }
            }
            return avgDistances;
        }

        private ColumnType[] InitColumnTypes()
        {
            Host.Assert(Infos.Length == _transformInfos.Length);
            var types = new ColumnType[Infos.Length];
            for (int i = 0; i < _transformInfos.Length; i++)
            {
                types[i] = new VectorType(NumberType.Float, _transformInfos[i].RotationTerms == null ?
                    _transformInfos[i].NewDim * 2 : _transformInfos[i].NewDim);
            }
            Metadata.Seal();
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            if (info.TypeSrc.IsVector)
                return GetterFromVectorType(input, iinfo);
            return GetterFromFloatType(input, iinfo);
        }

        private ValueGetter<VBuffer<Float>> GetterFromVectorType(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<VBuffer<Float>>(input, iinfo);
            var src = default(VBuffer<Float>);

            var featuresAligned = new float[RoundUp(Infos[iinfo].TypeSrc.ValueCount, _cfltAlign)];
            var productAligned = new float[RoundUp(_transformInfos[iinfo].NewDim, _cfltAlign)];

            return
                (ref VBuffer<Float> dst) =>
                {
                    getSrc(ref src);
                    TransformFeatures(Host, ref src, ref dst, _transformInfos[iinfo], featuresAligned, productAligned);
                };
        }

        private ValueGetter<VBuffer<Float>> GetterFromFloatType(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<Float>(input, iinfo);
            var src = default(Float);

            var featuresAligned = new float[RoundUp(1, _cfltAlign)];
            var productAligned = new float[RoundUp(_transformInfos[iinfo].NewDim, _cfltAlign)];

            var oneDimensionalVector = new VBuffer<Float>(1, new Float[] { 0 });

            return
                (ref VBuffer<Float> dst) =>
                {
                    getSrc(ref src);
                    oneDimensionalVector.Values[0] = src;
                    TransformFeatures(Host, ref oneDimensionalVector, ref dst, _transformInfos[iinfo], featuresAligned, productAligned);
                };
        }

        private static void TransformFeatures(IHost host, ref VBuffer<Float> src, ref VBuffer<Float> dst, TransformInfo transformInfo,
            float[] featuresAligned, float[] productAligned)
        {
            Contracts.AssertValue(host, "host");
            host.Check(src.Length == transformInfo.SrcDim, "column does not have the expected dimensionality.");

            var values = dst.Values;
            Float scale;
            if (transformInfo.RotationTerms != null)
            {
                if (Utils.Size(values) < transformInfo.NewDim)
                    values = new Float[transformInfo.NewDim];
                scale = MathUtils.Sqrt((Float)2.0 / transformInfo.NewDim);
            }
            else
            {
                if (Utils.Size(values) < 2 * transformInfo.NewDim)
                    values = new Float[2 * transformInfo.NewDim];
                scale = MathUtils.Sqrt((Float)1.0 / transformInfo.NewDim);
            }

            if (src.IsDense)
            {
                Array.Copy(src.Values, 0, featuresAligned, 0, src.Length);
                CpuMathUtils.MatTimesSrc(false, false, transformInfo.RndFourierVectors, featuresAligned, productAligned,
                    transformInfo.NewDim);
            }
            else
            {
                // This overload of MatTimesSrc ignores the values in slots that are not in src.Indices, so there is
                // no need to zero them out.
                for (int ipos = 0; ipos < src.Count; ++ipos)
                {
                    int iv = src.Indices[ipos];
                    featuresAligned[iv] = src.Values[ipos];
                }

                CpuMathUtils.MatTimesSrc(false, false, transformInfo.RndFourierVectors, src.Indices, featuresAligned, 0, 0,
                    src.Count, productAligned, transformInfo.NewDim);
            }

            for (int i = 0; i < transformInfo.NewDim; i++)
            {
                var dotProduct = productAligned[i];
                if (transformInfo.RotationTerms != null)
                    values[i] = (Float)MathUtils.Cos(dotProduct + transformInfo.RotationTerms[i]) * scale;
                else
                {
                    values[2 * i] = (Float)MathUtils.Cos(dotProduct) * scale;
                    values[2 * i + 1] = (Float)MathUtils.Sin(dotProduct) * scale;
                }
            }

            dst = new VBuffer<Float>(transformInfo.RotationTerms == null ? 2 * transformInfo.NewDim : transformInfo.NewDim,
                values, dst.Indices);
        }
    }
}
