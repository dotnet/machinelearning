// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Core.Data;

[assembly: LoadableClass(RandomizedPcaTrainer.Summary, typeof(RandomizedPcaTrainer), typeof(RandomizedPcaTrainer.Arguments),
    new[] { typeof(SignatureAnomalyDetectorTrainer), typeof(SignatureTrainer) },
    RandomizedPcaTrainer.UserNameValue,
    RandomizedPcaTrainer.LoadNameValue,
    RandomizedPcaTrainer.ShortName)]

[assembly: LoadableClass(typeof(PcaPredictor), null, typeof(SignatureLoadModel),
    "PCA Anomaly Executor", PcaPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(RandomizedPcaTrainer), null, typeof(SignatureEntryPointModule), RandomizedPcaTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers.PCA
{
    // REVIEW: make RFF transformer an option here.

    /// <summary>
    /// This trainer trains an approximate PCA using Randomized SVD algorithm
    /// Reference: https://web.stanford.edu/group/mmds/slides2010/Martinsson.pdf
    /// </summary>
    /// <remarks>
    /// This PCA can be made into Kernel PCA by using Random Fourier Features transform
    /// </remarks>
    public sealed class RandomizedPcaTrainer : TrainerEstimatorBase<AnomalyPredictionTransformer<PcaPredictor>, PcaPredictor>
    {
        public const string LoadNameValue = "pcaAnomaly";
        internal const string UserNameValue = "PCA Anomaly Detector";
        internal const string ShortName = "pcaAnom";
        internal const string Summary = "This algorithm trains an approximate PCA using Randomized SVD algorithm. "
            + "This PCA can be made into Kernel PCA by using Random Fourier Features transform.";

        public class Arguments : UnsupervisedLearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of components in the PCA", ShortName = "k", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "10,20,40,80")]
            [TlcModule.SweepableDiscreteParam("Rank", new object[] { 10, 20, 40, 80 })]
            public int Rank = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Oversampling parameter for randomized PCA training", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "10,20,40")]
            [TlcModule.SweepableDiscreteParam("Oversampling", new object[] { 10, 20, 40 })]
            public int Oversampling = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If enabled, data is centered to be zero mean", ShortName = "center")]
            [TlcModule.SweepableDiscreteParam("Center", null, isBool: true)]
            public bool Center = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed for random number generation", ShortName = "seed")]
            public int? Seed;
        }

        private readonly int _rank;
        private readonly int _oversampling;
        private readonly bool _center;
        private readonly int _seed;
        private readonly string _featureColumn;

        public override PredictionKind PredictionKind => PredictionKind.AnomalyDetection;

        // The training performs two passes, only. Probably not worth caching.
        private static readonly TrainerInfo _info = new TrainerInfo(caching: false);
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Initializes a new instance of <see cref="RandomizedPcaTrainer"/>.
        /// </summary>
        /// <param name="env">The local instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="oversampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        public RandomizedPcaTrainer(IHostEnvironment env, string featureColumn, string weightColumn = null,
            int rank = 20, int oversampling = 20, bool center = true, int? seed = null)
            : this(env, null, featureColumn, weightColumn, rank, oversampling, center, seed)
        {

        }

        internal RandomizedPcaTrainer(IHostEnvironment env, Arguments args)
            :this(env, args, args.FeatureColumn, args.WeightColumn)
        {

        }

        private RandomizedPcaTrainer(IHostEnvironment env, Arguments args, string featureColumn, string weightColumn,
            int rank = 20, int oversampling = 20, bool center = true, int? seed = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(featureColumn), null, TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            // if the args are not null, we got here from maml, and the internal ctor.
            if (args != null)
            {
                _rank = args.Rank;
                _center = args.Center;
                _oversampling = args.Oversampling;
                _seed = args.Seed ?? Host.Rand.Next();
            }
            else
            {
                _rank = rank;
                _center = center;
                _oversampling = oversampling;
                _seed = seed ?? Host.Rand.Next();
            }

            _featureColumn = featureColumn;

            Host.CheckUserArg(_rank > 0, nameof(_rank), "Rank must be positive");
            Host.CheckUserArg(_oversampling >= 0, nameof(_oversampling), "Oversampling must be non-negative");

        }

        //Note: the notations used here are the same as in https://web.stanford.edu/group/mmds/slides2010/Martinsson.pdf (pg. 9)
        protected override PcaPredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));

            context.TrainingSet.CheckFeatureFloatVector(out int dimension);

            using (var ch = Host.Start("Training"))
            {
                return TrainCore(ch, context.TrainingSet, dimension);
            }
        }

        private static SchemaShape.Column MakeWeightColumn(string weightColumn)
        {
            if (weightColumn == null)
                return null;
            return new SchemaShape.Column(weightColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
        }

        private static SchemaShape.Column MakeFeatureColumn(string featureColumn)
        {
            return new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
        }

        private PcaPredictor TrainCore(IChannel ch, RoleMappedData data, int dimension)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            if (_rank > dimension)
                throw ch.Except("Rank ({0}) cannot be larger than the original dimension ({1})", _rank, dimension);
            int oversampledRank = Math.Min(_rank + _oversampling, dimension);

            //exact: (size of the 2 big matrices + other minor allocations) / (2^30)
            Double memoryUsageEstimate = 2.0 * dimension * oversampledRank * sizeof(float) / 1e9;
            if (memoryUsageEstimate > 2)
                ch.Info("Estimate memory usage: {0:G2} GB. If running out of memory, reduce rank and oversampling factor.", memoryUsageEstimate);

            var y = Zeros(oversampledRank, dimension);
            var mean = _center ? VBufferUtils.CreateDense<float>(dimension) : VBufferUtils.CreateEmpty<float>(dimension);

            var omega = GaussianMatrix(oversampledRank, dimension, _seed);

            var cursorFactory = new FeatureFloatVectorCursor.Factory(data, CursOpt.Features | CursOpt.Weight);
            long numBad;
            Project(Host, cursorFactory, ref mean, omega, y, out numBad);
            if (numBad > 0)
                ch.Warning("Skipped {0} instances with missing features/weights during training", numBad);

            //Orthonormalize Y in-place using stabilized Gram Schmidt algorithm.
            //Ref: https://en.wikipedia.org/wiki/Gram-Schmidt#Algorithm
            for (var i = 0; i < oversampledRank; ++i)
            {
                var v = y[i];
                VectorUtils.ScaleBy(ref v, 1 / VectorUtils.Norm(y[i]));

                // Make the next vectors in the queue orthogonal to the orthonormalized vectors.
                for (var j = i + 1; j < oversampledRank; ++j) //subtract the projection of y[j] on v.
                    VectorUtils.AddMult(in v, -VectorUtils.DotProduct(in v, in y[j]), ref y[j]);
            }
            var q = y; // q in QR decomposition.

            var b = omega; // reuse the memory allocated by Omega.
            Project(Host, cursorFactory, ref mean, q, b, out numBad);

            //Compute B2 = B' * B
            var b2 = new float[oversampledRank * oversampledRank];
            for (var i = 0; i < oversampledRank; ++i)
            {
                for (var j = i; j < oversampledRank; ++j)
                    b2[i * oversampledRank + j] = b2[j * oversampledRank + i] = VectorUtils.DotProduct(in b[i], in b[j]);
            }

            float[] smallEigenvalues;// eigenvectors and eigenvalues of the small matrix B2.
            float[] smallEigenvectors;
            EigenUtils.EigenDecomposition(b2, out smallEigenvalues, out smallEigenvectors);
            PostProcess(b, smallEigenvalues, smallEigenvectors, dimension, oversampledRank);

            return new PcaPredictor(Host, _rank, b, in mean);
        }

        private static VBuffer<float>[] Zeros(int k, int d)
        {
            var rv = new VBuffer<float>[k];
            for (var i = 0; i < k; ++i)
                rv[i] = VBufferUtils.CreateDense<float>(d);
            return rv;
        }

        private static VBuffer<float>[] GaussianMatrix(int k, int d, int seed)
        {
            var rv = Zeros(k, d);
            var rng = new SysRandom(seed);

            // REVIEW: use a faster Gaussian random matrix generator
            //MKL has a fast vectorized random number generation.
            for (var i = 0; i < k; ++i)
            {
                for (var j = 0; j < d; ++j)
                    rv[i].Values[j] = (float)Stats.SampleFromGaussian(rng); // not fast for large matrix generation
            }
            return rv;
        }

        //Project the covariance matrix A on to Omega: Y <- A * Omega
        //A = X' * X / n, where X = data - mean
        //Note that the covariance matrix is not computed explicitly
        private static void Project(IHost host, FeatureFloatVectorCursor.Factory cursorFactory, ref VBuffer<float> mean, VBuffer<float>[] omega, VBuffer<float>[] y, out long numBad)
        {
            Contracts.AssertValue(host, "host");
            host.AssertNonEmpty(omega);
            host.Assert(Utils.Size(y) == omega.Length); // Size of Y and Omega: dimension x oversampled rank
            int numCols = omega.Length;

            for (int i = 0; i < y.Length; ++i)
                VBufferUtils.Clear(ref y[i]);

            bool center = mean.IsDense;
            float n = 0;
            long count = 0;
            using (var pch = host.StartProgressChannel("Project covariance matrix"))
            using (var cursor = cursorFactory.Create())
            {
                pch.SetHeader(new ProgressHeader(new[] { "rows" }), e => e.SetProgress(0, count));
                while (cursor.MoveNext())
                {
                    if (center)
                        VectorUtils.AddMult(in cursor.Features, cursor.Weight, ref mean);
                    for (int i = 0; i < numCols; i++)
                    {
                        VectorUtils.AddMult(
                            in cursor.Features,
                            cursor.Weight * VectorUtils.DotProduct(in omega[i], in cursor.Features),
                            ref y[i]);
                    }
                    n += cursor.Weight;
                    count++;
                }
                pch.Checkpoint(count);
                numBad = cursor.SkippedRowCount;
            }

            Contracts.Check(n > 0, "Empty training data");
            float invn = 1 / n;

            for (var i = 0; i < numCols; ++i)
                VectorUtils.ScaleBy(ref y[i], invn);

            if (center)
            {
                VectorUtils.ScaleBy(ref mean, invn);
                for (int i = 0; i < numCols; i++)
                    VectorUtils.AddMult(in mean, -VectorUtils.DotProduct(in omega[i], in mean), ref y[i]);
            }
        }

        /// <summary>
        /// Modifies <paramref name="y"/> in place so it becomes <paramref name="y"/> * eigenvectors / eigenvalues.
        /// </summary>
        // REVIEW: improve
        private static void PostProcess(VBuffer<float>[] y, float[] sigma, float[] z, int d, int k)
        {
            Contracts.Assert(y.All(v => v.IsDense));
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
                        tmp[j] += y[l].Values[i] * z[j * k + l];
                }
                for (int j = 0; j < k; j++)
                    y[j].Values[i] = pinv[j] * tmp[j];
            }
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
             return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberType.R4,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),

                new SchemaShape.Column(DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BoolType.Instance,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override AnomalyPredictionTransformer<PcaPredictor> MakeTransformer(PcaPredictor model, Schema trainSchema)
            => new AnomalyPredictionTransformer<PcaPredictor>(Host, model, trainSchema, _featureColumn);

        [TlcModule.EntryPoint(Name = "Trainers.PcaAnomalyDetector",
            Desc = "Train an PCA Anomaly model.",
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/member[@name=""PCA""]/*' />",
                                 @"<include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/example[@name=""PcaAnomalyDetector""]/*' />" })]
        public static CommonOutputs.AnomalyDetectionOutput TrainPcaAnomaly(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainPCAAnomaly");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.AnomalyDetectionOutput>(host, input,
                () => new RandomizedPcaTrainer(host, input),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }

    // An anomaly detector using PCA.
    // - The algorithm uses the top eigenvectors to approximate the subspace containing the normal class
    // - For each new instance, it computes the norm difference between the raw feature vector and the projected feature on that subspace.
    // - - If the error is close to 0, the instance is considered normal (non-anomaly).
    // REVIEW: move the predictor to a different file and fold EigenUtils.cs to this file.
    // REVIEW: Include the above detail in the XML documentation file.
    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
    public sealed class PcaPredictor : PredictorBase<float>,
        IValueMapper,
        ICanGetSummaryAsIDataView,
        ICanSaveInTextFormat, ICanSaveModel, ICanSaveSummary
    {
        public const string LoaderSignature = "pcaAnomExec";
        public const string RegistrationName = "PCAPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCA ANOM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PcaPredictor).Assembly.FullName);
        }

        private readonly int _dimension;
        private readonly int _rank;
        private readonly VBuffer<float>[] _eigenVectors; // top-k eigenvectors of the train data's covariance matrix
        private readonly float[] _meanProjected; // for centering
        private readonly VBuffer<float> _mean; // used to compute (x-mu)^2
        private readonly float _norm2Mean;

        private readonly ColumnType _inputType;

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.AnomalyDetection; }
        }

        internal PcaPredictor(IHostEnvironment env, int rank, VBuffer<float>[] eigenVectors, in VBuffer<float> mean)
            : base(env, RegistrationName)
        {
            _dimension = eigenVectors[0].Length;
            _rank = rank;
            _eigenVectors = new VBuffer<float>[rank];
            _meanProjected = new float[rank];

            for (var i = 0; i < rank; ++i) // Only want first k
            {
                _eigenVectors[i] = eigenVectors[i];
                _meanProjected[i] = VectorUtils.DotProduct(in eigenVectors[i], in mean);
            }

            _mean = mean;
            _norm2Mean = VectorUtils.NormSquared(mean);

            _inputType = new VectorType(NumberType.Float, _dimension);
        }

        private PcaPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // int: dimension (aka. number of features)
            // int: rank
            // bool: center
            // If (center)
            //  float[]: mean vector
            // float[][]: eigenvectors
            _dimension = ctx.Reader.ReadInt32();
            Host.CheckDecode(FloatUtils.IsFinite(_dimension));

            _rank = ctx.Reader.ReadInt32();
            Host.CheckDecode(FloatUtils.IsFinite(_rank));

            bool center = ctx.Reader.ReadBoolByte();
            if (center)
            {
                var meanArray = ctx.Reader.ReadFloatArray(_dimension);
                Host.CheckDecode(meanArray.All(FloatUtils.IsFinite));
                _mean = new VBuffer<float>(_dimension, meanArray);
                _norm2Mean = VectorUtils.NormSquared(_mean);
            }
            else
            {
                _mean = VBufferUtils.CreateEmpty<float>(_dimension);
                _norm2Mean = 0;
            }

            _eigenVectors = new VBuffer<float>[_rank];
            _meanProjected = new float[_rank];
            for (int i = 0; i < _rank; ++i)
            {
                var vi = ctx.Reader.ReadFloatArray(_dimension);
                Host.CheckDecode(vi.All(FloatUtils.IsFinite));
                _eigenVectors[i] = new VBuffer<float>(_dimension, vi);
                _meanProjected[i] = VectorUtils.DotProduct(in _eigenVectors[i], in _mean);
            }
            WarnOnOldNormalizer(ctx, GetType(), Host);

            _inputType = new VectorType(NumberType.Float, _dimension);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            var writer = ctx.Writer;

            // *** Binary format ***
            // int: dimension (aka. number of features)
            // int: rank
            // bool: center
            // If (center)
            //  float[]: mean vector
            // float[][]: eigenvectors
            writer.Write(_dimension);
            writer.Write(_rank);

            if (_mean.IsDense) // centered
            {
                writer.WriteBoolByte(true);
                writer.WriteFloatsNoCount(_mean.Values, _dimension);
            }
            else
                writer.WriteBoolByte(false);

            for (int i = 0; i < _rank; ++i)
                writer.WriteFloatsNoCount(_eigenVectors[i].Values, _dimension);
        }

        public static PcaPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PcaPredictor(env, ctx);
        }

        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            SaveAsText(writer, schema);
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("Dimension: {0}", _dimension);
            writer.WriteLine("Rank: {0}", _rank);

            if (_mean.IsDense)
            {
                writer.Write("Mean vector:");
                foreach (var value in _mean.Items(all: true))
                    writer.Write(" {0}", value.Value);

                writer.WriteLine();
                writer.Write("Projected mean vector:");
                foreach (var value in _meanProjected)
                    writer.Write(" {0}", value);
            }

            writer.WriteLine();
            writer.WriteLine("# V");
            for (var i = 0; i < _rank; ++i)
            {
                VBufferUtils.ForEachDefined(in _eigenVectors[i],
                    (ind, val) => { if (val != 0) writer.Write(" {0}:{1}", ind, val); });
                writer.WriteLine();
            }
        }

        public IDataView GetSummaryDataView(RoleMappedSchema schema)
        {
            var bldr = new ArrayDataViewBuilder(Host);

            var cols = new VBuffer<float>[_rank + 1];
            var names = new string[_rank + 1];
            for (var i = 0; i < _rank; ++i)
            {
                names[i] = "EigenVector" + i;
                cols[i] = _eigenVectors[i];
            }
            names[_rank] = "MeanVector";
            cols[_rank] = _mean;

            bldr.AddColumn("VectorName", names);
            bldr.AddColumn("VectorData", NumberType.R4, cols);

            return bldr.GetDataView();
        }

        public ColumnType InputType
        {
            get { return _inputType; }
        }

        public ColumnType OutputType
        {
            get { return NumberType.Float; }
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del =
                (ref VBuffer<float> src, ref float dst) =>
                {
                    Host.Check(src.Length == _dimension);
                    dst = Score(in src);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private float Score(in VBuffer<float> src)
        {
            Host.Assert(src.Length == _dimension);

            // REVIEW: Can this be done faster in a single pass over src and _mean?
            var mean = _mean;
            float norm2X = VectorUtils.NormSquared(in src) -
                2 * VectorUtils.DotProduct(in mean, in src) + _norm2Mean;
            // Because the distance between src and _mean is computed using the above expression, the result
            // may be negative due to round off error. If this happens, we let the distance be 0.
            if (norm2X < 0)
                norm2X = 0;

            float norm2U = 0;
            for (int i = 0; i < _rank; i++)
            {
                float component = VectorUtils.DotProduct(in _eigenVectors[i], in src) - _meanProjected[i];
                norm2U += component * component;
            }

            return MathUtils.Sqrt((norm2X - norm2U) / norm2X); // normalized error
        }

        /// <summary>
        /// Copies the top eigenvectors of the covariance matrix of the training data
        /// into a set of buffers.
        /// </summary>
        /// <param name="vectors">A possibly reusable set of vectors, which will
        /// be expanded as necessary to accomodate the data.</param>
        /// <param name="rank">Set to the rank, which is also the logical length
        /// of <paramref name="vectors"/>.</param>
        public void GetEigenVectors(ref VBuffer<float>[] vectors, out int rank)
        {
            rank = _eigenVectors.Length;
            Utils.EnsureSize(ref vectors, _eigenVectors.Length, _eigenVectors.Length);
            for (int i = 0; i < _eigenVectors.Length; i++)
                _eigenVectors[i].CopyTo(ref vectors[i]);
        }

        /// <summary>
        /// Copies the mean vector of the training data.
        /// </summary>
        public void GetMean(ref VBuffer<float> mean)
        {
            _mean.CopyTo(ref mean);
        }
    }
}
