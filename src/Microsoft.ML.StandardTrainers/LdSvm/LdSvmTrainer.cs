// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(LdSvmTrainer.Summary, typeof(LdSvmTrainer), typeof(LdSvmTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    LdSvmTrainer.UserNameValue,
    LdSvmTrainer.LoadNameValue
    )]

[assembly: LoadableClass(typeof(void), typeof(LdSvmTrainer), null, typeof(SignatureEntryPointModule), LdSvmTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> to predict a target using a non-linear binary classification model
    /// trained with Local Deep SVM.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [LdSvm](xref:Microsoft.ML.StandardTrainersCatalog.LdSvm(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,System.String,System.String,System.String,System.Int32,System.Int32,System.Boolean,System.Boolean))
    /// or [LdSvm(Options)](xref:Microsoft.ML.StandardTrainersCatalog.LdSvm(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.LdSvmTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-binary-classification-no-prob.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Binary classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | No |
    ///
    /// ### Training Algorithm Details
    /// Local Deep SVM (LD-SVM) is a generalization of Localized Multiple Kernel Learning for non-linear SVM. Multiple kernel methods learn a different
    /// kernel, and hence a different classifier, for each point in the feature space. The prediction time cost for multiple kernel methods can be prohibitively
    /// expensive for large training sets because it is proportional to the number of support vectors, and these grow linearly with the size of the training
    /// set. LD-SVM reduces the prediction cost by learning a tree-based local feature embedding that is high dimensional and sparse, efficiently encoding
    /// non-linearities. Using LD-SVM, the prediction cost grows logarithmically with the size of the training set, rather than linearly, with a tolerable loss
    /// in classification accuracy.
    ///
    /// Local Deep SVM is an implementation of the algorithm described in [C. Jose, P. Goyal, P. Aggrwal, and M. Varma, Local Deep
    /// Kernel Learning for Efficient Non-linear SVM Prediction, ICML, 2013](http://proceedings.mlr.press/v28/jose13.pdf).
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="StandardTrainersCatalog.LdSvm(BinaryClassificationCatalog.BinaryClassificationTrainers, LdSvmTrainer.Options)"/>
    /// <seealso cref="StandardTrainersCatalog.LdSvm(BinaryClassificationCatalog.BinaryClassificationTrainers, string, string, string, int, int, bool, bool)"/>
    public sealed class LdSvmTrainer : TrainerEstimatorBase<BinaryPredictionTransformer<LdSvmModelParameters>, LdSvmModelParameters>
    {
        internal const string LoadNameValue = "LDSVM";
        internal const string UserNameValue = "Local Deep SVM (LDSVM)";
        internal const string Summary = "LD-SVM learns a binary, non-linear SVM classifier with a kernel that is specifically designed to reduce prediction time. "
            + "LD-SVM learns decision boundaries that are locally linear.";

        public sealed class Options : TrainerInputBaseWithWeight
        {
            /// <summary>
            /// Depth of LDSVM Tree
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Depth of Local Deep SVM tree", ShortName = "depth", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "1,3,5,7")]
            [TlcModule.SweepableDiscreteParam("TreeDepth", new object[] { 1, 3, 5, 7 })]
            public int TreeDepth = Defaults.TreeDepth;

            /// <summary>
            ///  Regularizer for classifier parameter W
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularizer for classifier parameter W", ShortName = "lw", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0.1,0.01,0.001")]
            [TlcModule.SweepableDiscreteParam("LambdaW", new object[] { 0.1f, 0.01f, 0.001f })]
            public float LambdaW = Defaults.LambdaW;

            /// <summary>
            ///  Regularizer for kernel parameter Theta
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularizer for kernel parameter Theta", ShortName = "lt", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0.1,0.01,0.001")]
            [TlcModule.SweepableDiscreteParam("LambdaTheta", new object[] { 0.1f, 0.01f, 0.001f })]
            public float LambdaTheta = Defaults.LambdaTheta;

            /// <summary>
            ///  Regularizer for kernel parameter ThetaPrime
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularizer for kernel parameter Thetaprime", ShortName = "lp", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0.1,0.01,0.001")]
            [TlcModule.SweepableDiscreteParam("LambdaThetaprime", new object[] { 0.1f, 0.01f, 0.001f })]
            public float LambdaThetaprime = Defaults.LambdaThetaprime;

            /// <summary>
            ///  Parameter for sigmoid sharpness
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for sigmoid sharpness", ShortName = "s", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "1.0,0.1,0.01")]
            [TlcModule.SweepableDiscreteParam("Sigma", new object[] { 1.0f, 0.1f, 0.01f })]
            public float Sigma = Defaults.Sigma;

            /// <summary>
            /// Indicates if we should use Bias or not in our model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "No bias", ShortName = "bias")]
            [TlcModule.SweepableDiscreteParam("NoBias", null, isBool: true)]
            public bool UseBias = Defaults.UseBias;

            /// <summary>
            /// Number of iterations
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of iterations", ShortName = "iter,NumIterations", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "10000,15000")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 10000, 15000 })]
            public int NumberOfIterations = Defaults.NumberOfIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal int MaxCalibrationExamples = 1000000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to cache the data before the first iteration")]
            public bool Cache = Defaults.Cache;

            internal class Defaults
            {
                public const int NumberOfIterations = 15000;
                public const bool UseBias = true;
                public const float Sigma = 1.0f;
                public const float LambdaThetaprime = 0.01f;
                public const float LambdaTheta = 0.01f;
                public const float LambdaW = 0.1f;
                public const int TreeDepth = 3;
                public const bool Cache = true;
            }
        }

        private const int NumberOfSamplesForGammaUpdate = 100;

        private readonly Options _options;

        internal LdSvmTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue),
                  TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName),
                  TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName))
        {
            Host.CheckValue(options, nameof(options));
            CheckOptions(Host, options);
            _options = options;
        }

        private static readonly TrainerInfo _info = new TrainerInfo(calibration: true, caching: false);
        public override TrainerInfo Info => _info;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override LdSvmModelParameters TrainModelCore(TrainContext trainContext)
        {
            Host.CheckValue(trainContext, nameof(trainContext));
            using (var ch = Host.Start("Training"))
            {
                trainContext.TrainingSet.CheckFeatureFloatVector(out var numFeatures);
                trainContext.TrainingSet.CheckBinaryLabel();

                var numLeaf = 1 << _options.TreeDepth;
                return TrainCore(ch, trainContext.TrainingSet, numLeaf, numFeatures);
            }
        }

        /// <summary>
        /// Compute gradient w.r.t theta for an instance X
        /// </summary>
        private void ComputeGradTheta(in VBuffer<float> feat, float[] gradTheta, int numLeaf, float gamma,
            VBuffer<float>[] theta, float[] biasTheta, float[] pathWt, float[] localWt, VBuffer<float>[] w, float[] biasW)
        {
            Array.Clear(gradTheta, 0, numLeaf - 1);
            int numNodes = 2 * numLeaf - 1;
            float[] tanhThetaTx = new float[numLeaf - 1];
            for (int i = 0; i < numLeaf - 1; i++)
                tanhThetaTx[i] = (float)Math.Tanh(gamma * (VectorUtils.DotProduct(in feat, in theta[i]) + biasTheta[i]));
            for (int i = 0; i < numNodes; i++)
            {
                int current = i;
                float tempGrad = pathWt[i] * localWt[i] * (VectorUtils.DotProduct(in feat, in w[i]) + biasW[i]);
                while (current > 0)
                {
                    int parent = (current - 1) / 2;
                    gradTheta[parent] += tempGrad * (current % 2 == 1 ? (1 - tanhThetaTx[parent]) : (-1 - tanhThetaTx[parent]));
                    current = parent;
                }
            }
        }

        /// <summary>
        /// Adaptively update gamma for indicator function approximation.
        /// </summary>
        private void UpdateGamma(int iter, int numLeaf, ref float gamma, Data data, VBuffer<float>[] theta, float[] biasTheta)
        {
            if (numLeaf == 1)
                gamma = 1.0f;
            else
            {
                float tempSum = 0;
                var sample = data.SampleForGammaUpdate(Host.Rand);
                int sampleSize = 0;
                foreach (var s in sample)
                {
                    int thetaIdx = Host.Rand.Next(numLeaf - 1);
                    tempSum += Math.Abs(VectorUtils.DotProduct(in s, in theta[thetaIdx]) + biasTheta[thetaIdx]);
                    sampleSize++;
                }
                tempSum /= sampleSize;
                gamma = 0.1f / tempSum;
                gamma *= (float)Math.Pow(2.0, iter / (_options.NumberOfIterations / 10.0));
            }
        }

        /// <summary>
        /// Main LDSVM training routine.
        /// </summary>
        private LdSvmModelParameters TrainCore(IChannel ch, RoleMappedData trainingData, int numLeaf, int numFeatures)
        {
            int numNodes = 2 * numLeaf - 1;

            var w = new VBuffer<float>[numNodes];
            var thetaPrime = new VBuffer<float>[numNodes];
            var theta = new VBuffer<float>[numLeaf - 1];
            var biasW = new float[numNodes];
            var biasTheta = new float[numLeaf - 1];
            var biasThetaPrime = new float[numNodes];

            var tempW = new VBuffer<float>[numNodes];
            var tempThetaPrime = new VBuffer<float>[numNodes];
            var tempTheta = new VBuffer<float>[numLeaf - 1];
            var tempBiasW = new float[numNodes];
            var tempBiasTheta = new float[numLeaf - 1];
            var tempBiasThetaPrime = new float[numNodes];

            InitClassifierParam(numLeaf, numFeatures, tempW, w, theta, thetaPrime, biasW,
                biasTheta, biasThetaPrime, tempThetaPrime, tempTheta, tempBiasW, tempBiasTheta, tempBiasThetaPrime);

            var gamma = 0.01f;
            Data data = _options.Cache ?
                (Data)new CachedData(ch, trainingData) :
                new StreamingData(ch, trainingData);
            var pathWt = new float[numNodes];
            var localWt = new float[numNodes];
            var gradTheta = new float[numLeaf - 1];
            var wDotX = new float[numNodes];

            // Number of samples processed in each iteration
            int sampleSize = Math.Max(1, (int)Math.Sqrt(data.Length));
            for (int iter = 1; iter <= _options.NumberOfIterations; iter++)
            {
                // Update gamma adaptively
                if (iter % 100 == 1)
                    UpdateGamma(iter, numLeaf, ref gamma, data, theta, biasTheta);

                // Update learning rate
                float etaTW = (float)1.0 / (_options.LambdaW * (float)Math.Sqrt(iter + 1));
                float etaTTheta = (float)1.0 / (_options.LambdaTheta * (float)Math.Sqrt(iter + 1));
                float etaTThetaPrime = (float)1.0 / (_options.LambdaThetaprime * (float)Math.Sqrt(iter + 1));
                float coef = iter / (float)(iter + 1);

                // Update classifier parameters
                for (int i = 0; i < tempW.Length; ++i)
                    VectorUtils.ScaleBy(ref tempW[i], coef);
                for (int i = 0; i < tempTheta.Length; ++i)
                    VectorUtils.ScaleBy(ref tempTheta[i], coef);
                for (int i = 0; i < tempThetaPrime.Length; ++i)
                    VectorUtils.ScaleBy(ref tempThetaPrime[i], coef);

                for (int i = 0; i < numNodes; i++)
                {
                    tempBiasW[i] *= coef;
                    tempBiasThetaPrime[i] *= coef;
                }
                for (int i = 0; i < numLeaf - 1; i++)
                    tempBiasTheta[i] *= coef;

                var sample = data.SampleExamples(Host.Rand);
                foreach (var s in sample)
                {
                    float trueLabel = s.Label;
                    var features = s.Features;

                    // Compute path weight
                    for (int i = 0; i < numNodes; i++)
                        pathWt[i] = 1;
                    for (int i = 0; i < numLeaf - 1; i++)
                    {
                        var tanhDist = (float)Math.Tanh(gamma * (VectorUtils.DotProduct(in features, in theta[i]) + biasTheta[i]));
                        pathWt[2 * i + 1] = pathWt[i] * (1 + tanhDist) / (float)2.0;
                        pathWt[2 * i + 2] = pathWt[i] * (1 - tanhDist) / (float)2.0;
                    }

                    // Compute local weight
                    for (int l = 0; l < numNodes; l++)
                        localWt[l] = (float)Math.Tanh(_options.Sigma * (VectorUtils.DotProduct(in features, in thetaPrime[l]) + biasThetaPrime[l]));

                    // Make prediction
                    float yPredicted = 0;
                    for (int l = 0; l < numNodes; l++)
                    {
                        wDotX[l] = VectorUtils.DotProduct(in features, in w[l]) + biasW[l];
                        yPredicted += pathWt[l] * localWt[l] * wDotX[l];
                    }
                    float loss = 1 - trueLabel * yPredicted;

                    // If wrong prediction update classifier parameters
                    if (loss > 0)
                    {
                        // Compute gradient w.r.t current instance
                        ComputeGradTheta(in features, gradTheta, numLeaf, gamma, theta, biasTheta, pathWt, localWt, w, biasW);

                        // Check if bias is used ot not
                        int biasUpdateMult = _options.UseBias ? 1 : 0;

                        // Update W
                        for (int l = 0; l < numNodes; l++)
                        {
                            float tempGradW = trueLabel * etaTW / sampleSize * pathWt[l] * localWt[l];
                            VectorUtils.AddMult(in features, tempGradW, ref tempW[l]);
                            tempBiasW[l] += biasUpdateMult * tempGradW;
                        }

                        // Update ThetaPrime
                        for (int l = 0; l < numNodes; l++)
                        {
                            float tempGradThetaPrime = (1 - localWt[l] * localWt[l]) * trueLabel * etaTThetaPrime / sampleSize * pathWt[l] * wDotX[l];
                            VectorUtils.AddMult(in features, tempGradThetaPrime, ref tempThetaPrime[l]);
                            tempBiasThetaPrime[l] += biasUpdateMult * tempGradThetaPrime;
                        }

                        // Update Theta
                        for (int m = 0; m < numLeaf - 1; m++)
                        {
                            float tempGradTheta = trueLabel * etaTTheta / sampleSize * gradTheta[m];
                            VectorUtils.AddMult(in features, tempGradTheta, ref tempTheta[m]);
                            tempBiasTheta[m] += biasUpdateMult * tempGradTheta;
                        }
                    }
                }

                // Copy solution
                for (int i = 0; i < numNodes; i++)
                {
                    tempW[i].CopyTo(ref w[i]);
                    biasW[i] = tempBiasW[i];

                    tempThetaPrime[i].CopyTo(ref thetaPrime[i]);
                    biasThetaPrime[i] = tempBiasThetaPrime[i];
                }
                for (int i = 0; i < numLeaf - 1; i++)
                {
                    tempTheta[i].CopyTo(ref theta[i]);
                    biasTheta[i] = tempBiasTheta[i];
                }
            }

            return new LdSvmModelParameters(Host, w, thetaPrime, theta, _options.Sigma, biasW, biasTheta,
                biasThetaPrime, _options.TreeDepth);
        }

        /// <summary>
        /// Inititlize classifier parameters
        /// </summary>
        private void InitClassifierParam(int numLeaf, int numFeatures, VBuffer<float>[] tempW, VBuffer<float>[] w,
            VBuffer<float>[] theta, VBuffer<float>[] thetaPrime, float[] biasW, float[] biasTheta,
            float[] biasThetaPrime, VBuffer<float>[] tempThetaPrime, VBuffer<float>[] tempTheta,
            float[] tempBiasW, float[] tempBiasTheta, float[] tempBiasThetaPrime)
        {
            int count = 2 * numLeaf - 1;
            int half = numLeaf - 1;

            Host.Assert(Utils.Size(tempW) == count);
            Host.Assert(Utils.Size(w) == count);
            Host.Assert(Utils.Size(theta) == half);
            Host.Assert(Utils.Size(thetaPrime) == count);
            Host.Assert(Utils.Size(biasW) == count);
            Host.Assert(Utils.Size(biasTheta) == half);
            Host.Assert(Utils.Size(biasThetaPrime) == count);
            Host.Assert(Utils.Size(tempThetaPrime) == count);
            Host.Assert(Utils.Size(tempTheta) == half);
            Host.Assert(Utils.Size(tempBiasW) == count);
            Host.Assert(Utils.Size(tempBiasTheta) == half);
            Host.Assert(Utils.Size(tempBiasThetaPrime) == count);

            for (int i = 0; i < count; i++)
            {
                VBufferEditor<float> thetaInit = default;
                if (i < numLeaf - 1)
                    thetaInit = VBufferEditor.Create(ref theta[i], numFeatures);
                var wInit = VBufferEditor.Create(ref w[i], numFeatures);
                var thetaPrimeInit = VBufferEditor.Create(ref thetaPrime[i], numFeatures);
                for (int j = 0; j < numFeatures; j++)
                {
                    wInit.Values[j] = 2 * Host.Rand.NextSingle() - 1;
                    thetaPrimeInit.Values[j] = 2 * Host.Rand.NextSingle() - 1;
                    if (i < numLeaf - 1)
                        thetaInit.Values[j] = 2 * Host.Rand.NextSingle() - 1;
                }

                w[i] = wInit.Commit();
                w[i].CopyTo(ref tempW[i]);
                thetaPrime[i] = thetaPrimeInit.Commit();
                thetaPrime[i].CopyTo(ref tempThetaPrime[i]);

                if (_options.UseBias)
                {
                    float bW = 2 * Host.Rand.NextSingle() - 1;
                    biasW[i] = bW;
                    tempBiasW[i] = bW;
                    float bTP = 2 * Host.Rand.NextSingle() - 1;
                    biasThetaPrime[i] = bTP;
                    tempBiasThetaPrime[i] = bTP;
                }

                if (i >= half)
                    continue;

                theta[i] = thetaInit.Commit();
                theta[i].CopyTo(ref tempTheta[i]);

                if (_options.UseBias)
                {
                    float bT = 2 * Host.Rand.NextSingle() - 1;
                    biasTheta[i] = bT;
                    tempBiasTheta[i] = bT;
                }
            }
        }

        /// <summary>
        /// Initialization of model.
        /// </summary>
        private static void CheckOptions(IExceptionContext ectx, Options options)
        {
            ectx.AssertValue(options);

            ectx.CheckUserArg(options.TreeDepth >= 0, nameof(options.TreeDepth), "Tree depth can not be negative.");
            ectx.CheckUserArg(options.TreeDepth <= 24, nameof(options.TreeDepth), "Try running with a tree of smaller depth first and cross validate over other parameters.");
            ectx.CheckUserArg(options.LambdaW > 0, nameof(options.LambdaW), "Regularizer for W must be positive and non-zero.");
            ectx.CheckUserArg(options.LambdaTheta > 0, nameof(options.LambdaTheta), "Regularizer for Theta must be positive and non-zero.");
            ectx.CheckUserArg(options.LambdaThetaprime > 0, nameof(options.LambdaThetaprime), "Regularizer for Thetaprime must be positive and non-zero.");
        }

        internal struct LabelFeatures
        {
            public float Label;
            public VBuffer<float> Features;
        }

        private abstract class Data
        {
            protected readonly IChannel Ch;

            public abstract long Length { get; }

            protected Data(IChannel ch)
            {
                Ch = ch;
            }

            public abstract IEnumerable<VBuffer<float>> SampleForGammaUpdate(Random rand);
            public abstract IEnumerable<LabelFeatures> SampleExamples(Random rand);
        }

        private sealed class CachedData : Data
        {
            private readonly LabelFeatures[] _examples;
            private readonly int[] _indices;

            public override long Length => _examples.Length;

            public CachedData(IChannel ch, RoleMappedData data)
                : base(ch)
            {
                var examples = new List<LabelFeatures>();
                using (var cursor = new FloatLabelCursor(data, CursOpt.Label | CursOpt.Features))
                {
                    while (cursor.MoveNext())
                    {
                        var example = new LabelFeatures();
                        cursor.Features.CopyTo(ref example.Features);
                        example.Label = cursor.Label > 0 ? 1 : -1;
                        examples.Add(example);
                    }
                    Ch.Check(cursor.KeptRowCount > 0, NoTrainingInstancesMessage);
                    if (cursor.SkippedRowCount > 0)
                        Ch.Warning("Skipped {0} rows with missing feature/label values", cursor.SkippedRowCount);
                }
                _examples = examples.ToArray();
                _indices = Utils.GetIdentityPermutation((int)Length);
            }

            public override IEnumerable<LabelFeatures> SampleExamples(Random rand)
            {
                var sampleSize = Math.Max(1, (int)Math.Sqrt(Length));
                var length = (int)Length;
                // Select random subset of data - the first sampleSize indices will be
                // our subset.
                for (int k = 0; k < sampleSize; k++)
                {
                    int randIdx = k + rand.Next(length - k);
                    Utils.Swap(ref _indices[k], ref _indices[randIdx]);
                }

                for (int k = 0; k < sampleSize; k++)
                {
                    yield return _examples[_indices[k]];
                }
            }

            public override IEnumerable<VBuffer<float>> SampleForGammaUpdate(Random rand)
            {
                int length = (int)Length;
                for (int i = 0; i < NumberOfSamplesForGammaUpdate; i++)
                {
                    int index = rand.Next(length);
                    yield return _examples[index].Features;
                }
            }
        }

        private sealed class StreamingData : Data
        {
            private readonly RoleMappedData _data;
            private readonly int[] _indices;
            private readonly int[] _indices2;

            public override long Length { get; }

            public StreamingData(IChannel ch, RoleMappedData data)
                : base(ch)
            {
                Ch.AssertValue(data);

                _data = data;

                using (var cursor = _data.Data.GetRowCursor())
                {
                    while (cursor.MoveNext())
                        Length++;
                }
                _indices = Utils.GetIdentityPermutation((int)Length);
                _indices2 = new int[NumberOfSamplesForGammaUpdate];
            }

            public override IEnumerable<VBuffer<float>> SampleForGammaUpdate(Random rand)
            {
                int length = (int)Length;
                for (int i = 0; i < NumberOfSamplesForGammaUpdate; i++)
                {
                    _indices2[i] = rand.Next(length);
                }
                Array.Sort(_indices2);

                using (var cursor = _data.Data.GetRowCursor(_data.Data.Schema[_data.Schema.Feature.Value.Name]))
                {
                    var getter = cursor.GetGetter<VBuffer<float>>(_data.Data.Schema[_data.Schema.Feature.Value.Name]);
                    var features = default(VBuffer<float>);
                    int iIndex = 0;
                    while (cursor.MoveNext())
                    {
                        if (cursor.Position == _indices2[iIndex])
                        {
                            iIndex++;
                            getter(ref features);
                            var noNaNs = FloatUtils.IsFinite(features.GetValues());
                            if (noNaNs)
                                yield return features;
                            while (iIndex < NumberOfSamplesForGammaUpdate && cursor.Position == _indices2[iIndex])
                            {
                                iIndex++;
                                if (noNaNs)
                                    yield return features;
                            }
                            if (iIndex == NumberOfSamplesForGammaUpdate)
                                break;
                        }
                    }
                }
            }

            public override IEnumerable<LabelFeatures> SampleExamples(Random rand)
            {
                var sampleSize = Math.Max(1, (int)Math.Sqrt(Length));
                var length = (int)Length;
                // Select random subset of data - the first sampleSize indices will be
                // our subset.
                for (int k = 0; k < sampleSize; k++)
                {
                    int randIdx = k + rand.Next(length - k);
                    Utils.Swap(ref _indices[k], ref _indices[randIdx]);
                }

                Array.Sort(_indices, 0, sampleSize);

                var featureCol = _data.Data.Schema[_data.Schema.Feature.Value.Name];
                var labelCol = _data.Data.Schema[_data.Schema.Label.Value.Name];
                using (var cursor = _data.Data.GetRowCursor(featureCol, labelCol))
                {
                    var featureGetter = cursor.GetGetter<VBuffer<float>>(featureCol);
                    var labelGetter = RowCursorUtils.GetLabelGetter(cursor, labelCol.Index);
                    ValueGetter<LabelFeatures> getter =
                        (ref LabelFeatures dst) =>
                        {
                            featureGetter(ref dst.Features);
                            var label = default(float);
                            labelGetter(ref label);
                            dst.Label = label > 0 ? 1 : -1;
                        };

                    int iIndex = 0;
                    while (cursor.MoveNext())
                    {
                        if (cursor.Position == _indices[iIndex])
                        {
                            var example = new LabelFeatures();
                            getter(ref example);
                            iIndex++;
                            if (FloatUtils.IsFinite(example.Features.GetValues()))
                                yield return example;
                            if (iIndex == sampleSize)
                                break;
                        }
                    }
                }
            }
        }

        private protected override BinaryPredictionTransformer<LdSvmModelParameters> MakeTransformer(LdSvmModelParameters model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<LdSvmModelParameters>(Host, model, trainSchema, _options.FeatureColumnName);

        [TlcModule.EntryPoint(Name = "Trainers.LocalDeepSvmBinaryClassifier", Desc = Summary, UserName = UserNameValue, ShortName = LoadNameValue)]
        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLDSVM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LdSvmTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }
}
