// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

[assembly: LoadableClass(Pkpd.Summary, typeof(Pkpd), typeof(Pkpd.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    Pkpd.UserNameValue, Pkpd.LoadNameValue, DocName = "trainer/OvaPkpd.md")]

[assembly: LoadableClass(typeof(PkpdPredictor), null, typeof(SignatureLoadModel),
    "PKPD Executor",
    PkpdPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{

    using TDistPredictor = IDistPredictorProducing<float, float>;
    using TScalarTrainer = ITrainerEstimator<IPredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;
    using CR = RoleMappedSchema.ColumnRole;
    using TTransformer = MulticlassPredictionTransformer<PkpdPredictor>;

    /// <summary>
    /// In this strategy, a binary classification algorithm is trained on each pair of classes.
    /// The pairs are unordered but created with replacement: so, if there were three classes, 0, 1,
    /// 2, we would train classifiers for the pairs (0,0), (0,1), (0,2), (1,1), (1,2),
    /// and(2,2). For each binary classifier, an input data point is considered a
    /// positive example if it is in either of the two classes in the pair, and a
    /// negative example otherwise. At prediction time, the probabilities for each
    /// pair of classes is considered as the probability of being in either class of
    /// the pair given the data, and the final predictive probabilities out of that
    /// per class are calculated given the probability that an example is in any given
    /// pair.
    ///
    /// These two can allow you to exploit trainers that do not naturally have a
    /// multiclass option, e.g., using the Runtime.FastTree.FastTreeBinaryClassificationTrainer
    /// to solve a multiclass problem.
    /// Alternately, it can allow ML.NET to solve a "simpler" problem even in the cases
    /// where the trainer has a multiclass option, but using it directly is not
    /// practical due to, usually, memory constraints.For example, while a multiclass
    /// logistic regression is a more principled way to solve a multiclass problem, it
    /// requires that the learner store a lot more intermediate state in the form of
    /// L-BFGS history for all classes *simultaneously*, rather than just one-by-one
    /// as would be needed for OVA.
    /// </summary>
    public sealed class Pkpd : MetaMulticlassTrainer<MulticlassPredictionTransformer<PkpdPredictor>, PkpdPredictor>
    {
        internal const string LoadNameValue = "PKPD";
        internal const string UserNameValue = "Pairwise coupling (PKPD)";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each pair of classes. "
            + "Prediction is then performed by running these binary classifiers, and computing a score for each class by counting how many of the binary "
            + "classifiers predicted it. The prediction is the class with the highest score.";

        /// <summary>
        /// Arguments passed to PKPD.
        /// </summary>
        public sealed class Arguments : ArgumentsBase
        {
        }
        /// <summary>
        /// Legacy constructor that builds the <see cref="Pkpd"/> trainer supplying the base trainer to use, for the classification task
        /// through the <see cref="Arguments"/>arguments.
        /// Developers should instantiate <see cref="Pkpd"/> by supplying the trainer argument directly to the <see cref="Pkpd"/> constructor
        /// using the other public constructor.
        /// </summary>
        public Pkpd(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Pkpd"/>
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> instance.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorCalibratorTrainer"/></param>
        /// <param name="labelColumn">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maxCalibrationExamples">Number of instances to train the calibrator.</param>
        public Pkpd(IHostEnvironment env, TScalarTrainer binaryEstimator, string labelColumn = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false, ICalibratorTrainer calibrator = null, int maxCalibrationExamples = 1000000000)
           : base(env,
               new Arguments
               {
                   ImputeMissingLabelsAsNegative = imputeMissingLabelsAsNegative,
                   MaxCalibrationExamples = maxCalibrationExamples,
               },
               LoadNameValue, labelColumn, binaryEstimator, calibrator)
        {
            Host.CheckValue(labelColumn, nameof(labelColumn), "Label column should not be null.");
        }

        protected override PkpdPredictor TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train M * (M+1) / 2 models arranged as a lower triangular matrix.
            var predModels = new TDistPredictor[count][];

            for (int i = 0; i < predModels.Length; i++)
            {
                predModels[i] = new TDistPredictor[i + 1];

                for (int j = 0; j <= i; j++)
                {
                    ch.Info($"Training learner ({i},{j})");
                    predModels[i][j] = TrainOne(ch, GetTrainer(), data, i, j).Model;
                }
            }

            return new PkpdPredictor(Host, predModels);
        }

        private IPredictionTransformer<TDistPredictor> TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls1, int cls2)
        {
            // this should not be necessary when the legacy constructor doesn't exist, and the label column is not an optional parameter on the
            // MetaMulticlassTrainer constructor.
            string trainerLabel = data.Schema.Label.Name;

            var view = MapLabels(data, cls1, cls2);
            var transformer = trainer.Fit(view);

            // the validations in the calibrator check for the feature column, in the RoleMappedData
            var trainedData = new RoleMappedData(view, label: trainerLabel, feature: transformer.FeatureColumn);

            var calibratedModel = transformer.Model as TDistPredictor;
            if (calibratedModel == null)
                calibratedModel = CalibratorUtils.TrainCalibrator(Host, ch, Calibrator, Args.MaxCalibrationExamples, transformer.Model, trainedData) as TDistPredictor;

            return new BinaryPredictionTransformer<TDistPredictor>(Host, calibratedModel, data.Data.Schema, transformer.FeatureColumn);
        }

        private IDataView MapLabels(RoleMappedData data, int cls1, int cls2)
        {
            var lab = data.Schema.Label;
            Host.Assert(!data.Schema.Schema.IsHidden(lab.Index));
            Host.Assert(lab.Type.KeyCount > 0 || lab.Type == NumberType.R4 || lab.Type == NumberType.R8);

            if (lab.Type.KeyCount > 0)
            {
                // Key values are 1-based.
                uint key1 = (uint)(cls1 + 1);
                uint key2 = (uint)(cls2 + 1);
                return MapLabelsCore(NumberType.U4, (ref uint val) => val == key1 || val == key2, data);
            }
            if (lab.Type == NumberType.R4)
            {
                float key1 = cls1;
                float key2 = cls2;
                return MapLabelsCore(NumberType.R4, (ref float val) => val == key1 || val == key2, data);
            }
            if (lab.Type == NumberType.R8)
            {
                double key1 = cls1;
                double key2 = cls2;
                return MapLabelsCore(NumberType.R8, (ref double val) => val == key1 || val == key2, data);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by PKPD: {lab.Type}");
        }

        /// <summary>
        /// Fits the data to the transformer
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>The trained predictor.</returns>
        public override TTransformer Fit(IDataView input)
        {
            string featureColumn = null;

            var roles = new KeyValuePair<CR, string>[1];
            roles[0] = new KeyValuePair<CR, string>(new CR(DefaultColumnNames.Label), LabelColumn.Name);
            var td = new RoleMappedData(input, roles);

            td.CheckMultiClassLabel(out var numClasses);
            // Train M * (M+1) / 2 models arranged as a lower triangular matrix.
            var predictors = new TDistPredictor[numClasses][];

            using (var ch = Host.Start("Fitting"))
            {
                for (int i = 0; i < predictors.Length; i++)
                {
                    predictors[i] = new TDistPredictor[i + 1];

                    for (int j = 0; j <= i; j++)
                    {
                        ch.Info($"Training learner ({i},{j})");

                        // need to capture the featureColum, and it is the same for all the transformers
                        if (i == 0 && j == 0)
                        {
                            var transformer = TrainOne(ch, GetTrainer(), td, i, j);
                            featureColumn = transformer.FeatureColumn;
                        }

                        predictors[i][j] = TrainOne(ch, GetTrainer(), td, i, j).Model;
                    }
                }
            }

            return new MulticlassPredictionTransformer<PkpdPredictor>(Host, new PkpdPredictor(Host, predictors), input.Schema, featureColumn, LabelColumn.Name);
        }
    }

    public sealed class PkpdPredictor :
        PredictorBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveModel
    {
        internal const string LoaderSignature = "PKPDExec";
        internal const string RegistrationName = "PKPDPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TLC PKPD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";
        private const string SubPredictorFmt2 = "SubPredictor_{0:000}_{1:000}";

        private readonly int _numClasses;
        // There are M * (M + 1) / 2 predictors. These are really organized as a lower triangular
        // matrix of predictors, with indices (0,0) (1,0) (1,1) (2,0) (2,1) (2,2), etc. We store them
        // in a 1-d array mostly to make it easy to parallelize.
        private readonly TDistPredictor[] _predictors;
        private readonly IValueMapperDist[] _mappers;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;
        public ColumnType InputType { get; }
        public ColumnType OutputType { get; }

        internal PkpdPredictor(IHostEnvironment env, TDistPredictor[][] predictors) :
            base(env, RegistrationName)
        {
            Host.Assert(Utils.Size(predictors) > 0);

            // We store the predictors in a 1-d array to facilitate parallelizing the Predict calls.
            _numClasses = predictors.Length;
            _predictors = new TDistPredictor[checked(_numClasses * (_numClasses + 1) / 2)];
            int index = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                Host.Assert(Utils.Size(predictors[i]) == i + 1);
                for (int j = 0; j <= i; j++)
                {
                    Host.Assert(index == GetIndex(i, j));
                    _predictors[index++] = predictors[i][j];
                }
            }
            Host.Assert(index == _predictors.Length);

            InputType = InitializeMappers(out _mappers);
            OutputType = new VectorType(NumberType.Float, _numClasses);
        }

        private PkpdPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // int: number of classes
            _numClasses = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numClasses > 0);

            long count = (long)_numClasses * (_numClasses + 1) / 2;
            Host.CheckDecode(count <= int.MaxValue);

            // Load the predictors.
            _predictors = new TDistPredictor[(int)count];
            int index = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    Host.Assert(index == GetIndex(i, j));
                    ctx.LoadModel<TDistPredictor, SignatureLoadModel>(Host, out _predictors[index++], string.Format(SubPredictorFmt2, i, j));
                }
                Host.Assert(index == GetIndex(i, i));
                ctx.LoadModel<TDistPredictor, SignatureLoadModel>(Host, out _predictors[index++], string.Format(SubPredictorFmt, i));
            }
            InputType = InitializeMappers(out _mappers);
            OutputType = new VectorType(NumberType.Float, _numClasses);
        }

        private ColumnType InitializeMappers(out IValueMapperDist[] mappers)
        {
            mappers = new IValueMapperDist[_predictors.Length];
            ColumnType inputType = null;
            for (int i = 0; i < _predictors.Length; i++)
            {
                var vmd = _predictors[i] as IValueMapperDist;
                Host.Check(IsValid(vmd, ref inputType), "Predictor doesn't implement the expected interface");
                mappers[i] = vmd;
            }
            return inputType;
        }

        private bool IsValid(IValueMapperDist mapper, ref ColumnType inputType)
        {
            if (mapper == null)
                return false;
            if (!mapper.InputType.IsKnownSizeVector || mapper.InputType.ItemType != NumberType.Float)
                return false;
            if (inputType == null)
                inputType = mapper.InputType;
            else if (inputType.VectorSize != mapper.InputType.VectorSize)
                return false;
            if (mapper.OutputType != NumberType.Float)
                return false;
            if (mapper.DistType != NumberType.Float)
                return false;
            return true;
        }

        public static PkpdPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PkpdPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of classes
            ctx.Writer.Write(_numClasses);

            // Save other streams.
            for (int i = 0; i < _numClasses; i++)
            {
                int index = GetIndex(i, 0);
                ctx.SaveModel(_predictors[index + i], string.Format(SubPredictorFmt, i));
                for (int j = 0; j < i; j++)
                    ctx.SaveModel(_predictors[index + j], string.Format(SubPredictorFmt2, i, j));
            }
        }

        private void ComputeProbabilities(Double[] buffer, ref float[] output)
        {
            // Compute the probabilities and store them in the beginning of buffer. Note that this is safe to do since
            // once we've computed the ith probability, we are totally done with the ith row and all previous rows
            // (in the lower triangular matrix of pairwise probabilities).
            Double sum = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                var value = buffer[i] = Pi(i, buffer);
                Host.Assert(0 <= value && value <= 1);
                sum += value;
            }

            if (Utils.Size(output) < _numClasses)
                output = new float[_numClasses];

            // Normalize.
            if (sum <= 0)
                sum = 1;

            for (int i = 0; i < _numClasses; i++)
                output[i] = (float)(buffer[i] / sum);
        }

        // Reconcile the predictions - ensure that pij >= pii and pji >= pii (when pii > 0).
        private void ReconcilePredictions(Double[] buffer)
        {
            for (int i = 0; i < _numClasses; i++)
            {
                int index = GetIndex(i, 0);
                var pii = buffer[index + i];
                if (!(pii > 0))
                    continue;

                for (int j = 0; j < i; j++)
                {
                    if (!(buffer[index + j] >= pii))
                        buffer[index + j] = pii;
                }
                index += i;
                for (int j = i + 1; j < _numClasses; j++)
                {
                    index += j; // Move past previous row.
                    Host.Assert(index == GetIndex(j, i));
                    if (!(buffer[index] >= pii))
                        buffer[index] = pii;
                }
            }
        }

        private Double Pi(int i, Double[] values)
        {
            // values is the lower triangular matrix of pairwise probabilities pij = P(y=i or y=j | x).
            // Get pii = P(y=i | x)
            int index = GetIndex(i, 0);
            Double pii = values[index + i];

            if (!(pii > 0))
                return 0;

            // Compute sum { pij | j != i }
            Double sum = 0;
            for (int j = 0; j < i; j++)
            {
                Host.Assert(values[index + j] >= pii);
                sum += values[index + j];
            }
            index += i;
            for (int j = i + 1; j < _numClasses; j++)
            {
                index += j; // Move past previous row.
                Host.Assert(index == GetIndex(j, i));
                Host.Assert(values[index] >= pii);
                sum += values[index];
            }

            // Return pii / (sum - (k - 2) * pii).
            // Taking max simply protects against round-off error - in exact math it wouldn't be needed.
            return pii / Math.Max(pii, sum - (_numClasses - 2) * pii);
        }

        private int GetIndex(int i, int j)
        {
            Host.Assert(0 <= j && j <= i && i < _numClasses);
            return i * (i + 1) / 2 + j;
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            var maps = new ValueMapper<VBuffer<float>, float, float>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
                maps[i] = _mappers[i].GetMapper<VBuffer<float>, float, float>();

            var buffer = new Double[_numClasses];
            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (ref VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    if (InputType.VectorSize > 0)
                        Host.Check(src.Length == InputType.VectorSize);

                    var values = dst.Values;
                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        float score = 0;
                        float prob = 0;
                        maps[i](ref tmp, ref score, ref prob);
                        buffer[i] = prob;
                    });

                    ReconcilePredictions(buffer);
                    ComputeProbabilities(buffer, ref values);

                    dst = new VBuffer<float>(_numClasses, values, dst.Indices);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}