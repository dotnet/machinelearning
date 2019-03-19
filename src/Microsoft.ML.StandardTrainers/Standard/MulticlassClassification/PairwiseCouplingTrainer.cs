// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(PairwiseCouplingTrainer.Summary, typeof(PairwiseCouplingTrainer), typeof(PairwiseCouplingTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    PairwiseCouplingTrainer.UserNameValue, PairwiseCouplingTrainer.LoadNameValue, DocName = "trainer/OvaPkpd.md")]

[assembly: LoadableClass(typeof(PairwiseCouplingModelParameters), null, typeof(SignatureLoadModel),
    "PKPD Executor",
    PairwiseCouplingModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    using CR = RoleMappedSchema.ColumnRole;
    using TDistPredictor = IDistPredictorProducing<float, float>;
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;
    using TTransformer = MulticlassPredictionTransformer<PairwiseCouplingModelParameters>;

    /// <summary>
    /// In this strategy, a binary classification algorithm is trained on each pair of classes.
    /// The pairs are unordered but created with replacement: so, if there were three classes, 0, 1,
    /// 2, we would train classifiers for the pairs (0,0), (0,1), (0,2), (1,1), (1,2),
    /// and (2,2). For each binary classifier, an input data point is considered a
    /// positive example if it is in either of the two classes in the pair, and a
    /// negative example otherwise. At prediction time, the probabilities for each
    /// pair of classes is considered as the probability of being in either class of
    /// the pair given the data, and the final predictive probabilities out of that
    /// per class are calculated given the probability that an example is in any given
    /// pair.
    ///
    /// These two can allow you to exploit trainers that do not naturally have a
    /// multiclass option, for example, using the FastTree Binary Classification
    /// to solve a multiclass problem.
    /// Alternately, it can allow ML.NET to solve a "simpler" problem even in the cases
    /// where the trainer has a multiclass option, but using it directly is not
    /// practical due to, usually, memory constraints. For example, while a multiclass
    /// logistic regression is a more principled way to solve a multiclass problem, it
    /// requires that the learner store a lot more intermediate state in the form of
    /// L-BFGS history for all classes *simultaneously*, rather than just one-by-one
    /// as would be needed for a one-versus-all classification model.
    /// </summary>
    public sealed class PairwiseCouplingTrainer : MetaMulticlassClassificationTrainer<MulticlassPredictionTransformer<PairwiseCouplingModelParameters>, PairwiseCouplingModelParameters>
    {
        internal const string LoadNameValue = "PKPD";
        internal const string UserNameValue = "Pairwise coupling (PKPD)";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each pair of classes. "
            + "Prediction is then performed by running these binary classifiers, and computing a score for each class by counting how many of the binary "
            + "classifiers predicted it. The prediction is the class with the highest score.";

        /// <summary>
        /// Options passed to <see cref="Microsoft.ML.Trainers.PairwiseCouplingTrainer"/>.
        /// </summary>
        internal sealed class Options : OptionsBase
        {
        }

        /// <summary>
        /// Constructs a <see cref="PairwiseCouplingTrainer"/> trainer supplying the base trainer to use, for the classification task
        /// through the <see cref="Options"/>Options.
        /// </summary>
        internal PairwiseCouplingTrainer(IHostEnvironment env, Options options)
            : base(env, options, LoadNameValue)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PairwiseCouplingTrainer"/>
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> instance.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="labelColumnName">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="calibrator">The calibrator to use for each model instance. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="maximumCalibrationExampleCount">Number of instances to train the calibrator.</param>
        internal PairwiseCouplingTrainer(IHostEnvironment env,
            TScalarTrainer binaryEstimator,
            string labelColumnName = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            ICalibratorTrainer calibrator = null,
            int maximumCalibrationExampleCount = 1000000000)
           : base(env,
               new Options
               {
                   ImputeMissingLabelsAsNegative = imputeMissingLabelsAsNegative,
                   MaxCalibrationExamples = maximumCalibrationExampleCount,
               },
               LoadNameValue, labelColumnName, binaryEstimator, calibrator)
        {
            Host.CheckValue(labelColumnName, nameof(labelColumnName), "Label column should not be null.");
        }

        private protected override PairwiseCouplingModelParameters TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train M * (M+1) / 2 models arranged as a lower triangular matrix.
            var predModels = new TDistPredictor[count][];

            for (int i = 0; i < predModels.Length; i++)
            {
                predModels[i] = new TDistPredictor[i + 1];

                for (int j = 0; j <= i; j++)
                {
                    ch.Info($"Training learner ({i},{j})");
                    predModels[i][j] = TrainOne(ch, Trainer, data, i, j).Model;
                }
            }

            return new PairwiseCouplingModelParameters(Host, predModels);
        }

        private ISingleFeaturePredictionTransformer<TDistPredictor> TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls1, int cls2)
        {
            // this should not be necessary when the legacy constructor doesn't exist, and the label column is not an optional parameter on the
            // MetaMulticlassTrainer constructor.
            string trainerLabel = data.Schema.Label.Value.Name;

            var view = MapLabels(data, cls1, cls2);
            var transformer = trainer.Fit(view);

            // the validations in the calibrator check for the feature column, in the RoleMappedData
            var trainedData = new RoleMappedData(view, label: trainerLabel, feature: transformer.FeatureColumnName);

            var calibratedModel = transformer.Model as TDistPredictor;
            if (calibratedModel == null)
                calibratedModel = CalibratorUtils.GetCalibratedPredictor(Host, ch, Calibrator, transformer.Model, trainedData, Args.MaxCalibrationExamples) as TDistPredictor;

            return new BinaryPredictionTransformer<TDistPredictor>(Host, calibratedModel, trainedData.Data.Schema, transformer.FeatureColumnName);
        }

        private IDataView MapLabels(RoleMappedData data, int cls1, int cls2)
        {
            var label = data.Schema.Label.Value;
            Host.Assert(!label.IsHidden);
            Host.Assert(label.Type.GetKeyCount() > 0 || label.Type == NumberDataViewType.Single || label.Type == NumberDataViewType.Double);

            if (label.Type.GetKeyCount() > 0)
            {
                // Key values are 1-based.
                uint key1 = (uint)(cls1 + 1);
                uint key2 = (uint)(cls2 + 1);
                return MapLabelsCore(NumberDataViewType.UInt32, (in uint val) => val == key1 || val == key2, data);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by nameof(PairwiseCouplingTrainer): {label.Type.RawType}");
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

            td.CheckMulticlassLabel(out var numClasses);
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
                            var transformer = TrainOne(ch, Trainer, td, i, j);
                            featureColumn = transformer.FeatureColumnName;
                        }

                        predictors[i][j] = TrainOne(ch, Trainer, td, i, j).Model;
                    }
                }
            }

            return new MulticlassPredictionTransformer<PairwiseCouplingModelParameters>(Host, new PairwiseCouplingModelParameters(Host, predictors), input.Schema, featureColumn, LabelColumn.Name);
        }
    }

    /// <summary>
    /// Contains the model parameters and prediction functions for the PairwiseCouplingTrainer.
    /// </summary>
    public sealed class PairwiseCouplingModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PairwiseCouplingModelParameters).Assembly.FullName);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";
        private const string SubPredictorFmt2 = "SubPredictor_{0:000}_{1:000}";

        private readonly int _numClasses;
        // There are M * (M + 1) / 2 predictors. These are really organized as a lower triangular
        // matrix of predictors, with indices (0,0) (1,0) (1,1) (2,0) (2,1) (2,2), etc. We store them
        // in a 1-d array mostly to make it easy to parallelize.
        private readonly TDistPredictor[] _predictors;
        private readonly IValueMapperDist[] _mappers;

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;
        private readonly VectorType _inputType;
        private readonly DataViewType _outputType;
        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => _outputType;

        internal PairwiseCouplingModelParameters(IHostEnvironment env, TDistPredictor[][] predictors) :
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

            _inputType = InitializeMappers(out _mappers);
            _outputType = new VectorType(NumberDataViewType.Single, _numClasses);
        }

        private PairwiseCouplingModelParameters(IHostEnvironment env, ModelLoadContext ctx)
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
            _inputType = InitializeMappers(out _mappers);
            _outputType = new VectorType(NumberDataViewType.Single, _numClasses);
        }

        private VectorType InitializeMappers(out IValueMapperDist[] mappers)
        {
            mappers = new IValueMapperDist[_predictors.Length];
            VectorType inputType = null;
            for (int i = 0; i < _predictors.Length; i++)
            {
                var vmd = _predictors[i] as IValueMapperDist;
                Host.Check(IsValid(vmd, ref inputType), "Predictor doesn't implement the expected interface");
                mappers[i] = vmd;
            }
            return inputType;
        }

        private bool IsValid(IValueMapperDist mapper, ref VectorType inputType)
        {
            if (mapper == null)
                return false;
            VectorType vectorType = mapper.InputType as VectorType;
            if (vectorType == null || !vectorType.IsKnownSize || vectorType.ItemType != NumberDataViewType.Single)
                return false;
            if (inputType == null)
                inputType = vectorType;
            else if (inputType.Size != vectorType.Size)
                return false;
            if (mapper.OutputType != NumberDataViewType.Single)
                return false;
            if (mapper.DistType != NumberDataViewType.Single)
                return false;
            return true;
        }

        private static PairwiseCouplingModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PairwiseCouplingModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
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

        private void ComputeProbabilities(double[] buffer, Span<float> output)
        {
            // Compute the probabilities and store them in the beginning of buffer. Note that this is safe to do since
            // once we've computed the ith probability, we are totally done with the ith row and all previous rows
            // (in the lower triangular matrix of pairwise probabilities).
            double sum = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                var value = buffer[i] = Pi(i, buffer);
                Host.Assert(0 <= value && value <= 1);
                sum += value;
            }

            Contracts.Assert(output.Length >= _numClasses);

            // Normalize.
            if (sum <= 0)
                sum = 1;

            for (int i = 0; i < _numClasses; i++)
                output[i] = (float)(buffer[i] / sum);
        }

        // Reconcile the predictions - ensure that pij >= pii and pji >= pii (when pii > 0).
        private void ReconcilePredictions(double[] buffer)
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

        private double Pi(int i, double[] values)
        {
            // values is the lower triangular matrix of pairwise probabilities pij = P(y=i or y=j | x).
            // Get pii = P(y=i | x)
            int index = GetIndex(i, 0);
            double pii = values[index + i];

            if (!(pii > 0))
                return 0;

            // Compute sum { pij | j != i }
            double sum = 0;
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

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            var maps = new ValueMapper<VBuffer<float>, float, float>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
                maps[i] = _mappers[i].GetMapper<VBuffer<float>, float, float>();
            var parallelOptions = new ParallelOptions();
            var buffer = new double[_mappers.Length];
            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    if (_inputType.Size > 0)
                        Host.Check(src.Length == _inputType.Size);

                    var tmp = src;
                    Parallel.For(0, maps.Length, parallelOptions, i =>
                    {
                        float score = 0;
                        float prob = 0;
                        maps[i](in tmp, ref score, ref prob);
                        buffer[i] = prob;
                    });

                    ReconcilePredictions(buffer);

                    var editor = VBufferEditor.Create(ref dst, _numClasses);
                    ComputeProbabilities(buffer, editor.Values);
                    dst = editor.Commit();
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}
