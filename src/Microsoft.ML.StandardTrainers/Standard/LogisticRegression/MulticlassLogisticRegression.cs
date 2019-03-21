// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(LbfgsMaximumEntropyTrainer), typeof(LbfgsMaximumEntropyTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    LbfgsMaximumEntropyTrainer.UserNameValue,
    LbfgsMaximumEntropyTrainer.LoadNameValue,
    "MulticlassLogisticRegressionPredictorNew",
    LbfgsMaximumEntropyTrainer.ShortName,
    "multilr")]

[assembly: LoadableClass(typeof(MaximumEntropyModelParameters), null, typeof(SignatureLoadModel),
    "Multiclass LR Executor",
    MaximumEntropyModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(LbfgsMaximumEntropyTrainer), null, typeof(SignatureEntryPointModule), LbfgsMaximumEntropyTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers
{
    /// <include file = 'doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file = 'doc.xml' path='docs/members/example[@name="LogisticRegressionClassifier"]/*' />
    public sealed class LbfgsMaximumEntropyTrainer : LbfgsTrainerBase<LbfgsMaximumEntropyTrainer.Options,
        MulticlassPredictionTransformer<MaximumEntropyModelParameters>, MaximumEntropyModelParameters>
    {
        internal const string Summary = "Maximum entrypy classification is a method in statistics used to predict the probabilities of parallel events. The model predicts the probabilities of parallel events by fitting data to a softmax function.";
        internal const string LoadNameValue = "MultiClassLogisticRegression";
        internal const string UserNameValue = "Multi-class Logistic Regression";
        internal const string ShortName = "mlr";

        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// If set to <value>true</value> training statistics will be generated at the end of training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat, ShowTrainingStats", SortOrder = 50)]
            public bool ShowTrainingStatistics = false;
        }

        private int _numClasses;

        // The names for each label class, indexed by zero based class number.
        // These label names are used for model saving in place of class number
        // to make the model summary more user friendly. These names are populated
        // in the CheckLabel() method.
        // It could be null, if the label type is not a key type, or there is
        // missing label name for some class.
        private string[] _labelNames;

        // The prior distribution of data.
        // This array is of length equal to the number of classes.
        // After training, it stores the total weights of training examples in each class.
        private Double[] _prior;

        private ModelStatisticsBase _stats;

        private protected override int ClassCount => _numClasses;

        /// <summary>
        /// Initializes a new instance of <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        internal LbfgsMaximumEntropyTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = Options.Defaults.L1Regularization,
            float l2Weight = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int memorySize = Options.Defaults.HistorySize,
            bool enforceNoNegativity = Options.Defaults.EnforceNonNegativity)
            : base(env, featureColumn, TrainerUtils.MakeU4ScalarColumn(labelColumn), weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            ShowTrainingStats = LbfgsTrainerOptions.ShowTrainingStatistics;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        internal LbfgsMaximumEntropyTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeU4ScalarColumn(options.LabelColumnName))
        {
            ShowTrainingStats = LbfgsTrainerOptions.ShowTrainingStatistics;
        }

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            // REVIEW: For floating point labels, this will make a pass over the data.
            // Should we instead leverage the pass made by the LBFGS base class? Ideally, it wouldn't
            // make a pass over the data...
            data.CheckMulticlassLabel(out _numClasses);

            // Initialize prior counts.
            _prior = new Double[_numClasses];

            // Try to get the label key values metedata.
            var labelCol = data.Schema.Label.Value;
            var labelMetadataType = labelCol.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type;
            if (!(labelMetadataType is VectorType vecType && vecType.ItemType == TextDataViewType.Instance && vecType.Size == _numClasses))
            {
                _labelNames = null;
                return;
            }
            VBuffer<ReadOnlyMemory<char>> labelNames = default;
            labelCol.GetKeyValues(ref labelNames);

            // If label names is not dense or contain NA or default value, then it follows that
            // at least one class does not have a valid name for its label. If the label names we
            // try to get from the metadata are not unique, we may also not use them in model summary.
            // In both cases we set _labelNames to null and use the "Class_n", where n is the class number
            // for model summary saving instead.
            if (!labelNames.IsDense)
            {
                _labelNames = null;
                return;
            }

            _labelNames = new string[_numClasses];
            ReadOnlySpan<ReadOnlyMemory<char>> values = labelNames.GetValues();

            // This hashset is used to verify the uniqueness of label names.
            HashSet<string> labelNamesSet = new HashSet<string>();
            for (int i = 0; i < _numClasses; i++)
            {
                ReadOnlyMemory<char> value = values[i];
                if (value.IsEmpty)
                {
                    _labelNames = null;
                    break;
                }

                var vs = values[i].ToString();
                if (!labelNamesSet.Add(vs))
                {
                    _labelNames = null;
                    break;
                }

                _labelNames[i] = vs;

                Contracts.Assert(!string.IsNullOrEmpty(_labelNames[i]));
            }

            Contracts.Assert(_labelNames == null || _labelNames.Length == _numClasses);
        }

        //Override default termination criterion MeanRelativeImprovementCriterion with
        private protected override Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<float> init, out ITerminationCriterion terminationCriterion)
        {
            var opt = base.InitializeOptimizer(ch, cursorFactory, out init, out terminationCriterion);

            // MeanImprovementCriterion:
            //   Terminates when the geometrically-weighted average improvement falls below the tolerance
            terminationCriterion = new MeanImprovementCriterion(OptTol, 0.25f, MaxIterations);

            return opt;
        }

        private protected override float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
            in VBuffer<float> x, ref VBuffer<float> grad, ref float[] scores)
        {
            if (Utils.Size(scores) < _numClasses)
                scores = new float[_numClasses];

            float bias = 0;
            for (int c = 0, start = _numClasses; c < _numClasses; c++, start += NumFeatures)
            {
                x.GetItemOrDefault(c, ref bias);
                scores[c] = bias + VectorUtils.DotProductWithOffset(in x, start, in feat);
            }

            float logZ = MathUtils.SoftMax(scores.AsSpan(0, _numClasses));
            float datumLoss = logZ;

            int lab = (int)label;
            Contracts.Assert(0 <= lab && lab < _numClasses);
            for (int c = 0, start = _numClasses; c < _numClasses; c++, start += NumFeatures)
            {
                float probLabel = lab == c ? 1 : 0;
                datumLoss -= probLabel * scores[c];

                float modelProb = MathUtils.ExpSlow(scores[c] - logZ);
                float mult = weight * (modelProb - probLabel);
                VectorUtils.AddMultWithOffset(in feat, mult, ref grad, start);
                // Due to the call to EnsureBiases, we know this region is dense.
                var editor = VBufferEditor.CreateFromBuffer(ref grad);
                Contracts.Assert(editor.Values.Length >= BiasCount && (grad.IsDense || editor.Indices[BiasCount - 1] == BiasCount - 1));
                editor.Values[c] += mult;
            }

            Contracts.Check(FloatUtils.IsFinite(datumLoss), "Data contain bad values.");
            return weight * datumLoss;
        }

        private protected override VBuffer<float> InitializeWeightsFromPredictor(IPredictor srcPredictor)
        {
            var pred = srcPredictor as MaximumEntropyModelParameters;
            Contracts.AssertValue(pred);
            Contracts.Assert(pred.InputType.GetVectorSize() > 0);

            // REVIEW: Support initializing the weights of a superset of features.
            if (pred.InputType.GetVectorSize() != NumFeatures)
                throw Contracts.Except("The input training data must have the same features used to train the input predictor.");

            return InitializeWeights(pred.DenseWeightsEnumerable(), pred.GetBiases());
        }

        private protected override MaximumEntropyModelParameters CreatePredictor()
        {
            if (_numClasses < 1)
                throw Contracts.Except("Cannot create a multiclass predictor with {0} classes", _numClasses);
            if (_numClasses == 1)
            {
                using (var ch = Host.Start("Creating Predictor"))
                {
                    ch.Warning("Training resulted in a one class predictor");
                }
            }

            return new MaximumEntropyModelParameters(Host, in CurrentWeights, _numClasses, NumFeatures, _labelNames, _stats);
        }

        private protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, float loss, int numParams)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(cursorFactory);
            Contracts.Assert(NumGoodRows > 0);
            Contracts.Assert(WeightSum > 0);
            Contracts.Assert(BiasCount == _numClasses);
            Contracts.Assert(loss >= 0);
            Contracts.Assert(numParams >= BiasCount);
            Contracts.Assert(CurrentWeights.IsDense);

            ch.Info("Model trained with {0} training examples.", NumGoodRows);
            // Compute deviance: start with loss function.
            float deviance = (float)(2 * loss * WeightSum);

            if (L2Weight > 0)
            {
                // Need to subtract L2 regularization loss.
                // The bias term is not regularized.
                var regLoss = VectorUtils.NormSquared(CurrentWeights.GetValues().Slice(BiasCount)) * L2Weight;
                deviance -= regLoss;
            }

            if (L1Weight > 0)
            {
                // Need to subtract L1 regularization loss.
                // The bias term is not regularized.
                Double regLoss = 0;
                VBufferUtils.ForEachDefined(in CurrentWeights, (ind, value) => { if (ind >= BiasCount) regLoss += Math.Abs(value); });
                deviance -= (float)regLoss * L1Weight * 2;
            }

            ch.Info("Residual Deviance: \t{0}", deviance);

            // Compute null deviance, i.e., the deviance of null hypothesis.
            // Cap the prior positive rate at 1e-15.
            float nullDeviance = 0;
            for (int iLabel = 0; iLabel < _numClasses; iLabel++)
            {
                Contracts.Assert(_prior[iLabel] >= 0);
                if (_prior[iLabel] == 0)
                    continue;

                nullDeviance -= (float)(2 * _prior[iLabel] * Math.Log(_prior[iLabel] / WeightSum));
            }
            ch.Info("Null Deviance:    \t{0}", nullDeviance);

            // Compute AIC.
            ch.Info("AIC:              \t{0}", 2 * numParams + deviance);

            // REVIEW: Figure out how to compute the statistics for the coefficients.
            _stats = new ModelStatisticsBase(Host, NumGoodRows, numParams, deviance, nullDeviance);
        }

        private protected override void ProcessPriorDistribution(float label, float weight)
        {
            int iLabel = (int)label;
            Contracts.Assert(0 <= iLabel && iLabel < _numClasses);
            _prior[iLabel] += weight;
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
            };
        }

        private protected override MulticlassPredictionTransformer<MaximumEntropyModelParameters> MakeTransformer(MaximumEntropyModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<MaximumEntropyModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        /// <summary>
        /// Continues the training of a <see cref="LbfgsMaximumEntropyTrainer"/> using an already trained <paramref name="modelParameters"/> and returns
        /// a <see cref="MulticlassPredictionTransformer{MulticlassLogisticRegressionModelParameters}"/>.
        /// </summary>
        public MulticlassPredictionTransformer<MaximumEntropyModelParameters> Fit(IDataView trainData, MaximumEntropyModelParameters modelParameters)
            => TrainTransformer(trainData, initPredictor: modelParameters);

        [TlcModule.EntryPoint(Name = "Trainers.LogisticRegressionClassifier",
            Desc = LbfgsMaximumEntropyTrainer.Summary,
            UserName = LbfgsMaximumEntropyTrainer.UserNameValue,
            ShortName = LbfgsMaximumEntropyTrainer.ShortName)]
        internal static CommonOutputs.MulticlassClassificationOutput TrainMulticlass(IHostEnvironment env, LbfgsMaximumEntropyTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLRMultiClass");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LbfgsMaximumEntropyTrainer.Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new LbfgsMaximumEntropyTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }

    /// <summary>
    /// Common linear model of multiclass classifiers. <see cref="LinearMulticlassModelParameters"/> contains a single
    /// linear model per class.
    /// </summary>
    public abstract class LinearMulticlassModelParametersBase :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveSummary,
        ICanGetSummaryInKeyValuePairs,
        ICanGetSummaryAsIDataView,
        ICanGetSummaryAsIRow,
        ISingleCanSavePfa,
        ISingleCanSaveOnnx
    {
        private const string ModelStatsSubModelFilename = "ModelStats";
        private const string LabelNamesSubModelFilename = "LabelNames";
        private protected readonly int NumberOfClasses;
        private protected readonly int NumberOfFeatures;

        // The label names used to write model summary. Either null or of length _numClasses.
        private readonly string[] _labelNames;

        private protected readonly float[] Biases;
        private protected readonly VBuffer<float>[] Weights;
        public readonly ModelStatisticsBase Statistics;

        // This stores the _weights matrix in dense format for performance.
        // It is used to make efficient predictions when the instance is sparse, so we get
        // dense-sparse dot products and avoid the sparse-sparse case.
        // When the _weights matrix is dense to begin with, then _weights == _weightsDense at all times after construction.
        // When _weights is sparse, then this remains null until we see the first sparse instance,
        // at which point it is initialized.
        private volatile VBuffer<float>[] _weightsDense;

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;
        internal readonly DataViewType InputType;
        internal readonly DataViewType OutputType;
        DataViewType IValueMapper.InputType => InputType;
        DataViewType IValueMapper.OutputType => OutputType;

        bool ICanSavePfa.CanSavePfa => true;
        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        internal LinearMulticlassModelParametersBase(IHostEnvironment env, string name, in VBuffer<float> weights, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, name)
        {
            Contracts.Assert(weights.Length == numClasses + numClasses * numFeatures);
            NumberOfClasses = numClasses;
            NumberOfFeatures = numFeatures;

            // weights contains both bias and feature weights in a flat vector
            // Biases are stored in the first _numClass elements
            // followed by one weight vector for each class, in turn, all concatenated
            // (i.e.: in "row major", if we encode each weight vector as a row of a matrix)
            Contracts.Assert(weights.Length == NumberOfClasses + NumberOfClasses * NumberOfFeatures);

            Biases = new float[NumberOfClasses];
            for (int i = 0; i < Biases.Length; i++)
                weights.GetItemOrDefault(i, ref Biases[i]);
            Weights = new VBuffer<float>[NumberOfClasses];
            for (int i = 0; i < Weights.Length; i++)
                weights.CopyTo(ref Weights[i], NumberOfClasses + i * NumberOfFeatures, NumberOfFeatures);
            if (Weights.All(v => v.IsDense))
                _weightsDense = Weights;

            InputType = new VectorType(NumberDataViewType.Single, NumberOfFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, NumberOfClasses);

            Contracts.Assert(labelNames == null || labelNames.Length == numClasses);
            _labelNames = labelNames;

            Contracts.AssertValueOrNull(stats);
            Statistics = stats;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MaximumEntropyModelParameters"/> class.
        /// This constructor is called by <see cref="SdcaCalibratedMulticlassTrainer"/> to create the predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="name">Registration name of this model's actual type.</param>
        /// <param name="weights">The array of weights vectors. It should contain <paramref name="numClasses"/> weights.</param>
        /// <param name="bias">The array of biases. It should contain contain <paramref name="numClasses"/> weights.</param>
        /// <param name="numClasses">The number of classes for multi-class classification. Must be at least 2.</param>
        /// <param name="numFeatures">The length of the feature vector.</param>
        /// <param name="labelNames">The optional label names. If specified not null, it should have the same length as <paramref name="numClasses"/>.</param>
        /// <param name="stats">The model statistics.</param>
        internal LinearMulticlassModelParametersBase(IHostEnvironment env, string name, VBuffer<float>[] weights, float[] bias, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, name)
        {
            Contracts.CheckValue(weights, nameof(weights));
            Contracts.CheckValue(bias, nameof(bias));
            Contracts.CheckParam(numClasses >= 2, nameof(numClasses), "Must be at least 2.");
            NumberOfClasses = numClasses;
            Contracts.CheckParam(numFeatures >= 1, nameof(numFeatures), "Must be positive.");
            NumberOfFeatures = numFeatures;
            Contracts.Check(Utils.Size(weights) == NumberOfClasses);
            Contracts.Check(Utils.Size(bias) == NumberOfClasses);
            Weights = new VBuffer<float>[NumberOfClasses];
            Biases = new float[NumberOfClasses];
            for (int iClass = 0; iClass < NumberOfClasses; iClass++)
            {
                Contracts.Assert(weights[iClass].Length == NumberOfFeatures);
                weights[iClass].CopyTo(ref Weights[iClass]);
                Biases[iClass] = bias[iClass];
            }

            if (Weights.All(v => v.IsDense))
                _weightsDense = Weights;

            InputType = new VectorType(NumberDataViewType.Single, NumberOfFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, NumberOfClasses);

            Contracts.Assert(labelNames == null || labelNames.Length == numClasses);
            _labelNames = labelNames;

            Contracts.AssertValueOrNull(stats);
            Statistics = stats;
        }

        private protected LinearMulticlassModelParametersBase(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
            // *** Binary format ***
            // int: number of features
            // int: number of classes = number of biases
            // float[]: biases
            // (weight matrix, in CSR if sparse)
            // (see https://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000)
            // int: number of row start indices (_numClasses + 1 if sparse, 0 if dense)
            // int[]: row start indices
            // int: total number of column indices (0 if dense)
            // int[]: column index of each non-zero weight
            // int: total number of non-zero weights  (same as number of column indices if sparse, num of classes * num of features if dense)
            // float[]: non-zero weights
            // int[]: Id of label names (optional, in a separate stream)
            // ModelStatisticsBase: model statistics (optional, in a separate stream)

            NumberOfFeatures = ctx.Reader.ReadInt32();
            Host.CheckDecode(NumberOfFeatures >= 1);

            NumberOfClasses = ctx.Reader.ReadInt32();
            Host.CheckDecode(NumberOfClasses >= 1);

            Biases = ctx.Reader.ReadFloatArray(NumberOfClasses);

            int numStarts = ctx.Reader.ReadInt32();

            if (numStarts == 0)
            {
                // The weights are entirely dense.
                int numIndices = ctx.Reader.ReadInt32();
                Host.CheckDecode(numIndices == 0);
                int numWeights = ctx.Reader.ReadInt32();
                Host.CheckDecode(numWeights == NumberOfClasses * NumberOfFeatures);
                Weights = new VBuffer<float>[NumberOfClasses];
                for (int i = 0; i < Weights.Length; i++)
                {
                    var w = ctx.Reader.ReadFloatArray(NumberOfFeatures);
                    Weights[i] = new VBuffer<float>(NumberOfFeatures, w);
                }
                _weightsDense = Weights;
            }
            else
            {
                // Read weight matrix as CSR.
                Host.CheckDecode(numStarts == NumberOfClasses + 1);
                int[] starts = ctx.Reader.ReadIntArray(numStarts);
                Host.CheckDecode(starts[0] == 0);
                Host.CheckDecode(Utils.IsMonotonicallyIncreasing(starts));

                int numIndices = ctx.Reader.ReadInt32();
                Host.CheckDecode(numIndices == starts[starts.Length - 1]);

                var indices = new int[NumberOfClasses][];
                for (int i = 0; i < indices.Length; i++)
                {
                    indices[i] = ctx.Reader.ReadIntArray(starts[i + 1] - starts[i]);
                    Host.CheckDecode(Utils.IsIncreasing(0, indices[i], NumberOfFeatures));
                }

                int numValues = ctx.Reader.ReadInt32();
                Host.CheckDecode(numValues == numIndices);

                Weights = new VBuffer<float>[NumberOfClasses];
                for (int i = 0; i < Weights.Length; i++)
                {
                    float[] values = ctx.Reader.ReadFloatArray(starts[i + 1] - starts[i]);
                    Weights[i] = new VBuffer<float>(NumberOfFeatures, Utils.Size(values), values, indices[i]);
                }
            }
            WarnOnOldNormalizer(ctx, GetType(), Host);
            InputType = new VectorType(NumberDataViewType.Single, NumberOfFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, NumberOfClasses);

            // REVIEW: Should not save the label names duplicately with the predictor again.
            // Get it from the label column schema metadata instead.
            string[] labelNames = null;
            if (ctx.TryLoadBinaryStream(LabelNamesSubModelFilename, r => labelNames = LoadLabelNames(ctx, r)))
                _labelNames = labelNames;

            // backwards compatibility:MLR used to serialize a LinearModelSStatistics object, before there existed two separate classes
            // for ModelStatisticsBase and LinearModelParameterStatistics.
            // It always only populated only the fields now found on ModelStatisticsBase.
            ModelStatisticsBase stats;
            ctx.LoadModelOrNull<ModelStatisticsBase, SignatureLoadModel>(Host, out stats, ModelStatsSubModelFilename);
            Statistics = stats;
        }

        private protected abstract VersionInfo GetVersionInfo();

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(Biases.Length == NumberOfClasses);
            Host.Assert(Biases.Length == Weights.Length);
#if DEBUG
            foreach (var fw in Weights)
                Host.Assert(fw.Length == NumberOfFeatures);
#endif
            // *** Binary format ***
            // int: number of features
            // int: number of classes = number of biases
            // float[]: biases
            // (weight matrix, in CSR if sparse)
            // (see https://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000)
            // int: number of row start indices (_numClasses + 1 if sparse, 0 if dense)
            // int[]: row start indices
            // int: total number of column indices (0 if dense)
            // int[]: column index of each non-zero weight
            // int: total number of non-zero weights  (same as number of column indices if sparse, num of classes * num of features if dense)
            // float[]: non-zero weights
            // bool: whether label names are present
            // int[]: Id of label names (optional, in a separate stream)
            // LinearModelParameterStatistics: model statistics (optional, in a separate stream)

            ctx.Writer.Write(NumberOfFeatures);
            ctx.Writer.Write(NumberOfClasses);
            ctx.Writer.WriteSinglesNoCount(Biases.AsSpan(0, NumberOfClasses));
            // _weights == _weighsDense means we checked that all vectors in _weights
            // are actually dense, and so we assigned the same object, or it came dense
            // from deserialization.
            if (Weights == _weightsDense)
            {
                ctx.Writer.Write(0); // Number of starts.
                ctx.Writer.Write(0); // Number of indices.
                ctx.Writer.Write(NumberOfFeatures * Weights.Length);
                foreach (var fv in Weights)
                {
                    Host.Assert(fv.Length == NumberOfFeatures);
                    ctx.Writer.WriteSinglesNoCount(fv.GetValues());
                }
            }
            else
            {
                // Number of starts.
                ctx.Writer.Write(NumberOfClasses + 1);

                // Starts always starts with 0.
                int numIndices = 0;
                ctx.Writer.Write(numIndices);
                for (int i = 0; i < Weights.Length; i++)
                {
                    // REVIEW: Assuming the presence of *any* zero justifies
                    // writing in sparse format seems stupid, but might be difficult
                    // to change without changing the format since the presence of
                    // any sparse vector means we're writing indices anyway. Revisit.
                    // This is actually a bug waiting to happen: sparse/dense vectors
                    // can have different dot products even if they are logically the
                    // same vector.
                    numIndices += NonZeroCount(in Weights[i]);
                    ctx.Writer.Write(numIndices);
                }

                ctx.Writer.Write(numIndices);
                {
                    // just scoping the count so we can use another further down
                    int count = 0;
                    foreach (var fw in Weights)
                    {
                        var fwValues = fw.GetValues();
                        if (fw.IsDense)
                        {
                            for (int i = 0; i < fwValues.Length; i++)
                            {
                                if (fwValues[i] != 0)
                                {
                                    ctx.Writer.Write(i);
                                    count++;
                                }
                            }
                        }
                        else
                        {
                            var fwIndices = fw.GetIndices();
                            ctx.Writer.WriteIntsNoCount(fwIndices);
                            count += fwIndices.Length;
                        }
                    }
                    Host.Assert(count == numIndices);
                }

                ctx.Writer.Write(numIndices);

                {
                    int count = 0;
                    foreach (var fw in Weights)
                    {
                        var fwValues = fw.GetValues();
                        if (fw.IsDense)
                        {
                            for (int i = 0; i < fwValues.Length; i++)
                            {
                                if (fwValues[i] != 0)
                                {
                                    ctx.Writer.Write(fwValues[i]);
                                    count++;
                                }
                            }
                        }
                        else
                        {
                            ctx.Writer.WriteSinglesNoCount(fwValues);
                            count += fwValues.Length;
                        }
                    }
                    Host.Assert(count == numIndices);
                }
            }

            Contracts.AssertValueOrNull(_labelNames);
            if (_labelNames != null)
                ctx.SaveBinaryStream(LabelNamesSubModelFilename, w => SaveLabelNames(ctx, w));

            Contracts.AssertValueOrNull(Statistics);
            if (Statistics != null)
                ctx.SaveModel(Statistics, ModelStatsSubModelFilename);
        }

        // REVIEW: Destroy.
        private static int NonZeroCount(in VBuffer<float> vector)
        {
            int count = 0;
            var values = vector.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] != 0)
                    count++;
            }
            return count;
        }

        ValueMapper<TSrc, TDst> IValueMapper.GetMapper<TSrc, TDst>()
        {
            Host.Check(typeof(TSrc) == typeof(VBuffer<float>), "Invalid source type in GetMapper");
            Host.Check(typeof(TDst) == typeof(VBuffer<float>), "Invalid destination type in GetMapper");

            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    Host.Check(src.Length == NumberOfFeatures);

                    PredictCore(in src, ref dst);
                };
            return (ValueMapper<TSrc, TDst>)(Delegate)del;
        }

        private void PredictCore(in VBuffer<float> src, ref VBuffer<float> dst)
        {
            Host.Check(src.Length == NumberOfFeatures, "src length should equal the number of features");
            var weights = Weights;
            if (!src.IsDense)
                weights = DensifyWeights();

            var editor = VBufferEditor.Create(ref dst, NumberOfClasses);
            for (int i = 0; i < Biases.Length; i++)
                editor.Values[i] = Biases[i] + VectorUtils.DotProduct(in weights[i], in src);

            Calibrate(editor.Values);
            dst = editor.Commit();
        }

        private VBuffer<float>[] DensifyWeights()
        {
            if (_weightsDense == null)
            {
                lock (Weights)
                {
                    if (_weightsDense == null)
                    {
                        var weightsDense = new VBuffer<float>[NumberOfClasses];
                        for (int i = 0; i < Weights.Length; i++)
                        {
                            // Haven't yet created dense version of the weights.
                            // REVIEW: Should we always expand to full weights or should this be subject to an option?
                            var w = Weights[i];
                            if (w.IsDense)
                                weightsDense[i] = w;
                            else
                                w.CopyToDense(ref weightsDense[i]);
                        }
                        _weightsDense = weightsDense;
                    }
                }
                Host.AssertValue(_weightsDense);
            }
            return _weightsDense;
        }

        /// <summary>
        /// Post-processing function applied to scores of each class' linear model output.
        /// In <see cref="PredictCore(in VBuffer{float}, ref VBuffer{float})"/> we compute the i-th class' score
        /// by using inner product of the i-th linear coefficient vector <see cref="Weights"/>[i] and the input feature vector (plus bias).
        /// Then, <see cref="Calibrate(Span{float})"/> will be called to adjust those raw scores.
        /// </summary>
        private protected abstract void Calibrate(Span<float> dst);

        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValueOrNull(schema);

            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, NumberOfFeatures, ref names);
            for (int classNumber = 0; classNumber < Biases.Length; classNumber++)
            {
                results.Add(new KeyValuePair<string, object>(
                    string.Format("{0}+(Bias)", GetLabelName(classNumber)),
                    Biases[classNumber]
                    ));
            }

            for (int classNumber = 0; classNumber < Weights.Length; classNumber++)
            {
                var orderedWeights = Weights[classNumber].Items().OrderByDescending(kv => Math.Abs(kv.Value));
                foreach (var weight in orderedWeights)
                {
                    var value = weight.Value;
                    if (value == 0)
                        break;
                    int index = weight.Key;
                    var name = names.GetItemOrDefault(index);

                    results.Add(new KeyValuePair<string, object>(
                        string.Format("{0}+{1}", GetLabelName(classNumber), name.IsEmpty ? $"f{index}" : name.ToString()),
                        value
                    ));
                }
            }

            return results;
        }

        /// <summary>
        /// Actual implementation of <see cref="ICanSaveInTextFormat.SaveAsText(TextWriter, RoleMappedSchema)"/> should happen in derived classes.
        /// </summary>
        private void SaveAsTextCore(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine(GetTrainerName() + " bias and non-zero weights");

            foreach (var namedValues in ((ICanGetSummaryInKeyValuePairs)this).GetSummaryInKeyValuePairs(schema))
            {
                Host.Assert(namedValues.Value is float);
                writer.WriteLine("\t{0}\t{1}", namedValues.Key, (float)namedValues.Value);
            }

            if (Statistics != null)
                Statistics.SaveText(writer, schema.Feature.Value, 20);
        }

        private protected abstract string GetTrainerName();

        /// <summary>
        /// Redirect <see cref="ICanSaveInTextFormat.SaveAsText(TextWriter, RoleMappedSchema)"/> call to the right function.
        /// </summary>
        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema) => SaveAsTextCore(writer, schema);

        /// <summary>
        /// Summary is equivalent to its information in text format.
        /// </summary>
        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            ((ICanSaveInTextFormat)this).SaveAsText(writer, schema);
        }

        /// <summary>
        /// Actual implementation of <see cref="ICanSaveInSourceCode.SaveAsCode(TextWriter, RoleMappedSchema)"/> should happen in derived classes.
        /// </summary>
        private void SaveAsCodeCore(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValueOrNull(schema);

            writer.WriteLine(string.Format("var scores = new float[{0}];", NumberOfClasses));

            for (int i = 0; i < Biases.Length; i++)
            {
                LinearPredictorUtils.SaveAsCode(writer,
                    in Weights[i],
                    Biases[i],
                    schema,
                    "scores[" + i.ToString() + "]");
            }
        }

        /// <summary>
        /// The raw scores of all linear classifiers are stored in <see langword="float"/>[] <paramref name="scoresName"/>.
        /// Derived classes can use this functin to add C# code for post-transformation.
        /// </summary>
        private protected abstract void SavePostTransformAsCode(TextWriter writer, string scoresName);

        /// <summary>
        /// Redirect <see cref="ICanSaveInSourceCode.SaveAsCode(TextWriter, RoleMappedSchema)"/> call to the right function.
        /// </summary>
        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema) => SaveAsCodeCore(writer, schema);

        /// <summary>
        /// Actual implementation of <see cref="ISingleCanSavePfa.SaveAsPfa(BoundPfaContext, JToken)"/> should happen in derived classes.
        /// </summary>
        private JToken SaveAsPfaCore(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));

            const string typeName = "MCLinearPredictor";
            JToken typeDecl = typeName;
            if (ctx.Pfa.RegisterType(typeName))
            {
                JObject type = new JObject();
                type["type"] = "record";
                type["name"] = typeName;
                JArray fields = new JArray();
                JObject jobj = null;
                fields.Add(jobj.AddReturn("name", "coeff").AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Array(PfaUtils.Type.Double))));
                fields.Add(jobj.AddReturn("name", "const").AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Double)));
                type["fields"] = fields;
                typeDecl = type;
            }

            JObject predictor = new JObject();
            predictor["coeff"] = new JArray(Weights.Select(w => new JArray(w.DenseValues())));
            predictor["const"] = new JArray(Biases);
            var cell = ctx.DeclareCell("MCLinearPredictor", typeDecl, predictor);
            var cellRef = PfaUtils.Cell(cell);
            return ApplyPfaPostTransform(PfaUtils.Call("model.reg.linear", input, cellRef));
        }

        /// <summary>
        /// This is called at the end of <see cref="SaveAsPfaCore(BoundPfaContext, JToken)"/> to adjust the final outputs of all linear models.
        /// </summary>
        private protected abstract JToken ApplyPfaPostTransform(JToken input);

        /// <summary>
        /// Redirect <see cref="ISingleCanSavePfa.SaveAsPfa(BoundPfaContext, JToken)"/> call to the right function.
        /// </summary>
        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input) => SaveAsPfaCore(ctx, input);

        /// <summary>
        /// Actual implementation of <see cref="ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext, string[], string)"/> should happen in derived classes.
        /// It's ok to make <see cref="SaveAsOnnxCore(OnnxContext, string[], string)"/> a <see langword="private protected"/> method in the future
        /// if any derived class wants to override.
        /// </summary>
        private bool SaveAsOnnxCore(OnnxContext ctx, string[] outputs, string featureColumn)
        {
            Host.CheckValue(ctx, nameof(ctx));

            string opType = "LinearClassifier";
            var node = ctx.CreateNode(opType, new[] { featureColumn }, outputs, ctx.GetNodeName(opType));
            node.AddAttribute("post_transform", GetOnnxPostTransform());
            node.AddAttribute("multi_class", true);
            node.AddAttribute("coefficients", Weights.SelectMany(w => w.DenseValues()));
            node.AddAttribute("intercepts", Biases);
            node.AddAttribute("classlabels_ints", Enumerable.Range(0, NumberOfClasses).Select(x => (long)x));
            return true;
        }

        /// <summary>
        /// Post-transform applied to the raw scores produced by those linear models of all classes. For maximum entropy classification, it should be
        /// a softmax function. This function is used only in <see cref="SaveAsOnnxCore(OnnxContext, string[], string)"/>.
        /// </summary>
        private protected abstract string GetOnnxPostTransform();

        /// <summary>
        /// Redirect <see cref="ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext, string[], string)"/> call to the right function.
        /// </summary>
        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputs, string featureColumn) => SaveAsOnnxCore(ctx, outputs, featureColumn);

        /// <summary>
        /// Copies the weight vector for each class into a set of buffers.
        /// </summary>
        /// <param name="weights">A possibly reusable set of vectors, which will
        /// be expanded as necessary to accomodate the data.</param>
        /// <param name="numClasses">Set to the rank, which is also the logical length
        /// of <paramref name="weights"/>.</param>
        public void GetWeights(ref VBuffer<float>[] weights, out int numClasses)
        {
            numClasses = NumberOfClasses;
            Utils.EnsureSize(ref weights, NumberOfClasses, NumberOfClasses);
            for (int i = 0; i < NumberOfClasses; i++)
                Weights[i].CopyTo(ref weights[i]);
        }

        /// <summary>
        /// Gets the biases for the logistic regression predictor.
        /// </summary>
        public IEnumerable<float> GetBiases()
        {
            return Biases;
        }

        internal IEnumerable<float> DenseWeightsEnumerable()
        {
            Contracts.Assert(Weights.Length == Biases.Length);

            int featuresCount = Weights[0].Length;
            for (var i = 0; i < Weights.Length; i++)
            {
                Host.Assert(featuresCount == Weights[i].Length);
                foreach (var weight in Weights[i].Items(all: true))
                    yield return weight.Value;
            }
        }

        internal string GetLabelName(int classNumber)
        {
            const string classNumberFormat = "Class_{0}";
            Contracts.Assert(0 <= classNumber && classNumber < NumberOfClasses);
            return _labelNames == null ? string.Format(classNumberFormat, classNumber) : _labelNames[classNumber];
        }

        private string[] LoadLabelNames(ModelLoadContext ctx, BinaryReader reader)
        {
            Contracts.AssertValue(ctx);
            Contracts.AssertValue(reader);
            string[] labelNames = new string[NumberOfClasses];
            for (int i = 0; i < NumberOfClasses; i++)
            {
                int id = reader.ReadInt32();
                Host.CheckDecode(0 <= id && id < Utils.Size(ctx.Strings));
                var str = ctx.Strings[id];
                Host.CheckDecode(str.Length > 0);
                labelNames[i] = str;
            }

            return labelNames;
        }

        private void SaveLabelNames(ModelSaveContext ctx, BinaryWriter writer)
        {
            Contracts.AssertValue(ctx);
            Contracts.AssertValue(writer);
            Contracts.Assert(Utils.Size(_labelNames) == NumberOfClasses);
            for (int i = 0; i < NumberOfClasses; i++)
            {
                Host.AssertValue(_labelNames[i]);
                writer.Write(ctx.Strings.Add(_labelNames[i]).Id);
            }
        }

        IDataView ICanGetSummaryAsIDataView.GetSummaryDataView(RoleMappedSchema schema)
        {
            var bldr = new ArrayDataViewBuilder(Host);

            ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames =
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, NumberOfFeatures, ref dst);

            // Add the bias and the weight columns.
            bldr.AddColumn("Bias", NumberDataViewType.Single, Biases);
            bldr.AddColumn("Weights", getSlotNames, NumberDataViewType.Single, Weights);
            bldr.AddColumn("ClassNames", Enumerable.Range(0, NumberOfClasses).Select(i => GetLabelName(i)).ToArray());
            return bldr.GetDataView();
        }

        DataViewRow ICanGetSummaryAsIRow.GetSummaryIRowOrNull(RoleMappedSchema schema)
        {
            return null;
        }

        DataViewRow ICanGetSummaryAsIRow.GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            if (Statistics == null)
                return null;

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weights.Length, ref names);
            var meta = Statistics.MakeStatisticsMetadata(schema, in names);
            return AnnotationUtils.AnnotationsAsRow(meta);
        }
    }

    /// <summary>
    /// Linear model of multiclass classifiers. It outputs raw scores of all its linear models, and no probablistic output is provided.
    /// </summary>
    public sealed class LinearMulticlassModelParameters : LinearMulticlassModelParametersBase
    {
        internal const string LoaderSignature = "MulticlassLinear";
        internal const string RegistrationName = "MulticlassLinearPredictor";

        private static VersionInfo VersionInfo =>
            new VersionInfo(
                modelSignature: "MCLINEAR",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LinearMulticlassModelParameters).Assembly.FullName);

        /// <summary>
        /// Function used to pass <see cref="VersionInfo"/> into parent class. It may be used when saving the model.
        /// </summary>
        private protected override VersionInfo GetVersionInfo() => VersionInfo;

        internal LinearMulticlassModelParameters(IHostEnvironment env, in VBuffer<float> weights, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, RegistrationName, weights, numClasses, numFeatures, labelNames, stats)
        {
        }

        internal LinearMulticlassModelParameters(IHostEnvironment env, VBuffer<float>[] weights, float[] bias, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, RegistrationName, weights, bias, numClasses, numFeatures, labelNames, stats)
        {
        }

        private LinearMulticlassModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        /// <summary>
        /// This function does not do any calibration. It's common in multi-class support vector machines where probabilitic outputs are not provided.
        /// </summary>
        /// <param name="dst">Score vector should be calibrated.</param>
        private protected override void Calibrate(Span<float> dst)
        {
        }

        private static LinearMulticlassModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(VersionInfo);
            return new LinearMulticlassModelParameters(env, ctx);
        }

        private protected override void SavePostTransformAsCode(TextWriter writer, string scoresName) { }

        /// <summary>
        /// No post-transform is needed for non-clibrated classifier.
        /// </summary>
        private protected override string GetOnnxPostTransform() => "NONE";

        /// <summary>
        /// No post-transform is needed for non-clibrated classifier.
        /// </summary>
        private protected override JToken ApplyPfaPostTransform(JToken input) => input;

        private protected override string GetTrainerName() => nameof(LinearMulticlassModelParameters);
    }

    /// <summary>
    /// Linear maximum entropy model of multiclass classifiers. It outputs classes probabilities.
    /// This model is also known as multinomial logistic regression.
    /// Please see https://en.wikipedia.org/wiki/Multinomial_logistic_regression for details.
    /// </summary>
    public sealed class MaximumEntropyModelParameters : LinearMulticlassModelParametersBase
    {
        internal const string LoaderSignature = "MultiClassLRExec";
        internal const string RegistrationName = "MulticlassLogisticRegressionPredictor";

        private static VersionInfo VersionInfo =>
            new VersionInfo(
                modelSignature: "MULTI LR",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Added class names
                verWrittenCur: 0x00010003, // Added model stats
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MaximumEntropyModelParameters).Assembly.FullName);

        /// <summary>
        /// Function used to pass <see cref="VersionInfo"/> into parent class. It may be used when saving the model.
        /// </summary>
        private protected override VersionInfo GetVersionInfo() => VersionInfo;

        internal MaximumEntropyModelParameters(IHostEnvironment env, in VBuffer<float> weights, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, RegistrationName, weights, numClasses, numFeatures, labelNames, stats)
        {
        }

        internal MaximumEntropyModelParameters(IHostEnvironment env, VBuffer<float>[] weights, float[] bias, int numClasses, int numFeatures, string[] labelNames, ModelStatisticsBase stats = null)
            : base(env, RegistrationName, weights, bias, numClasses, numFeatures, labelNames, stats)
        {
        }

        private MaximumEntropyModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        /// <summary>
        /// This function applies softmax to <paramref name="dst"/>. For details about softmax, see https://en.wikipedia.org/wiki/Softmax_function.
        /// </summary>
        /// <param name="dst">Score vector should be calibrated.</param>
        private protected override void Calibrate(Span<float> dst)
        {
            Host.Assert(dst.Length == NumberOfClasses);

            // scores are in log-space; convert and fix underflow/overflow
            // TODO:   re-normalize probabilities to account for underflow/overflow?
            float softmax = MathUtils.SoftMax(dst.Slice(0, NumberOfClasses));
            for (int i = 0; i < NumberOfClasses; ++i)
                dst[i] = MathUtils.ExpSlow(dst[i] - softmax);
        }

        /// <summary>
        /// Apply softmax function to <paramref name="scoresName"/>, which contains raw scores from all linear models.
        /// </summary>
        private protected override void SavePostTransformAsCode(TextWriter writer, string scoresName)
        {
            writer.WriteLine(string.Format("var softmax = MathUtils.SoftMax({0}.AsSpan(0, {1}));", scoresName, NumberOfClasses));

            for (int c = 0; c < Biases.Length; c++)
                writer.WriteLine("{1}[{0}] = Math.Exp({1}[{0}] - softmax);", c, scoresName);
        }

        /// <summary>
        /// Apply softmax to the raw scores produced by the lienar models of all classes.
        /// </summary>
        private protected override string GetOnnxPostTransform() => "SOFTMAX";

        /// <summary>
        /// Apply softmax to the raw scores produced by the lienar models of all classes.
        /// </summary>
        private protected override JToken ApplyPfaPostTransform(JToken input) => PfaUtils.Call("m.link.softmax", input);

        private protected override string GetTrainerName() => nameof(LbfgsMaximumEntropyTrainer);
    }
}
