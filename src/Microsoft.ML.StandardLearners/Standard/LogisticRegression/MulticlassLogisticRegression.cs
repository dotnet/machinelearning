// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(MulticlassLogisticRegression), typeof(MulticlassLogisticRegression.Options),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MulticlassLogisticRegression.UserNameValue,
    MulticlassLogisticRegression.LoadNameValue,
    "MulticlassLogisticRegressionPredictorNew",
    MulticlassLogisticRegression.ShortName,
    "multilr")]

[assembly: LoadableClass(typeof(MulticlassLogisticRegressionModelParameters), null, typeof(SignatureLoadModel),
    "Multiclass LR Executor",
    MulticlassLogisticRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    /// <include file = 'doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file = 'doc.xml' path='docs/members/example[@name="LogisticRegressionClassifier"]/*' />
    public sealed class MulticlassLogisticRegression : LbfgsTrainerBase<MulticlassLogisticRegression.Options,
        MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>, MulticlassLogisticRegressionModelParameters>
    {
        internal const string LoadNameValue = "MultiClassLogisticRegression";
        internal const string UserNameValue = "Multi-class Logistic Regression";
        internal const string ShortName = "mlr";

        public sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat", SortOrder = 50)]
            public bool ShowTrainingStats = false;
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

        private LinearModelStatistics _stats;

        protected override int ClassCount => _numClasses;

        /// <summary>
        /// Initializes a new instance of <see cref="MulticlassLogisticRegression"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegression"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        internal MulticlassLogisticRegression(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = Options.Defaults.L1Weight,
            float l2Weight = Options.Defaults.L2Weight,
            float optimizationTolerance = Options.Defaults.OptTol,
            int memorySize = Options.Defaults.MemorySize,
            bool enforceNoNegativity = Options.Defaults.EnforceNonNegativity)
            : base(env, featureColumn, TrainerUtils.MakeU4ScalarColumn(labelColumn), weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            ShowTrainingStats = Args.ShowTrainingStats;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="MulticlassLogisticRegression"/>
        /// </summary>
        internal MulticlassLogisticRegression(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeU4ScalarColumn(options.LabelColumn))
        {
            ShowTrainingStats = Args.ShowTrainingStats;
        }

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            // REVIEW: For floating point labels, this will make a pass over the data.
            // Should we instead leverage the pass made by the LBFGS base class? Ideally, it wouldn't
            // make a pass over the data...
            data.CheckMultiClassLabel(out _numClasses);

            // Initialize prior counts.
            _prior = new Double[_numClasses];

            // Try to get the label key values metedata.
            var labelCol = data.Schema.Label.Value;
            var labelMetadataType = labelCol.Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.KeyValues)?.Type;
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

        protected override float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
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

        protected override VBuffer<float> InitializeWeightsFromPredictor(MulticlassLogisticRegressionModelParameters srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);
            Contracts.Assert(srcPredictor.InputType.GetVectorSize() > 0);

            // REVIEW: Support initializing the weights of a superset of features.
            if (srcPredictor.InputType.GetVectorSize() != NumFeatures)
                throw Contracts.Except("The input training data must have the same features used to train the input predictor.");

            return InitializeWeights(srcPredictor.DenseWeightsEnumerable(), srcPredictor.GetBiases());
        }

        protected override MulticlassLogisticRegressionModelParameters CreatePredictor()
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

            return new MulticlassLogisticRegressionModelParameters(Host, in CurrentWeights, _numClasses, NumFeatures, _labelNames, _stats);
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
            _stats = new LinearModelStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance);
        }

        protected override void ProcessPriorDistribution(float label, float weight)
        {
            int iLabel = (int)label;
            Contracts.Assert(0 <= iLabel && iLabel < _numClasses);
            _prior[iLabel] += weight;
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Metadata.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                .Concat(MetadataUtils.GetTrainerOutputMetadata()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
            };
        }

        protected override MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters> MakeTransformer(MulticlassLogisticRegressionModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        public MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters> Train(IDataView trainData, IPredictor initialPredictor = null)
            => TrainTransformer(trainData, initPredictor: initialPredictor);
    }

    public sealed class MulticlassLogisticRegressionModelParameters :
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
        internal const string LoaderSignature = "MultiClassLRExec";
        internal const string RegistrationName = "MulticlassLogisticRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULTI LR",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Added class names
                verWrittenCur: 0x00010003, // Added model stats
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MulticlassLogisticRegressionModelParameters).Assembly.FullName);
        }

        private const string ModelStatsSubModelFilename = "ModelStats";
        private const string LabelNamesSubModelFilename = "LabelNames";
        private readonly int _numClasses;
        private readonly int _numFeatures;

        // The label names used to write model summary. Either null or of length _numClasses.
        private readonly string[] _labelNames;

        private readonly float[] _biases;
        private readonly VBuffer<float>[] _weights;
        private readonly LinearModelStatistics _stats;

        // This stores the _weights matrix in dense format for performance.
        // It is used to make efficient predictions when the instance is sparse, so we get
        // dense-sparse dot products and avoid the sparse-sparse case.
        // When the _weights matrix is dense to begin with, then _weights == _weightsDense at all times after construction.
        // When _weights is sparse, then this remains null until we see the first sparse instance,
        // at which point it is initialized.
        private volatile VBuffer<float>[] _weightsDense;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;
        internal readonly DataViewType InputType;
        internal readonly DataViewType OutputType;
        DataViewType IValueMapper.InputType => InputType;
        DataViewType IValueMapper.OutputType => OutputType;

        bool ICanSavePfa.CanSavePfa => true;
        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        internal MulticlassLogisticRegressionModelParameters(IHostEnvironment env, in VBuffer<float> weights, int numClasses, int numFeatures, string[] labelNames, LinearModelStatistics stats = null)
            : base(env, RegistrationName)
        {
            Contracts.Assert(weights.Length == numClasses + numClasses * numFeatures);
            _numClasses = numClasses;
            _numFeatures = numFeatures;

            // weights contains both bias and feature weights in a flat vector
            // Biases are stored in the first _numClass elements
            // followed by one weight vector for each class, in turn, all concatenated
            // (i.e.: in "row major", if we encode each weight vector as a row of a matrix)
            Contracts.Assert(weights.Length == _numClasses + _numClasses * _numFeatures);

            _biases = new float[_numClasses];
            for (int i = 0; i < _biases.Length; i++)
                weights.GetItemOrDefault(i, ref _biases[i]);
            _weights = new VBuffer<float>[_numClasses];
            for (int i = 0; i < _weights.Length; i++)
                weights.CopyTo(ref _weights[i], _numClasses + i * _numFeatures, _numFeatures);
            if (_weights.All(v => v.IsDense))
                _weightsDense = _weights;

            InputType = new VectorType(NumberDataViewType.Single, _numFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, _numClasses);

            Contracts.Assert(labelNames == null || labelNames.Length == numClasses);
            _labelNames = labelNames;

            Contracts.AssertValueOrNull(stats);
            _stats = stats;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassLogisticRegressionModelParameters"/> class.
        /// This constructor is called by <see cref="SdcaMultiClassTrainer"/> to create the predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="weights">The array of weights vectors. It should contain <paramref name="numClasses"/> weights.</param>
        /// <param name="bias">The array of biases. It should contain contain <paramref name="numClasses"/> weights.</param>
        /// <param name="numClasses">The number of classes for multi-class classification. Must be at least 2.</param>
        /// <param name="numFeatures">The length of the feature vector.</param>
        /// <param name="labelNames">The optional label names. If specified not null, it should have the same length as <paramref name="numClasses"/>.</param>
        /// <param name="stats">The model statistics.</param>
        public MulticlassLogisticRegressionModelParameters(IHostEnvironment env, VBuffer<float>[] weights, float[] bias, int numClasses, int numFeatures, string[] labelNames, LinearModelStatistics stats = null)
            : base(env, RegistrationName)
        {
            Contracts.CheckValue(weights, nameof(weights));
            Contracts.CheckValue(bias, nameof(bias));
            Contracts.CheckParam(numClasses >= 2, nameof(numClasses), "Must be at least 2.");
            _numClasses = numClasses;
            Contracts.CheckParam(numFeatures >= 1, nameof(numFeatures), "Must be positive.");
            _numFeatures = numFeatures;
            Contracts.Check(Utils.Size(weights) == _numClasses);
            Contracts.Check(Utils.Size(bias) == _numClasses);
            _weights = new VBuffer<float>[_numClasses];
            _biases = new float[_numClasses];
            for (int iClass = 0; iClass < _numClasses; iClass++)
            {
                Contracts.Assert(weights[iClass].Length == _numFeatures);
                weights[iClass].CopyTo(ref _weights[iClass]);
                _biases[iClass] = bias[iClass];
            }

            if (_weights.All(v => v.IsDense))
                _weightsDense = _weights;

            InputType = new VectorType(NumberDataViewType.Single, _numFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, _numClasses);

            Contracts.Assert(labelNames == null || labelNames.Length == numClasses);
            _labelNames = labelNames;

            Contracts.AssertValueOrNull(stats);
            _stats = stats;
        }

        private MulticlassLogisticRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
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
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            _numFeatures = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numFeatures >= 1);

            _numClasses = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numClasses >= 1);

            _biases = ctx.Reader.ReadFloatArray(_numClasses);

            int numStarts = ctx.Reader.ReadInt32();

            if (numStarts == 0)
            {
                // The weights are entirely dense.
                int numIndices = ctx.Reader.ReadInt32();
                Host.CheckDecode(numIndices == 0);
                int numWeights = ctx.Reader.ReadInt32();
                Host.CheckDecode(numWeights == _numClasses * _numFeatures);
                _weights = new VBuffer<float>[_numClasses];
                for (int i = 0; i < _weights.Length; i++)
                {
                    var w = ctx.Reader.ReadFloatArray(_numFeatures);
                    _weights[i] = new VBuffer<float>(_numFeatures, w);
                }
                _weightsDense = _weights;
            }
            else
            {
                // Read weight matrix as CSR.
                Host.CheckDecode(numStarts == _numClasses + 1);
                int[] starts = ctx.Reader.ReadIntArray(numStarts);
                Host.CheckDecode(starts[0] == 0);
                Host.CheckDecode(Utils.IsMonotonicallyIncreasing(starts));

                int numIndices = ctx.Reader.ReadInt32();
                Host.CheckDecode(numIndices == starts[starts.Length - 1]);

                var indices = new int[_numClasses][];
                for (int i = 0; i < indices.Length; i++)
                {
                    indices[i] = ctx.Reader.ReadIntArray(starts[i + 1] - starts[i]);
                    Host.CheckDecode(Utils.IsIncreasing(0, indices[i], _numFeatures));
                }

                int numValues = ctx.Reader.ReadInt32();
                Host.CheckDecode(numValues == numIndices);

                _weights = new VBuffer<float>[_numClasses];
                for (int i = 0; i < _weights.Length; i++)
                {
                    float[] values = ctx.Reader.ReadFloatArray(starts[i + 1] - starts[i]);
                    _weights[i] = new VBuffer<float>(_numFeatures, Utils.Size(values), values, indices[i]);
                }
            }
            WarnOnOldNormalizer(ctx, GetType(), Host);
            InputType = new VectorType(NumberDataViewType.Single, _numFeatures);
            OutputType = new VectorType(NumberDataViewType.Single, _numClasses);

            // REVIEW: Should not save the label names duplicately with the predictor again.
            // Get it from the label column schema metadata instead.
            string[] labelNames = null;
            if (ctx.TryLoadBinaryStream(LabelNamesSubModelFilename, r => labelNames = LoadLabelNames(ctx, r)))
                _labelNames = labelNames;

            ctx.LoadModelOrNull<LinearModelStatistics, SignatureLoadModel>(Host, out _stats, ModelStatsSubModelFilename);
        }

        private static MulticlassLogisticRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MulticlassLogisticRegressionModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(_biases.Length == _numClasses);
            Host.Assert(_biases.Length == _weights.Length);
#if DEBUG
            foreach (var fw in _weights)
                Host.Assert(fw.Length == _numFeatures);
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
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            ctx.Writer.Write(_numFeatures);
            ctx.Writer.Write(_numClasses);
            ctx.Writer.WriteSinglesNoCount(_biases.AsSpan(0, _numClasses));
            // _weights == _weighsDense means we checked that all vectors in _weights
            // are actually dense, and so we assigned the same object, or it came dense
            // from deserialization.
            if (_weights == _weightsDense)
            {
                ctx.Writer.Write(0); // Number of starts.
                ctx.Writer.Write(0); // Number of indices.
                ctx.Writer.Write(_numFeatures * _weights.Length);
                foreach (var fv in _weights)
                {
                    Host.Assert(fv.Length == _numFeatures);
                    ctx.Writer.WriteSinglesNoCount(fv.GetValues());
                }
            }
            else
            {
                // Number of starts.
                ctx.Writer.Write(_numClasses + 1);

                // Starts always starts with 0.
                int numIndices = 0;
                ctx.Writer.Write(numIndices);
                for (int i = 0; i < _weights.Length; i++)
                {
                    // REVIEW: Assuming the presence of *any* zero justifies
                    // writing in sparse format seems stupid, but might be difficult
                    // to change without changing the format since the presence of
                    // any sparse vector means we're writing indices anyway. Revisit.
                    // This is actually a bug waiting to happen: sparse/dense vectors
                    // can have different dot products even if they are logically the
                    // same vector.
                    numIndices += NonZeroCount(in _weights[i]);
                    ctx.Writer.Write(numIndices);
                }

                ctx.Writer.Write(numIndices);
                {
                    // just scoping the count so we can use another further down
                    int count = 0;
                    foreach (var fw in _weights)
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
                    foreach (var fw in _weights)
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

            Contracts.AssertValueOrNull(_stats);
            if (_stats != null)
                ctx.SaveModel(_stats, ModelStatsSubModelFilename);
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
                    Host.Check(src.Length == _numFeatures);

                    PredictCore(in src, ref dst);
                };
            return (ValueMapper<TSrc, TDst>)(Delegate)del;
        }

        private void PredictCore(in VBuffer<float> src, ref VBuffer<float> dst)
        {
            Host.Check(src.Length == _numFeatures, "src length should equal the number of features");
            var weights = _weights;
            if (!src.IsDense)
                weights = DensifyWeights();

            var editor = VBufferEditor.Create(ref dst, _numClasses);
            for (int i = 0; i < _biases.Length; i++)
                editor.Values[i] = _biases[i] + VectorUtils.DotProduct(in weights[i], in src);

            Calibrate(editor.Values);
            dst = editor.Commit();
        }

        private VBuffer<float>[] DensifyWeights()
        {
            if (_weightsDense == null)
            {
                lock (_weights)
                {
                    if (_weightsDense == null)
                    {
                        var weightsDense = new VBuffer<float>[_numClasses];
                        for (int i = 0; i < _weights.Length; i++)
                        {
                            // Haven't yet created dense version of the weights.
                            // REVIEW: Should we always expand to full weights or should this be subject to an option?
                            var w = _weights[i];
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

        private void Calibrate(Span<float> dst)
        {
            Host.Assert(dst.Length >= _numClasses);

            // scores are in log-space; convert and fix underflow/overflow
            // TODO:   re-normalize probabilities to account for underflow/overflow?
            float softmax = MathUtils.SoftMax(dst.Slice(0, _numClasses));
            for (int i = 0; i < _numClasses; ++i)
                dst[i] = MathUtils.ExpSlow(dst[i] - softmax);
        }

        /// <summary>
        /// Output the text model to a given writer
        /// </summary>
        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine(nameof(MulticlassLogisticRegression) + " bias and non-zero weights");

            foreach (var namedValues in ((ICanGetSummaryInKeyValuePairs)this).GetSummaryInKeyValuePairs(schema))
            {
                Host.Assert(namedValues.Value is float);
                writer.WriteLine("\t{0}\t{1}", namedValues.Key, (float)namedValues.Value);
            }

            if (_stats != null)
                _stats.SaveText(writer, null, schema.Feature.Value, 20);
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValueOrNull(schema);

            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, _numFeatures, ref names);
            for (int classNumber = 0; classNumber < _biases.Length; classNumber++)
            {
                results.Add(new KeyValuePair<string, object>(
                    string.Format("{0}+(Bias)", GetLabelName(classNumber)),
                    _biases[classNumber]
                    ));
            }

            for (int classNumber = 0; classNumber < _weights.Length; classNumber++)
            {
                var orderedWeights = _weights[classNumber].Items().OrderByDescending(kv => Math.Abs(kv.Value));
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
        /// Output the text model to a given writer
        /// </summary>
        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValueOrNull(schema);

            for (int i = 0; i < _biases.Length; i++)
            {
                LinearPredictorUtils.SaveAsCode(writer,
                    in _weights[i],
                    _biases[i],
                    schema,
                    "score[" + i.ToString() + "]");
            }

            writer.WriteLine(string.Format("var softmax = MathUtils.SoftMax(scores.AsSpan(0, {0}));", _numClasses));
            for (int c = 0; c < _biases.Length; c++)
                writer.WriteLine("output[{0}] = Math.Exp(scores[{0}] - softmax);", c);
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            ((ICanSaveInTextFormat)this).SaveAsText(writer, schema);
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
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
            predictor["coeff"] = new JArray(_weights.Select(w => new JArray(w.DenseValues())));
            predictor["const"] = new JArray(_biases);
            var cell = ctx.DeclareCell("MCLinearPredictor", typeDecl, predictor);
            var cellRef = PfaUtils.Cell(cell);
            return PfaUtils.Call("m.link.softmax", PfaUtils.Call("model.reg.linear", input, cellRef));
        }

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputs, string featureColumn)
        {
            Host.CheckValue(ctx, nameof(ctx));

            string opType = "LinearClassifier";
            var node = ctx.CreateNode(opType, new[] { featureColumn }, outputs, ctx.GetNodeName(opType));
            // Selection of logit or probit output transform. enum {'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT}
            node.AddAttribute("post_transform", "NONE");
            node.AddAttribute("multi_class", true);
            node.AddAttribute("coefficients", _weights.SelectMany(w => w.DenseValues()));
            node.AddAttribute("intercepts", _biases);
            node.AddAttribute("classlabels_ints", Enumerable.Range(0, _numClasses).Select(x => (long)x));
            return true;
        }

        /// <summary>
        /// Copies the weight vector for each class into a set of buffers.
        /// </summary>
        /// <param name="weights">A possibly reusable set of vectors, which will
        /// be expanded as necessary to accomodate the data.</param>
        /// <param name="numClasses">Set to the rank, which is also the logical length
        /// of <paramref name="weights"/>.</param>
        public void GetWeights(ref VBuffer<float>[] weights, out int numClasses)
        {
            numClasses = _numClasses;
            Utils.EnsureSize(ref weights, _numClasses, _numClasses);
            for (int i = 0; i < _numClasses; i++)
                _weights[i].CopyTo(ref weights[i]);
        }

        internal IEnumerable<float> DenseWeightsEnumerable()
        {
            Contracts.Assert(_weights.Length == _biases.Length);

            int featuresCount = _weights[0].Length;
            for (var i = 0; i < _weights.Length; i++)
            {
                Host.Assert(featuresCount == _weights[i].Length);
                foreach (var weight in _weights[i].Items(all: true))
                    yield return weight.Value;
            }
        }

        /// <summary>
        /// Gets the biases for the logistic regression predictor.
        /// </summary>
        public IEnumerable<float> GetBiases()
        {
            return _biases;
        }

        internal string GetLabelName(int classNumber)
        {
            const string classNumberFormat = "Class_{0}";
            Contracts.Assert(0 <= classNumber && classNumber < _numClasses);
            return _labelNames == null ? string.Format(classNumberFormat, classNumber) : _labelNames[classNumber];
        }

        private string[] LoadLabelNames(ModelLoadContext ctx, BinaryReader reader)
        {
            Contracts.AssertValue(ctx);
            Contracts.AssertValue(reader);
            string[] labelNames = new string[_numClasses];
            for (int i = 0; i < _numClasses; i++)
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
            Contracts.Assert(Utils.Size(_labelNames) == _numClasses);
            for (int i = 0; i < _numClasses; i++)
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
                    MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, _numFeatures, ref dst);

            // Add the bias and the weight columns.
            bldr.AddColumn("Bias", NumberDataViewType.Single, _biases);
            bldr.AddColumn("Weights", getSlotNames, NumberDataViewType.Single, _weights);
            bldr.AddColumn("ClassNames", Enumerable.Range(0, _numClasses).Select(i => GetLabelName(i)).ToArray());
            return bldr.GetDataView();
        }

        DataViewRow ICanGetSummaryAsIRow.GetSummaryIRowOrNull(RoleMappedSchema schema)
        {
            return null;
        }

        DataViewRow ICanGetSummaryAsIRow.GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            if (_stats == null)
                return null;

            VBuffer<ReadOnlyMemory<char>> names = default;
            var meta = _stats.MakeStatisticsMetadata(null, schema, in names);
            return MetadataUtils.MetadataAsRow(meta);
        }
    }

    /// <summary>
    /// A component to train a logistic regression model.
    /// </summary>
    public partial class LogisticRegression
    {
        [TlcModule.EntryPoint(Name = "Trainers.LogisticRegressionClassifier",
            Desc = Summary,
            UserName = MulticlassLogisticRegression.UserNameValue,
            ShortName = MulticlassLogisticRegression.ShortName)]
        internal static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, MulticlassLogisticRegression.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLRMultiClass");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<MulticlassLogisticRegression.Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new MulticlassLogisticRegression(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
