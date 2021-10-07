// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(NaiveBayesMulticlassTrainer.Summary, typeof(NaiveBayesMulticlassTrainer), typeof(NaiveBayesMulticlassTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    NaiveBayesMulticlassTrainer.UserName,
    NaiveBayesMulticlassTrainer.LoadName,
    NaiveBayesMulticlassTrainer.ShortName)]

[assembly: LoadableClass(typeof(NaiveBayesMulticlassModelParameters), null, typeof(SignatureLoadModel),
    "Multi Class Naive Bayes predictor", NaiveBayesMulticlassModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(NaiveBayesMulticlassTrainer), null, typeof(SignatureEntryPointModule), NaiveBayesMulticlassTrainer.LoadName)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a multiclass Naive Bayes model that supports binary feature values.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [NaiveBayes](xref:Microsoft.ML.StandardTrainersCatalog.NaiveBayes(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Training Algorithm Details
    /// [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
    /// is a probabilistic classifier that can be used for multiclass problems.
    /// Using Bayes' theorem, the conditional probability for a sample belonging to a class
    /// can be calculated based on the sample count for each feature combination groups.
    /// However, Naive Bayes Classifier is feasible only if the number of features and
    /// the values each feature can take is relatively small.
    /// It assumes independence among the presence of features in a class even though
    /// they may be dependent on each other.
    /// This multi-class trainer accepts "binary" feature values of type float:
    /// feature values that are greater than zero are treated as `true` and feature values
    /// that are less or equal to 0 are treated as `false`.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="StandardTrainersCatalog.NaiveBayes(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String)"/>
    public sealed class NaiveBayesMulticlassTrainer : TrainerEstimatorBase<MulticlassPredictionTransformer<NaiveBayesMulticlassModelParameters>, NaiveBayesMulticlassModelParameters>
    {
        internal const string LoadName = "MultiClassNaiveBayes";
        internal const string UserName = "Multiclass Naive Bayes";
        internal const string ShortName = "MNB";
        internal const string Summary = "Trains a multiclass Naive Bayes predictor that supports binary feature values.";

        internal sealed class Options : TrainerInputBaseWithLabel
        {
        }

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Initializes a new instance of <see cref="NaiveBayesMulticlassTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        internal NaiveBayesMulticlassTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadName), TrainerUtils.MakeR4VecFeature(featureColumn),
                  TrainerUtils.MakeU4ScalarColumn(labelColumn))
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NaiveBayesMulticlassTrainer"/>
        /// </summary>
        internal NaiveBayesMulticlassTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadName), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  TrainerUtils.MakeU4ScalarColumn(options.LabelColumnName))
        {
            Host.CheckValue(options, nameof(options));
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var predLabelMetadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, predLabelMetadata)
            };
        }

        private protected override MulticlassPredictionTransformer<NaiveBayesMulticlassModelParameters> MakeTransformer(NaiveBayesMulticlassModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<NaiveBayesMulticlassModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        private protected override NaiveBayesMulticlassModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            Host.Check(data.Schema.Label.HasValue, "Missing Label column");
            var labelCol = data.Schema.Label.Value;
            Host.Check(labelCol.Type == NumberDataViewType.Single || labelCol.Type is KeyDataViewType,
                "Invalid type for Label column, only floats and known-size keys are supported");

            Host.Check(data.Schema.Feature.HasValue, "Missing Feature column");
            int featureCount;
            data.CheckFeatureFloatVector(out featureCount);
            int labelCount = 0;
            if (labelCol.Type is KeyDataViewType labelKeyType)
                labelCount = labelKeyType.GetCountAsInt32(Host);

            long[] labelHistogram = new long[labelCount];
            long[][] featureHistogram = new long[labelCount][];
            using (var pch = Host.StartProgressChannel("Multi Class Naive Bayes training"))
            using (var ch = Host.Start("Training"))
            using (var cursor = new MulticlassLabelCursor(labelCount, data, CursOpt.Features | CursOpt.Label))
            {
                int examplesProcessed = 0;
                pch.SetHeader(new ProgressHeader(new[] { "Examples Processed" }, new[] { "count" }), e =>
                   {
                       e.SetProgress(0, examplesProcessed, int.MaxValue);
                   });

                while (cursor.MoveNext())
                {
                    if (cursor.Row.Position > int.MaxValue)
                    {
                        ch.Warning("Stopping training because maximum number of rows have been traversed");
                        break;
                    }

                    int size = cursor.Label + 1;
                    Utils.EnsureSize(ref labelHistogram, size);
                    Utils.EnsureSize(ref featureHistogram, size);
                    if (featureHistogram[cursor.Label] == null)
                        featureHistogram[cursor.Label] = new long[featureCount];
                    labelHistogram[cursor.Label] += 1;
                    labelCount = labelCount < size ? size : labelCount;

                    var featureValues = cursor.Features.GetValues();
                    if (cursor.Features.IsDense)
                    {
                        for (int i = 0; i < featureValues.Length; i += 1)
                        {
                            if (featureValues[i] > 0)
                                featureHistogram[cursor.Label][i] += 1;
                        }
                    }
                    else
                    {
                        var featureIndices = cursor.Features.GetIndices();
                        for (int i = 0; i < featureValues.Length; i += 1)
                        {
                            if (featureValues[i] > 0)
                                featureHistogram[cursor.Label][featureIndices[i]] += 1;
                        }
                    }

                    examplesProcessed += 1;
                }
            }

            Array.Resize(ref labelHistogram, labelCount);
            Array.Resize(ref featureHistogram, labelCount);
            return new NaiveBayesMulticlassModelParameters(Host, labelHistogram, featureHistogram, featureCount);
        }

        [TlcModule.EntryPoint(Name = "Trainers.NaiveBayesClassifier",
            Desc = "Train a MulticlassNaiveBayesTrainer.",
            UserName = UserName,
            ShortName = ShortName)]
        internal static CommonOutputs.MulticlassClassificationOutput TrainMulticlassNaiveBayesTrainer(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainMultiClassNaiveBayes");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new NaiveBayesMulticlassTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName));
        }
    }

    /// <summary>
    /// Model parameters for <see cref="NaiveBayesMulticlassTrainer"/>.
    /// </summary>
    public sealed class NaiveBayesMulticlassModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper,
        ISingleCanSaveOnnx
    {
        internal const string LoaderSignature = "MultiClassNaiveBayesPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MNABYPRD",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Histograms are of type long
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NaiveBayesMulticlassModelParameters).Assembly.FullName);
        }

        private readonly long[] _labelHistogram;
        private readonly long[][] _featureHistogram;
        private readonly double[] _absentFeaturesLogProb;
        private readonly long _totalTrainingCount;
        private readonly int _labelCount;
        private readonly int _featureCount;
        private readonly VectorDataViewType _inputType;
        private readonly VectorDataViewType _outputType;

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        DataViewType IValueMapper.InputType => _inputType;

        DataViewType IValueMapper.OutputType => _outputType;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        /// <summary>
        /// Get the label histogram.
        /// </summary>
        [Obsolete("This API is deprecated, please use GetLabelHistogramLong() which returns _labelHistogram " +
            "with type IReadOnlyList<long> to avoid overflow errors with large datasets.", true)]
        public IReadOnlyList<int> GetLabelHistogram() => Array.ConvertAll(_labelHistogram, x => (int)x);

        /// <summary>
        /// Get the label histogram with generic type long.
        /// </summary>
        public IReadOnlyList<long> GetLabelHistogramLong() => _labelHistogram;

        /// <summary>
        /// Get the feature histogram.
        /// </summary>
        [Obsolete("This API is deprecated, please use GetFeatureHistogramLong() which returns _featureHistogram " +
            "with type IReadOnlyList<long> to avoid overflow errors with large datasets.", true)]
        public IReadOnlyList<IReadOnlyList<int>> GetFeatureHistogram() => Array.ConvertAll(_featureHistogram, x => Array.ConvertAll(x, y => (int)y));

        /// <summary>
        /// Get the feature histogram with generic type long.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<long>> GetFeatureHistogramLong() => _featureHistogram;

        /// <summary>
        /// Instantiates new model parameters from trained model.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="labelHistogram">The histogram of labels.</param>
        /// <param name="featureHistogram">The feature histogram.</param>
        /// <param name="featureCount">The number of features.</param>
        internal NaiveBayesMulticlassModelParameters(IHostEnvironment env, long[] labelHistogram, long[][] featureHistogram, int featureCount)
            : base(env, LoaderSignature)
        {
            Host.AssertValue(labelHistogram);
            Host.AssertValue(featureHistogram);
            Host.Assert(labelHistogram.Length == featureHistogram.Length);
            Host.Assert(featureHistogram.All(h => h == null || h.Length == featureCount));
            _labelHistogram = labelHistogram;
            _featureHistogram = featureHistogram;
            _totalTrainingCount = _labelHistogram.Sum();
            _labelCount = _labelHistogram.Length;
            _featureCount = featureCount;
            _absentFeaturesLogProb = CalculateAbsentFeatureLogProbabilities(_labelHistogram, _featureHistogram, _featureCount);
            _inputType = new VectorDataViewType(NumberDataViewType.Single, _featureCount);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, _labelCount);
        }

        /// <remarks>
        /// The unit test TestEntryPoints.LoadEntryPointModel() exercises the ReadIntArrary(int size) codepath below
        /// as its ctx.Header.ModelVerWritten is 0x00010001, and the persistent model that gets loaded and executed
        /// for this unit test is located at test\data\backcompat\ep_model3.zip/>
        /// </remarks>
        private NaiveBayesMulticlassModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _labelCount (read during reading of _labelHistogram in ReadLongArray())
            // long[_labelCount]: _labelHistogram
            // int: _featureCount
            // long[_labelCount][_featureCount]: _featureHistogram
            // int[_labelCount]: _absentFeaturesLogProb
            if (ctx.Header.ModelVerWritten >= 0x00010002)
                _labelHistogram = ctx.Reader.ReadLongArray() ?? new long[0];
            else
            {
                _labelHistogram = Array.ConvertAll(ctx.Reader.ReadIntArray() ?? new int[0], x => (long)x);
            }
            _labelCount = _labelHistogram.Length;

            foreach (int labelCount in _labelHistogram)
                Host.CheckDecode(labelCount >= 0);

            _featureCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(_featureCount >= 0);
            _featureHistogram = new long[_labelCount][];
            for (int iLabel = 0; iLabel < _labelCount; iLabel += 1)
            {
                if (_labelHistogram[iLabel] > 0)
                {
                    if (ctx.Header.ModelVerWritten >= 0x00010002)
                        _featureHistogram[iLabel] = ctx.Reader.ReadLongArray(_featureCount);
                    else
                        _featureHistogram[iLabel] = Array.ConvertAll(ctx.Reader.ReadIntArray(_featureCount) ?? new int[0], x => (long)x);
                    for (int iFeature = 0; iFeature < _featureCount; iFeature += 1)
                        Host.CheckDecode(_featureHistogram[iLabel][iFeature] >= 0);
                }
            }

            _absentFeaturesLogProb = ctx.Reader.ReadDoubleArray(_labelCount);
            _totalTrainingCount = _labelHistogram.Sum();
            _inputType = new VectorDataViewType(NumberDataViewType.Single, _featureCount);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, _labelCount);
        }

        internal static NaiveBayesMulticlassModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new NaiveBayesMulticlassModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _labelCount
            // long[_labelCount]: _labelHistogram
            // int: _featureCount
            // long[_labelCount][_featureCount]: _featureHistogram
            // int[_labelCount]: _absentFeaturesLogProb
            ctx.Writer.Write(_labelCount);
            ctx.Writer.WriteLongStream(_labelHistogram);
            ctx.Writer.Write(_featureCount);
            for (int i = 0; i < _labelCount; i += 1)
            {
                if (_labelHistogram[i] > 0)
                    ctx.Writer.WriteLongStream(_featureHistogram[i]);
            }

            ctx.Writer.WriteDoublesNoCount(_absentFeaturesLogProb.AsSpan(0, _labelCount));
        }

        private static double[] CalculateAbsentFeatureLogProbabilities(long[] labelHistogram, long[][] featureHistogram, int featureCount)
        {
            int labelCount = labelHistogram.Length;
            double[] absentFeaturesLogProb = new double[labelCount];
            for (int iLabel = 0; iLabel < labelHistogram.Length; iLabel += 1)
            {
                if (labelHistogram[iLabel] > 0)
                {
                    double logProb = 0;
                    for (int iFeature = 0; iFeature < featureCount; iFeature += 1)
                    {
                        long labelOccuranceCount = labelHistogram[iLabel];
                        logProb +=
                            Math.Log(1 + ((double)labelOccuranceCount - featureHistogram[iLabel][iFeature])) -
                            Math.Log(labelOccuranceCount + labelCount);
                    }

                    absentFeaturesLogProb[iLabel] = logProb;
                }
            }

            return absentFeaturesLogProb;
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            ValueMapper<VBuffer<float>, VBuffer<float>> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        /// <summary>
        /// Creates an Onnx inferencing model by vectorizing and following the logic found in <see cref="Map"/>
        /// </summary>
        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
        {
            const int minimumOpSetVersion = 9;
            ctx.CheckOpSetVersion(minimumOpSetVersion, "MulticlassNaiveBayes");

            float[] featureHistogram = new float[_featureHistogram[0].Length * _labelHistogram.Length];
            float[] labelHistogramExpanded = new float[_featureHistogram[0].Length * _labelHistogram.Length];

            for (int i = 0; i < _featureHistogram.Length; i++)
            {
                Array.Copy(_featureHistogram[i], 0, featureHistogram, i * _featureHistogram[i].Length, _featureHistogram[i].Length);
            }
            for (int i = 0; i < _featureHistogram[0].Length; i++)
            {
                Array.Copy(_labelHistogram, 0, labelHistogramExpanded, i * _featureHistogram.Length, _featureHistogram.Length);
            }

            var one = ctx.AddInitializer(1.0f, "one");
            var oneInt = ctx.AddInitializer(1, typeof(int), "oneInt");
            var zero = ctx.AddInitializer(0.0f, "zero");
            var labelCount = ctx.AddInitializer((float)_labelCount, "labelCount");
            var trainingCount = ctx.AddInitializer((float)_totalTrainingCount, "totalTrainingCount");
            var labelHistogram = ctx.AddInitializer(labelHistogramExpanded.Take(_labelHistogram.Length), new long[] { _labelHistogram.Length, 1 }, "labelHistogram");

            var featureHistogramName = ctx.AddInitializer(featureHistogram, new long[] { _featureHistogram.Length, _featureHistogram[0].Length }, "featureHistogram");
            var labelHistogramName = ctx.AddInitializer(labelHistogramExpanded, new long[] { _featureHistogram[0].Length, _labelHistogram.Length }, "labelHistogramExpanded");
            var learnedAbsentFeatureLogProb = ctx.AddInitializer(_absentFeaturesLogProb, new long[] { _absentFeaturesLogProb.Length, 1 }, "absentFeaturesLogProb");

            var typeOne = new VectorDataViewType(NumberDataViewType.Single, 1);
            var typeFea = new VectorDataViewType(NumberDataViewType.Single, _featureHistogram[0].Length);
            var typeLabelByFea = new VectorDataViewType(NumberDataViewType.Single, _labelHistogram.Length, _featureHistogram[0].Length);
            var typeLabelByOne = new VectorDataViewType(NumberDataViewType.Single, _labelHistogram.Length, 1);

            var greaterOutput = ctx.AddIntermediateVariable(new VectorDataViewType(BooleanDataViewType.Instance, _featureHistogram[0].Length), "greaterOutput");
            var opType = "Greater";
            ctx.CreateNode(opType, new[] { featureColumn, zero }, new[] { greaterOutput }, ctx.GetNodeName(opType), "");

            opType = "Cast";
            var castOutput = ctx.AddIntermediateVariable(typeFea, "CastOutput");
            var node = ctx.CreateNode(opType, greaterOutput, castOutput, ctx.GetNodeName(opType), "");
            var t = InternalDataKindExtensions.ToInternalDataKind(DataKind.Single).ToType();
            node.AddAttribute("to", t);

            opType = "ExpandDims";
            var isFeaturePresent = ctx.AddIntermediateVariable(new VectorDataViewType(NumberDataViewType.Single, 1, _featureHistogram[0].Length), "isFeaturePresent");
            ctx.CreateNode(opType, new[] { castOutput, oneInt }, new[] { isFeaturePresent }, ctx.GetNodeName(opType), "com.microsoft");

            //initialize logProb
            opType = "Div";
            var divOutput = ctx.AddIntermediateVariable(typeOne, "DivOutput");
            ctx.CreateNode(opType, new[] { labelHistogram, trainingCount }, new[] { divOutput }, ctx.GetNodeName(opType), "");

            opType = "Log";
            var logOutput = ctx.AddIntermediateVariable(typeOne, "LogOutput");
            ctx.CreateNode(opType, divOutput, logOutput, ctx.GetNodeName(opType), "");

            //log1
            opType = "Sum";
            var sumOutput = ctx.AddIntermediateVariable(_inputType, "SumOutput");
            ctx.CreateNode(opType, new[] { featureHistogramName, one }, new[] { sumOutput }, ctx.GetNodeName(opType), "");

            var logOutput1 = ctx.AddIntermediateVariable(typeLabelByFea, "LogOutput");
            LogMul(ctx, sumOutput, isFeaturePresent, logOutput1);

            //log2
            opType = "Transpose";
            var labelHistogramTrans = ctx.AddIntermediateVariable(typeFea, "Transpose");
            ctx.CreateNode(opType, labelHistogramName, labelHistogramTrans, ctx.GetNodeName(opType), "");

            opType = "Sub";
            var absentFeatureCount = ctx.AddIntermediateVariable(typeFea, "AbsentFeatureCounts");
            ctx.CreateNode(opType, new[] { labelHistogramTrans, featureHistogramName }, new[] { absentFeatureCount }, ctx.GetNodeName(opType), "");

            opType = "Sum";
            sumOutput = ctx.AddIntermediateVariable(typeFea, "SumOutput");
            ctx.CreateNode(opType, new[] { labelHistogramTrans, labelCount }, new[] { sumOutput }, ctx.GetNodeName(opType), "");

            var logOutput2 = ctx.AddIntermediateVariable(typeLabelByFea, "LogOutput");
            LogMul(ctx, sumOutput, isFeaturePresent, logOutput2);

            //log3
            opType = "Sum";
            sumOutput = ctx.AddIntermediateVariable(typeFea, "SumOutput");
            ctx.CreateNode(opType, new[] { absentFeatureCount, one }, new[] { sumOutput }, ctx.GetNodeName(opType), "");

            var logOutput3 = ctx.AddIntermediateVariable(typeLabelByFea, "LogOutput");
            LogMul(ctx, sumOutput, isFeaturePresent, logOutput3);

            //result
            opType = "Sub";
            var logProb = ctx.AddIntermediateVariable(typeLabelByFea, "LogProb");
            ctx.CreateNode(opType, new[] { logOutput1, logOutput2 }, new[] { logProb }, ctx.GetNodeName(opType), "");

            opType = "Sub";
            var absentFeatureLogProb = ctx.AddIntermediateVariable(typeLabelByFea, "AbsentFeatureLogProb");
            ctx.CreateNode(opType, new[] { logOutput3, logOutput2 }, new[] { absentFeatureLogProb }, ctx.GetNodeName(opType), "");

            opType = "ReduceSum";
            var logProbReduceSum = ctx.AddIntermediateVariable(typeLabelByOne, "ReduceSum");
            node = ctx.CreateNode(opType, new[] { logProb }, new[] { logProbReduceSum }, ctx.GetNodeName(opType), "");
            long[] list = { 2 };
            node.AddAttribute("axes", list);

            opType = "ReduceSum";
            var absentFeatureLogProbReduceSum = ctx.AddIntermediateVariable(typeLabelByOne, "ReduceSum");
            node = ctx.CreateNode(opType, new[] { absentFeatureLogProb }, new[] { absentFeatureLogProbReduceSum }, ctx.GetNodeName(opType), "");
            node.AddAttribute("axes", list);

            opType = "Cast";
            castOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, "CastOutput");
            node = ctx.CreateNode(opType, learnedAbsentFeatureLogProb, castOutput, ctx.GetNodeName(opType), "");
            t = InternalDataKindExtensions.ToInternalDataKind(DataKind.Single).ToType();
            node.AddAttribute("to", t);

            opType = "Sub";
            var subOutput = ctx.AddIntermediateVariable(typeLabelByOne, "SubOutput");
            ctx.CreateNode(opType, new[] { castOutput, absentFeatureLogProbReduceSum }, new[] { subOutput }, ctx.GetNodeName(opType), "");

            opType = "Sum";
            sumOutput = ctx.AddIntermediateVariable(typeLabelByOne, "SumOutput");
            ctx.CreateNode(opType, new[] { subOutput, logProbReduceSum, logOutput }, new[] { sumOutput }, ctx.GetNodeName(opType), "");

            opType = "Squeeze";
            var squeezeNode = ctx.CreateNode(opType, sumOutput, outputNames[1], ctx.GetNodeName(opType), "");
            squeezeNode.AddAttribute("axes", new long[] { 2 });

            opType = "ArgMax";
            var scoreIndex = ctx.AddIntermediateVariable(new VectorDataViewType(NumberDataViewType.Int64, 1), "ScoreIndex");
            node = ctx.CreateNode(opType, new[] { sumOutput }, new[] { scoreIndex }, ctx.GetNodeName(opType), "");
            node.AddAttribute("axis", 1);
            node.AddAttribute("keepdims", 0);

            opType = "Cast";
            castOutput = ctx.AddIntermediateVariable(typeOne, "CastOutput");
            node = ctx.CreateNode(opType, scoreIndex, castOutput, ctx.GetNodeName(opType), "");
            t = InternalDataKindExtensions.ToInternalDataKind(DataKind.Single).ToType();
            node.AddAttribute("to", t);

            //log3
            opType = "Sum";
            sumOutput = ctx.AddIntermediateVariable(typeOne, "SumOutput");
            ctx.CreateNode(opType, new[] { castOutput, one }, new[] { sumOutput }, ctx.GetNodeName(opType), "");

            opType = "Cast";
            node = ctx.CreateNode(opType, sumOutput, outputNames[0], ctx.GetNodeName(opType), "");
            t = InternalDataKindExtensions.ToInternalDataKind(DataKind.UInt32).ToType();
            node.AddAttribute("to", t);

            return true;
        }

        private void LogMul(OnnxContext ctx, string input, string isFeaturePresent, string output)
        {
            var opType = "Log";
            var logOutput = ctx.AddIntermediateVariable(new VectorDataViewType(NumberDataViewType.Single, _featureHistogram[0].Length), "LogOutput");
            ctx.CreateNode(opType, input, logOutput, ctx.GetNodeName(opType), "");

            opType = "Mul";
            ctx.CreateNode(opType, new[] { logOutput, isFeaturePresent }, new[] { output }, ctx.GetNodeName(opType), "");
        }

        private void ComputeLabelProbabilityFromFeature(double labelOccurrenceCount, int labelIndex, int featureIndex,
            float featureValue, ref double logProb, ref double absentFeatureLogProb)
        {
            if (featureValue <= 0)
                return;

            double featureCount = _featureHistogram[labelIndex][featureIndex];
            double absentFeatureCount = labelOccurrenceCount - featureCount;
            Host.Assert(featureCount >= 0);
            logProb += Math.Log(featureCount + 1) - Math.Log(labelOccurrenceCount + _labelCount);
            absentFeatureLogProb += Math.Log(absentFeatureCount + 1) - Math.Log(labelOccurrenceCount + _labelCount);
        }

        private void Map(in VBuffer<float> src, ref VBuffer<float> dst)
        {
            Host.Check(src.Length == _featureCount, "Invalid number of features passed.");

            var srcValues = src.GetValues();
            var srcIndices = src.GetIndices();

            var editor = VBufferEditor.Create(ref dst, _labelCount);
            Span<float> labelScores = editor.Values;
            for (int iLabel = 0; iLabel < _labelCount; iLabel += 1)
            {
                double labelOccurrenceCount = _labelHistogram[iLabel];
                double logProb = Math.Log(labelOccurrenceCount / _totalTrainingCount);
                double absentFeatureLogProb = 0;
                if (_labelHistogram[iLabel] > 0)
                {
                    if (src.IsDense)
                    {
                        for (int iFeature = 0; iFeature < srcValues.Length; iFeature += 1)
                        {
                            ComputeLabelProbabilityFromFeature(labelOccurrenceCount, iLabel, iFeature,
                                srcValues[iFeature], ref logProb, ref absentFeatureLogProb);
                        }
                    }
                    else
                    {
                        for (int iFeature = 0; iFeature < srcValues.Length; iFeature += 1)
                        {
                            ComputeLabelProbabilityFromFeature(labelOccurrenceCount, iLabel, srcIndices[iFeature],
                                srcValues[iFeature], ref logProb, ref absentFeatureLogProb);
                        }
                    }
                }

                labelScores[iLabel] =
                    (float)(logProb + (_absentFeaturesLogProb[iLabel] - absentFeatureLogProb));
            }

            dst = editor.Commit();
        }
    }
}
