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
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(NaiveBayesMulticlassTrainer.Summary, typeof(NaiveBayesMulticlassTrainer), typeof(NaiveBayesMulticlassTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    NaiveBayesMulticlassTrainer.UserName,
    NaiveBayesMulticlassTrainer.LoadName,
    NaiveBayesMulticlassTrainer.ShortName, DocName = "trainer/NaiveBayes.md")]

[assembly: LoadableClass(typeof(NaiveBayesMulticlassModelParameters), null, typeof(SignatureLoadModel),
    "Multi Class Naive Bayes predictor", NaiveBayesMulticlassModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(NaiveBayesMulticlassTrainer), null, typeof(SignatureEntryPointModule), NaiveBayesMulticlassTrainer.LoadName)]

namespace Microsoft.ML.Trainers
{
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
            Host.Check(labelCol.Type == NumberDataViewType.Single || labelCol.Type is KeyType,
                "Invalid type for Label column, only floats and known-size keys are supported");

            Host.Check(data.Schema.Feature.HasValue, "Missing Feature column");
            int featureCount;
            data.CheckFeatureFloatVector(out featureCount);
            int labelCount = 0;
            if (labelCol.Type is KeyType labelKeyType)
                labelCount = labelKeyType.GetCountAsInt32(Host);

            int[] labelHistogram = new int[labelCount];
            int[][] featureHistogram = new int[labelCount][];
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
                    Utils.EnsureSize(ref featureHistogram[cursor.Label], featureCount);
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

    public sealed class NaiveBayesMulticlassModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper
    {
        internal const string LoaderSignature = "MultiClassNaiveBayesPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MNABYPRD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NaiveBayesMulticlassModelParameters).Assembly.FullName);
        }

        private readonly int[] _labelHistogram;
        private readonly int[][] _featureHistogram;
        private readonly double[] _absentFeaturesLogProb;
        private readonly int _totalTrainingCount;
        private readonly int _labelCount;
        private readonly int _featureCount;
        private readonly VectorType _inputType;
        private readonly VectorType _outputType;

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        DataViewType IValueMapper.InputType => _inputType;

        DataViewType IValueMapper.OutputType => _outputType;

        /// <summary>
        /// Get the label histogram.
        /// </summary>
        public IReadOnlyList<int> GetLabelHistogram() => _labelHistogram;

        /// <summary>
        /// Get the feature histogram.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<int>> GetFeatureHistogram() => _featureHistogram;

        /// <summary>
        /// Instantiates new model parameters from trained model.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="labelHistogram">The histogram of labels.</param>
        /// <param name="featureHistogram">The feature histogram.</param>
        /// <param name="featureCount">The number of features.</param>
        internal NaiveBayesMulticlassModelParameters(IHostEnvironment env, int[] labelHistogram, int[][] featureHistogram, int featureCount)
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
            _inputType = new VectorType(NumberDataViewType.Single, _featureCount);
            _outputType = new VectorType(NumberDataViewType.Single, _labelCount);
        }

        private NaiveBayesMulticlassModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _labelCount
            // int[_labelCount]: _labelHistogram
            // int: _featureCount
            // int[_labelCount][_featureCount]: _featureHistogram
            // int[_labelCount]: _absentFeaturesLogProb
            _labelHistogram = ctx.Reader.ReadIntArray() ?? new int[0];
            _labelCount = _labelHistogram.Length;

            foreach (int labelCount in _labelHistogram)
                Host.CheckDecode(labelCount >= 0);

            _featureCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(_featureCount >= 0);
            _featureHistogram = new int[_labelCount][];
            for (int iLabel = 0; iLabel < _labelCount; iLabel += 1)
            {
                if (_labelHistogram[iLabel] > 0)
                {
                    _featureHistogram[iLabel] = ctx.Reader.ReadIntArray(_featureCount);
                    for (int iFeature = 0; iFeature < _featureCount; iFeature += 1)
                        Host.CheckDecode(_featureHistogram[iLabel][iFeature] >= 0);
                }
            }

            _absentFeaturesLogProb = ctx.Reader.ReadDoubleArray(_labelCount);
            _totalTrainingCount = _labelHistogram.Sum();
            _inputType = new VectorType(NumberDataViewType.Single, _featureCount);
            _outputType = new VectorType(NumberDataViewType.Single, _labelCount);
        }

        private static NaiveBayesMulticlassModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
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
            // int[_labelCount]: _labelHistogram
            // int: _featureCount
            // int[_labelCount][_featureCount]: _featureHistogram
            // int[_labelCount]: _absentFeaturesLogProb
            ctx.Writer.WriteIntArray(_labelHistogram.AsSpan(0, _labelCount));
            ctx.Writer.Write(_featureCount);
            for (int i = 0; i < _labelCount; i += 1)
            {
                if (_labelHistogram[i] > 0)
                    ctx.Writer.WriteIntsNoCount(_featureHistogram[i].AsSpan(0, _featureCount));
            }

            ctx.Writer.WriteDoublesNoCount(_absentFeaturesLogProb.AsSpan(0, _labelCount));
        }

        private static double[] CalculateAbsentFeatureLogProbabilities(int[] labelHistogram, int[][] featureHistogram, int featureCount)
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
                        int labelOccuranceCount = labelHistogram[iLabel];
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
