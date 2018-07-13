// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(MultiClassNaiveBayesTrainer.Summary, typeof(MultiClassNaiveBayesTrainer), typeof(MultiClassNaiveBayesTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MultiClassNaiveBayesTrainer.UserName,
    MultiClassNaiveBayesTrainer.LoadName,
    MultiClassNaiveBayesTrainer.ShortName, DocName = "trainer/NaiveBayes.md")]

[assembly: LoadableClass(typeof(MultiClassNaiveBayesPredictor), null, typeof(SignatureLoadModel),
    "Multi Class Naive Bayes predictor", MultiClassNaiveBayesPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(MultiClassNaiveBayesTrainer), null, typeof(SignatureEntryPointModule), "MultiClassNaiveBayes")]

namespace Microsoft.ML.Runtime.Learners
{
    public sealed class MultiClassNaiveBayesTrainer : TrainerBase<MultiClassNaiveBayesPredictor>
    {
        public const string LoadName = "MultiClassNaiveBayes";
        internal const string UserName = "Multiclass Naive Bayes";
        internal const string ShortName = "MNB";
        internal const string Summary = "Trains a multiclass Naive Bayes predictor that supports binary feature values.";
        internal const string Remarks = @"<remarks>
<a href ='https://en.wikipedia.org/wiki/Naive_Bayes_classifier'>Naive Bayes</a> is a probabilistic classifier that can be used for multiclass problems. 
Using Bayes' theorem, the conditional probability for a sample belonging to a class can be calculated based on the sample count for each feature combination groups.
However, Naive Bayes Classifier is feasible only if the number of features and the values each feature can take is relatively small.
It also assumes that the features are strictly independent.
</remarks>";

        public sealed class Arguments : LearnerInputBaseWithLabel
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public override TrainerInfo Info { get; }

        public MultiClassNaiveBayesTrainer(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Info = new TrainerInfo(normalization: false, caching: false);
        }

        public override MultiClassNaiveBayesPredictor Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            Host.Check(data.Schema.Label != null, "Missing Label column");
            Host.Check(data.Schema.Label.Type == NumberType.Float || data.Schema.Label.Type is KeyType,
                "Invalid type for Label column, only floats and known-size keys are supported");

            Host.Check(data.Schema.Feature != null, "Missing Feature column");
            int featureCount;
            data.CheckFeatureFloatVector(out featureCount);
            int labelCount = 0;
            if (data.Schema.Label.Type.IsKey)
                labelCount = data.Schema.Label.Type.KeyCount;

            int[] labelHistogram = new int[labelCount];
            int[][] featureHistogram = new int[labelCount][];
            using (var pch = Host.StartProgressChannel("Multi Class Naive Bayes training"))
            using (var ch = Host.Start("Training"))
            using (var cursor = new MultiClassLabelCursor(labelCount, data, CursOpt.Features | CursOpt.Label))
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
                        ch.Done();
                        break;
                    }

                    int size = cursor.Label + 1;
                    Utils.EnsureSize(ref labelHistogram, size);
                    Utils.EnsureSize(ref featureHistogram, size);
                    Utils.EnsureSize(ref featureHistogram[cursor.Label], featureCount);
                    labelHistogram[cursor.Label] += 1;
                    labelCount = labelCount < size ? size : labelCount;

                    if (cursor.Features.IsDense)
                    {
                        for (int i = 0; i < cursor.Features.Count; i += 1)
                        {
                            if (cursor.Features.Values[i] > 0)
                                featureHistogram[cursor.Label][i] += 1;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < cursor.Features.Count; i += 1)
                        {
                            if (cursor.Features.Values[i] > 0)
                                featureHistogram[cursor.Label][cursor.Features.Indices[i]] += 1;
                        }
                    }

                    examplesProcessed += 1;
                }
                ch.Done();
            }

            Array.Resize(ref labelHistogram, labelCount);
            Array.Resize(ref featureHistogram, labelCount);
            return new MultiClassNaiveBayesPredictor(Host, labelHistogram, featureHistogram, featureCount);
        }

        [TlcModule.EntryPoint(Name = "Trainers.NaiveBayesClassifier",
            Desc = "Train a MultiClassNaiveBayesTrainer.",
            UserName = UserName, ShortName = ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClassNaiveBayesTrainer(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainMultiClassNaiveBayes");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new MultiClassNaiveBayesTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    public sealed class MultiClassNaiveBayesPredictor :
        PredictorBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveModel
    {
        public const string LoaderSignature = "MultiClassNaiveBayesPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MNABYPRD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly int[] _labelHistogram;
        private readonly int[][] _featureHistogram;
        private readonly double[] _absentFeaturesLogProb;
        private readonly int _totalTrainingCount;
        private readonly int _labelCount;
        private readonly int _featureCount;
        private readonly VectorType _inputType;
        private readonly VectorType _outputType;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public ColumnType InputType => _inputType;

        public ColumnType OutputType => _outputType;

        internal MultiClassNaiveBayesPredictor(IHostEnvironment env, int[] labelHistogram, int[][] featureHistogram, int featureCount)
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
            _inputType = new VectorType(NumberType.Float, _featureCount);
            _outputType = new VectorType(NumberType.R4, _labelCount);
        }

        private MultiClassNaiveBayesPredictor(IHostEnvironment env, ModelLoadContext ctx)
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
            _inputType = new VectorType(NumberType.Float, _featureCount);
            _outputType = new VectorType(NumberType.R4, _labelCount);
        }

        public static MultiClassNaiveBayesPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiClassNaiveBayesPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _labelCount
            // int[_labelCount]: _labelHistogram
            // int: _featureCount
            // int[_labelCount][_featureCount]: _featureHistogram
            // int[_labelCount]: _absentFeaturesLogProb
            ctx.Writer.WriteIntArray(_labelHistogram, _labelCount);
            ctx.Writer.Write(_featureCount);
            for (int i = 0; i < _labelCount; i += 1)
            {
                if (_labelHistogram[i] > 0)
                    ctx.Writer.WriteIntsNoCount(_featureHistogram[i], _featureCount);
            }

            ctx.Writer.WriteDoublesNoCount(_absentFeaturesLogProb, _labelCount);
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

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
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

        private void Map(ref VBuffer<float> src, ref VBuffer<float> dst)
        {
            Host.Check(src.Length == _featureCount, "Invalid number of features passed.");
            float[] labelScores = (dst.Length >= _labelCount) ? dst.Values : new float[_labelCount];
            for (int iLabel = 0; iLabel < _labelCount; iLabel += 1)
            {
                double labelOccurrenceCount = _labelHistogram[iLabel];
                double logProb = Math.Log(labelOccurrenceCount / _totalTrainingCount);
                double absentFeatureLogProb = 0;
                if (_labelHistogram[iLabel] > 0)
                {
                    if (src.IsDense)
                    {
                        for (int iFeature = 0; iFeature < src.Count; iFeature += 1)
                        {
                            ComputeLabelProbabilityFromFeature(labelOccurrenceCount, iLabel, iFeature,
                                src.Values[iFeature], ref logProb, ref absentFeatureLogProb);
                        }
                    }
                    else
                    {
                        for (int iFeature = 0; iFeature < src.Count; iFeature += 1)
                        {
                            ComputeLabelProbabilityFromFeature(labelOccurrenceCount, iLabel, src.Indices[iFeature],
                                src.Values[iFeature], ref logProb, ref absentFeatureLogProb);
                        }
                    }
                }

                labelScores[iLabel] =
                    (float)(logProb + (_absentFeaturesLogProb[iLabel] - absentFeatureLogProb));
            }

            dst = new VBuffer<float>(_labelCount, labelScores, dst.Indices);
        }
    }
}
