// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(LogisticRegression.Summary, typeof(LogisticRegression), typeof(LogisticRegression.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LogisticRegression.UserNameValue,
    LogisticRegression.LoadNameValue,
    LogisticRegression.ShortName,
    "logisticregressionwrapper")]

[assembly: LoadableClass(typeof(void), typeof(LogisticRegression), null, typeof(SignatureEntryPointModule), "LogisticRegression")]

namespace Microsoft.ML.Runtime.Learners
{
    /// <include file='doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file='doc.xml' path='docs/members/example[@name="LogisticRegressionBinaryClassifier"]/*' />
    public sealed partial class LogisticRegression : LbfgsTrainerBase<Float, ParameterMixingCalibratedPredictor>
    {
        public const string LoadNameValue = "LogisticRegression";
        internal const string UserNameValue = "Logistic Regression";
        internal const string ShortName = "lr";
        internal const string Summary = "Logistic Regression is a method in statistics used to predict the probability of occurrence of an event and can "
            + "be used as a classification algorithm. The algorithm predicts the probability of occurrence of an event by fitting data to a logistical function.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat", SortOrder = 50)]
            public bool ShowTrainingStats = false;
        }

        private Double _posWeight;

        public LogisticRegression(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue, Contracts.CheckRef(args, nameof(args)).ShowTrainingStats)
        {
            _posWeight = 0;
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        protected override Float AccumulateOneGradient(ref VBuffer<Float> feat, Float label, Float weight,
            ref VBuffer<Float> x, ref VBuffer<Float> grad, ref Float[] scratch)
        {
            Float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            Float score = bias + VectorUtils.DotProductWithOffset(ref x, 1, ref feat);

            Float s = score / 2;

            Float logZ = MathUtils.SoftMax(s, -s);
            Float label01 = Math.Min(1, Math.Max(label, 0));
            Float label11 = 2 * label01 - 1; //(-1..1) label
            Float modelProb1 = MathUtils.ExpSlow(s - logZ);
            Float ls = label11 * s;
            Float datumLoss = logZ - ls;
            //Float loss2 = MathUtil.SoftMax(s-l_s, -s-l_s);

            Contracts.Check(!Float.IsNaN(datumLoss), "Unexpected NaN");

            Float mult = weight * (modelProb1 - label01);
            VectorUtils.AddMultWithOffset(ref feat, mult, ref grad, 1); // Note that 0th L-BFGS weight is for bias.
            // Add bias using this strange trick that has advantage of working well for dense and sparse arrays.
            // Due to the call to EnsureBiases, we know this region is dense.
            Contracts.Assert(grad.Count >= BiasCount && (grad.IsDense || grad.Indices[BiasCount - 1] == BiasCount - 1));
            grad.Values[0] += mult;

            return weight * datumLoss;
        }

        protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, Float loss, int numParams)
        {
            throw new NotImplementedException();
        }

        protected override void ProcessPriorDistribution(Float label, Float weight)
        {
            if (label > 0)
                _posWeight += weight;
        }

        //Override default termination criterion MeanRelativeImprovementCriterion with
        protected override Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<Float> init, out ITerminationCriterion terminationCriterion)
        {
            var opt = base.InitializeOptimizer(ch, cursorFactory, out init, out terminationCriterion);

            // MeanImprovementCriterion:
            //   Terminates when the geometrically-weighted average improvement falls below the tolerance
            //terminationCriterion = new GradientCheckingMonitor(new MeanImprovementCriterion(CmdArgs.optTol, 0.25, MaxIterations),2);
            terminationCriterion = new MeanImprovementCriterion(OptTol, (Float)0.25, MaxIterations);

            return opt;
        }

        protected override VBuffer<Float> InitializeWeightsFromPredictor(ParameterMixingCalibratedPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);

            var pred = srcPredictor.SubPredictor as LinearBinaryPredictor;
            Contracts.AssertValue(pred);
            return InitializeWeights(pred.Weights2, new[] { pred.Bias });
        }

        protected override ParameterMixingCalibratedPredictor CreatePredictor()
        {
            // Logistic regression is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so there is no need to
            // train a separate calibrator.
            VBuffer<Float> weights = default(VBuffer<Float>);
            Float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            return new ParameterMixingCalibratedPredictor(Host,
                new LinearBinaryPredictor(Host, ref weights, bias),
                new PlattCalibrator(Host, -1, 0));
        }

        [TlcModule.EntryPoint(Name = "Trainers.LogisticRegressionBinaryClassifier",
            Desc = Summary,
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/LogisticRegression/doc.xml' path='doc/members/member[@name=""LBFGS""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/LogisticRegression/doc.xml' path='doc/members/example[@name=""LogisticRegressionBinaryClassifier""]/*' />"})]

        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLRBinary");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LogisticRegression(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
