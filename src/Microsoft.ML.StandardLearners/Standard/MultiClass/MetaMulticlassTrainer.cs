// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Runtime.Learners
{
    using TScalarTrainer = ITrainer<IPredictorProducing<Float>>;

    public abstract class MetaMulticlassTrainer<TPred, TArgs> : TrainerBase<TPred>
        where TPred : IPredictor
        where TArgs : MetaMulticlassTrainer<TPred, TArgs>.ArgumentsBase
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 1)]
            [TGUI(Label = "Predictor Type", Description = "Type of underlying binary predictor")]
            public SubComponent<TScalarTrainer, SignatureBinaryClassifierTrainer> PredictorType =
                new SubComponent<TScalarTrainer, SignatureBinaryClassifierTrainer>(LinearSvm.LoadNameValue);

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>")]
            public SubComponent<ICalibratorTrainer, SignatureCalibrator> Calibrator = new SubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Whether to treat missing labels as having negative labels, instead of keeping them missing", ShortName = "missNeg")]
            public bool ImputeMissingLabelsAsNegative;
        }

        protected readonly TArgs Args;
        private TScalarTrainer _trainer;

        public sealed override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;
        public sealed override bool NeedNormalization { get; }
        public sealed override bool NeedCalibration => false;

        // No matter what the internal predictor, we're performing many passes
        // simply by virtue of this being a meta-trainer.
        public sealed override bool WantCaching => true;

        internal MetaMulticlassTrainer(IHostEnvironment env, TArgs args, string name)
            : base(env, name)
        {
            Host.CheckValue(args, nameof(args));
            Args = args;
            Host.CheckUserArg(Args.PredictorType.IsGood(), nameof(Args.PredictorType));
            // Create the first trainer so errors in the args surface early.
            _trainer = Args.PredictorType.CreateInstance(Host);
            var ex = _trainer as ITrainerEx;
            NeedNormalization = ex != null && ex.NeedNormalization;
        }

        protected IDataView MapLabelsCore<T>(ColumnType type, RefPredicate<T> equalsTarget, RoleMappedData data, string dstName)
        {
            Host.AssertValue(type);
            Host.Assert(type.RawType == typeof(T));
            Host.AssertValue(equalsTarget);
            Host.AssertValue(data);
            Host.AssertValue(data.Schema.Label);
            Host.AssertNonWhiteSpace(dstName);

            var lab = data.Schema.Label;

            RefPredicate<T> isMissing;
            if (!Args.ImputeMissingLabelsAsNegative && Conversions.Instance.TryGetIsNAPredicate(type, out isMissing))
            {
                return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                    lab.Name, dstName, type, NumberType.Float,
                    (ref T src, ref Float dst) =>
                        dst = equalsTarget(ref src) ? 1 : (isMissing(ref src) ? Float.NaN : default(Float)));
            }
            return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                lab.Name, dstName, type, NumberType.Float,
                (ref T src, ref Float dst) =>
                    dst = equalsTarget(ref src) ? 1 : default(Float));
        }

        protected TScalarTrainer GetTrainer()
        {
            // We may have instantiated the first trainer to use already, from the constructor.
            // If so capture it and set the retained trainer to null; otherwise create a new one.
            var train = _trainer ?? Args.PredictorType.CreateInstance(Host);
            _trainer = null;
            return train;
        }

        protected abstract TPred TrainCore(IChannel ch, RoleMappedData data, int count);

        public override TPred Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.Train;

            data.CheckFeatureFloatVector();

            int count;
            data.CheckMultiClassLabel(out count);
            Host.Assert(count > 0);

            using (var ch = Host.Start("Training"))
            {
                var pred = TrainCore(ch, data, count);
                ch.Check(pred != null, "Training did not result in a predictor");
                ch.Done();
                return pred;
            }
        }
    }
}
