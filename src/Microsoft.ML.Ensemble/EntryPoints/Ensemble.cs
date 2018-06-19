// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(Ensemble), null, typeof(SignatureEntryPointModule), "TrainEnsemble")]

namespace Microsoft.ML.Ensemble.EntryPoints
{
    public static class Ensemble
    {
        [TlcModule.EntryPoint(Name = "Trainers.BinaryEnsemble", Desc = "Train binary ensemble.", UserName = EnsembleTrainer.UserNameValue)]
        public static CommonOutputs.BinaryClassificationOutput CreateBinaryEnsemble(IHostEnvironment env, EnsembleTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainBinaryEnsemble");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<EnsembleTrainer.Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new EnsembleTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        [TlcModule.EntryPoint(Name = "Trainers.ClassificationEnsemble", Desc = "Train multiclass ensemble.", UserName = EnsembleTrainer.UserNameValue)]
        public static CommonOutputs.MulticlassClassificationOutput CreateMultiClassEnsemble(IHostEnvironment env, MulticlassDataPartitionEnsembleTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainMultiClassEnsemble");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<MulticlassDataPartitionEnsembleTrainer.Arguments, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new MulticlassDataPartitionEnsembleTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        [TlcModule.EntryPoint(Name = "Trainers.RegressionEnsemble", Desc = "Train regression ensemble.", UserName = EnsembleTrainer.UserNameValue)]
        public static CommonOutputs.RegressionOutput CreateRegressionEnsemble(IHostEnvironment env, RegressionEnsembleTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainRegressionEnsemble");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<RegressionEnsembleTrainer.Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new RegressionEnsembleTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}
