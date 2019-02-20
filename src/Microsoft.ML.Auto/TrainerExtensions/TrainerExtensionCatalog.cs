// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal class TrainerExtensionCatalog
    {
        private static readonly IDictionary<TrainerName, Type> _trainerNamesToExtensionTypes =
            new Dictionary<TrainerName, Type>()
            {
                { TrainerName.AveragedPerceptronBinary, typeof(AveragedPerceptronBinaryExtension) },
                { TrainerName.AveragedPerceptronOva, typeof(AveragedPerceptronOvaExtension) },
                { TrainerName.FastForestBinary, typeof(FastForestBinaryExtension) },
                { TrainerName.FastForestOva, typeof(FastForestOvaExtension) },
                { TrainerName.FastForestRegression, typeof(FastForestRegressionExtension) },
                { TrainerName.FastTreeBinary, typeof(FastTreeBinaryExtension) },
                { TrainerName.FastTreeOva, typeof(FastTreeOvaExtension) },
                { TrainerName.FastTreeRegression, typeof(FastTreeRegressionExtension) },
                { TrainerName.FastTreeTweedieRegression, typeof(FastTreeTweedieRegressionExtension) },
                { TrainerName.LightGbmBinary, typeof(LightGbmBinaryExtension) },
                { TrainerName.LightGbmMulti, typeof(LightGbmMultiExtension) },
                { TrainerName.LightGbmRegression, typeof(LightGbmRegressionExtension) },
                { TrainerName.LinearSvmBinary, typeof(LinearSvmBinaryExtension) },
                { TrainerName.LinearSvmOva, typeof(LinearSvmOvaExtension) },
                { TrainerName.LogisticRegressionBinary, typeof(LogisticRegressionBinaryExtension) },
                { TrainerName.LogisticRegressionMulti, typeof(LogisticRegressionMultiExtension) },
                { TrainerName.LogisticRegressionOva, typeof(LogisticRegressionOvaExtension) },
                { TrainerName.OnlineGradientDescentRegression, typeof(OnlineGradientDescentRegressionExtension) },
                { TrainerName.OrdinaryLeastSquaresRegression, typeof(OrdinaryLeastSquaresRegressionExtension) },
                { TrainerName.PoissonRegression, typeof(PoissonRegressionExtension) },
                { TrainerName.SdcaBinary, typeof(SdcaBinaryExtension) },
                { TrainerName.SdcaMulti, typeof(SdcaMultiExtension) },
                { TrainerName.SdcaRegression, typeof(SdcaRegressionExtension) },
                { TrainerName.StochasticGradientDescentBinary, typeof(SgdBinaryExtension) },
                { TrainerName.StochasticGradientDescentOva, typeof(SgdOvaExtension) },
                { TrainerName.SymSgdBinary, typeof(SymSgdBinaryExtension) },
                { TrainerName.SymSgdOva, typeof(SymSgdOvaExtension) }
            };

        private static readonly IDictionary<Type, TrainerName> _extensionTypesToTrainerNames =
            _trainerNamesToExtensionTypes.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        public static TrainerName GetTrainerName(ITrainerExtension trainerExtension)
        {
            return _extensionTypesToTrainerNames[trainerExtension.GetType()];
        }

        public static ITrainerExtension GetTrainerExtension(TrainerName trainerName)
        {
            var trainerExtensionType = _trainerNamesToExtensionTypes[trainerName];
            return (ITrainerExtension)Activator.CreateInstance(trainerExtensionType);
        }

        public static IEnumerable<ITrainerExtension> GetTrainers(TaskKind task,
            IEnumerable<TrainerName> whitelist)
        {
            IEnumerable<ITrainerExtension> trainers;
            if (task == TaskKind.BinaryClassification)
            {
                trainers = GetBinaryLearners();
            }
            else if (task == TaskKind.MulticlassClassification)
            {
                trainers = GetMultiLearners();
            }
            else if (task == TaskKind.Regression)
            {
                trainers = GetRegressionLearners();
            }
            else
            {
                // should not be possible to reach here
                throw new NotSupportedException($"unsupported machine learning task type {task}");
            }

            if (whitelist != null)
            {
                whitelist = new HashSet<TrainerName>(whitelist);
                trainers = trainers.Where(t => whitelist.Contains(GetTrainerName(t)));
            }

            return trainers;
        }

        private static IEnumerable<ITrainerExtension> GetBinaryLearners()
        {
            return new ITrainerExtension[]
            {
                new AveragedPerceptronBinaryExtension(),
                new SdcaBinaryExtension(),
                new LightGbmBinaryExtension(),
                new SymSgdBinaryExtension(),
                new LinearSvmBinaryExtension(),
                new FastTreeBinaryExtension(),
                new LogisticRegressionBinaryExtension(),
                new FastForestBinaryExtension(),
                new SgdBinaryExtension()
            };
        }

        private static IEnumerable<ITrainerExtension> GetMultiLearners()
        {
            return new ITrainerExtension[]
            {
                new AveragedPerceptronOvaExtension(),
                new SdcaMultiExtension(),
                new LightGbmMultiExtension(),
                new SymSgdOvaExtension(),
                new FastTreeOvaExtension(),
                new LinearSvmOvaExtension(),
                new LogisticRegressionOvaExtension(),
                new SgdOvaExtension(),
                new FastForestOvaExtension(),
                new LogisticRegressionMultiExtension()
            };
        }

        private static IEnumerable<ITrainerExtension> GetRegressionLearners()
        {
            return new ITrainerExtension[]
            {
                new SdcaRegressionExtension(),
                new LightGbmRegressionExtension(),
                new FastTreeRegressionExtension(),
                new FastTreeTweedieRegressionExtension(),
                new FastForestRegressionExtension(),
                new PoissonRegressionExtension(),
                new OnlineGradientDescentRegressionExtension(),
                new OrdinaryLeastSquaresRegressionExtension(),
            };
        }
    }
}
