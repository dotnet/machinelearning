// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML
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
                { TrainerName.LbfgsLogisticRegressionBinary, typeof(LbfgsLogisticRegressionBinaryExtension) },
                { TrainerName.LbfgsMaximumEntropyMulti, typeof(LbfgsMaximumEntropyMultiExtension) },
                { TrainerName.LbfgsLogisticRegressionOva, typeof(LbfgsLogisticRegressionOvaExtension) },
                { TrainerName.OnlineGradientDescentRegression, typeof(OnlineGradientDescentRegressionExtension) },
                { TrainerName.OlsRegression, typeof(OlsRegressionExtension) },
                { TrainerName.LbfgsPoissonRegression, typeof(LbfgsPoissonRegressionExtension) },
                { TrainerName.SdcaLogisticRegressionBinary, typeof(SdcaLogisticRegressionBinaryExtension) },
                { TrainerName.SdcaMaximumEntropyMulti, typeof(SdcaMaximumEntropyMultiExtension) },
                { TrainerName.SdcaRegression, typeof(SdcaRegressionExtension) },
                { TrainerName.SgdCalibratedBinary, typeof(SgdCalibratedBinaryExtension) },
                { TrainerName.SgdCalibratedOva, typeof(SgdCalibratedOvaExtension) },
                { TrainerName.SymbolicSgdLogisticRegressionBinary, typeof(SymbolicSgdLogisticRegressionBinaryExtension) },
                { TrainerName.SymbolicSgdLogisticRegressionOva, typeof(SymbolicSgdLogisticRegressionOvaExtension) },
                { TrainerName.MatrixFactorization, typeof(MatrixFactorizationExtension) },
                { TrainerName.ImageClassification, typeof(ImageClassificationExtension) },
                { TrainerName.LightGbmRanking, typeof(LightGbmRankingExtension) },
                { TrainerName.FastTreeRanking, typeof(FastTreeRankingExtension) },
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
            IEnumerable<TrainerName> allowList, ColumnInformation columnInfo)
        {
            IEnumerable<ITrainerExtension> trainers;
            if (task == TaskKind.BinaryClassification)
            {
                trainers = GetBinaryLearners();
            }
            else if (task == TaskKind.MulticlassClassification &&
                columnInfo.ImagePathColumnNames.Count == 1 &&
                columnInfo.CategoricalColumnNames.Count == 0 &&
                columnInfo.NumericColumnNames.Count == 0 &&
                columnInfo.TextColumnNames.Count == 0)
            {
                // Image Classification case where all you have is a label column and image column.
                // This trainer takes features column of vector of bytes which will not work with any
                // other trainer.
                return new List<ITrainerExtension>() { new ImageClassificationExtension() };
            }
            else if (task == TaskKind.MulticlassClassification)
            {
                trainers = GetMultiLearners();
            }
            else if (task == TaskKind.Regression)
            {
                trainers = GetRegressionLearners();
            }
            else if (task == TaskKind.Recommendation)
            {
                trainers = GetRecommendationLearners();
            }
            else if (task == TaskKind.Ranking)
            {
                trainers = GetRankingLearners();
            }
            else
            {
                // should not be possible to reach here
                throw new NotSupportedException($"unsupported machine learning task type {task}");
            }

            if (allowList != null)
            {
                allowList = new HashSet<TrainerName>(allowList);
                trainers = trainers.Where(t => allowList.Contains(GetTrainerName(t)));
            }

            return trainers;
        }

        private static IEnumerable<ITrainerExtension> GetBinaryLearners()
        {
            return new ITrainerExtension[]
            {
                new AveragedPerceptronBinaryExtension(),
                new SdcaLogisticRegressionBinaryExtension(),
                new LightGbmBinaryExtension(),
                new SymbolicSgdLogisticRegressionBinaryExtension(),
                new LinearSvmBinaryExtension(),
                new FastTreeBinaryExtension(),
                new LbfgsLogisticRegressionBinaryExtension(),
                new FastForestBinaryExtension(),
                new SgdCalibratedBinaryExtension()
            };
        }

        private static IEnumerable<ITrainerExtension> GetMultiLearners()
        {
            return new ITrainerExtension[]
            {
                new AveragedPerceptronOvaExtension(),
                new SdcaMaximumEntropyMultiExtension(),
                new LightGbmMultiExtension(),
                new SymbolicSgdLogisticRegressionOvaExtension(),
                new FastTreeOvaExtension(),
                new LinearSvmOvaExtension(),
                new LbfgsLogisticRegressionOvaExtension(),
                new SgdCalibratedOvaExtension(),
                new FastForestOvaExtension(),
                new LbfgsMaximumEntropyMultiExtension()
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
                new LbfgsPoissonRegressionExtension(),
                new OnlineGradientDescentRegressionExtension(),
                new OlsRegressionExtension(),
            };
        }

        private static IEnumerable<ITrainerExtension> GetRecommendationLearners()
        {
            return new ITrainerExtension[]
            {
                new MatrixFactorizationExtension()
            };
        }

        private static IEnumerable<ITrainerExtension> GetRankingLearners()
        {
            return new ITrainerExtension[]
            {
                new LightGbmRankingExtension(),
                new FastTreeRankingExtension()
            };
        }
    }
}
