// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto
{
    internal class TrainerExtensionCatalog
    {
        public static IEnumerable<ITrainerExtension> GetTrainers(TaskKind task, int maxIterations)
        {
            if(task == TaskKind.BinaryClassification)
            {
                return GetBinaryLearners(maxIterations);
            }
            else if (task == TaskKind.BinaryClassification)
            {
                return GetMultiLearners(maxIterations);
            }
            else if (task == TaskKind.Regression)
            {
                return GetRegressionLearners(maxIterations);
            }
            else
            {
                // should not be possible to reach here
                throw new NotSupportedException($"unsupported machine learning task type {task}");
            }
        }

        private static IEnumerable<ITrainerExtension> GetBinaryLearners(int maxIterations)
        {
            var learners = new List<ITrainerExtension>()
            {
                new AveragedPerceptronBinaryExtension(),
                new SdcaBinaryExtension(),
                new LightGbmBinaryExtension(),
                new SymSgdBinaryExtension()
            };

            if(maxIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new LinearSvmBinaryExtension(),
                new FastTreeBinaryExtension()
            });

            if(maxIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new LogisticRegressionBinaryExtension(),
                new FastForestBinaryExtension(),
                new SgdBinaryExtension()
            });

            return learners;
        }

        private static IEnumerable<ITrainerExtension> GetMultiLearners(int maxIterations)
        {
            var learners = new List<ITrainerExtension>()
            {
                new AveragedPerceptronOvaExtension(),
                new SdcaMultiExtension(),
                new LightGbmMultiExtension(),
                new SymSgdOvaExtension()
            };

            if (maxIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new FastTreeOvaExtension(),
                new LinearSvmOvaExtension(),
                new LogisticRegressionOvaExtension()
            });

            if (maxIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new SgdOvaExtension(),
                new FastForestOvaExtension(),
                new LogisticRegressionMultiExtension(),
            });

            return learners;
        }

        private static IEnumerable<ITrainerExtension> GetRegressionLearners(int maxIterations)
        {
            var learners = new List<ITrainerExtension>()
            {
                new SdcaRegressionExtension(),
                new LightGbmRegressionExtension(),
                new FastTreeRegressionExtension(),
            };

            if(maxIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[]
            {
                new FastTreeTweedieRegressionExtension(),
                new FastForestRegressionExtension(),
            });

            if(maxIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new PoissonRegressionExtension(),
                new OnlineGradientDescentRegressionExtension(),
                new OrdinaryLeastSquaresRegressionExtension()
            });

            return learners;
        }
    }
}
