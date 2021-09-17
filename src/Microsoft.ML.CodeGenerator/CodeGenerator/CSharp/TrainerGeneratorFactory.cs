// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.AutoML;
using static Microsoft.ML.CodeGenerator.CSharp.TrainerGenerators;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal interface ITrainerGenerator
    {
        string GenerateTrainer();

        string[] GenerateUsings();
    }

    internal static class TrainerGeneratorFactory
    {
        internal static ITrainerGenerator GetInstance(PipelineNode node)
        {
            if (Enum.TryParse(node.Name, out TrainerName trainer))
            {
                switch (trainer)
                {
                    case TrainerName.LightGbmBinary:
                        return new LightGbmBinary(node);
                    case TrainerName.LightGbmMulti:
                        return new LightGbmMulti(node);
                    case TrainerName.LightGbmRegression:
                        return new LightGbmRegression(node);
                    case TrainerName.AveragedPerceptronBinary:
                        return new AveragedPerceptron(node);
                    case TrainerName.FastForestBinary:
                        return new FastForestClassification(node);
                    case TrainerName.FastForestRegression:
                        return new FastForestRegression(node);
                    case TrainerName.FastTreeBinary:
                        return new FastTreeClassification(node);
                    case TrainerName.FastTreeRegression:
                        return new FastTreeRegression(node);
                    case TrainerName.FastTreeTweedieRegression:
                        return new FastTreeTweedie(node);
                    case TrainerName.LinearSvmBinary:
                        return new LinearSvm(node);
                    case TrainerName.LbfgsLogisticRegressionBinary:
                        return new LbfgsLogisticRegressionBinary(node);
                    case TrainerName.LbfgsMaximumEntropyMulti:
                        return new LbfgsMaximumEntropyMulti(node);
                    case TrainerName.OnlineGradientDescentRegression:
                        return new OnlineGradientDescentRegression(node);
                    case TrainerName.OlsRegression:
                        return new OlsRegression(node);
                    case TrainerName.LbfgsPoissonRegression:
                        return new LbfgsPoissonRegression(node);
                    case TrainerName.SdcaLogisticRegressionBinary:
                        return new StochasticDualCoordinateAscentBinary(node);
                    case TrainerName.SdcaMaximumEntropyMulti:
                        return new StochasticDualCoordinateAscentMulti(node);
                    case TrainerName.SdcaRegression:
                        return new StochasticDualCoordinateAscentRegression(node);
                    case TrainerName.SgdCalibratedBinary:
                        return new SgdCalibratedBinary(node);
                    case TrainerName.SymbolicSgdLogisticRegressionBinary:
                        return new SymbolicSgdLogisticRegressionBinary(node);
                    case TrainerName.Ova:
                        return new OneVersusAll(node);
                    case TrainerName.ImageClassification:
                        return new ImageClassificationTrainer(node);
                    case TrainerName.MatrixFactorization:
                        return new MatrixFactorization(node);
                    case TrainerName.LightGbmRanking:
                        return new LightGbmRanking(node);
                    case TrainerName.FastTreeRanking:
                        return new FastTreeRanking(node);
                    default:
                        throw new ArgumentException($"The trainer '{trainer}' is not handled currently.");
                }
            }
            throw new ArgumentException($"The trainer '{node.Name}' is not handled currently.");
        }
    }
}
