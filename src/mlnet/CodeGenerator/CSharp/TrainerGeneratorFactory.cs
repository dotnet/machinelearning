// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Auto;
using static Microsoft.ML.CLI.CodeGenerator.CSharp.TrainerGenerators;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
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
                    case TrainerName.LightGbmMulti:
                    case TrainerName.LightGbmRegression:
                        return new LightGbm(node);
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
                        return new LogisticRegressionBinary(node);
                    case TrainerName.LbfgsMaximumEntropyMulti:
                        return new LogisticRegressionMulti(node);
                    case TrainerName.OnlineGradientDescentRegression:
                        return new OnlineGradientDescentRegression(node);
                    case TrainerName.OlsRegression:
                        return new OrdinaryLeastSquaresRegression(node);
                    case TrainerName.LbfgsPoissonRegression:
                        return new PoissonRegression(node);
                    case TrainerName.SdcaLogisticRegressionBinary:
                        return new StochasticDualCoordinateAscentBinary(node);
                    case TrainerName.SdcaMaximumEntropyMulti:
                        return new StochasticDualCoordinateAscentMulti(node);
                    case TrainerName.SdcaRegression:
                        return new StochasticDualCoordinateAscentRegression(node);
                    case TrainerName.SgdCalibratedBinary:
                        return new StochasticGradientDescentClassification(node);
                    case TrainerName.SymbolicSgdLogisticRegressionBinary:
                        return new SymbolicStochasticGradientDescent(node);
                    case TrainerName.Ova:
                        return new OneVersusAll(node);
                    default:
                        throw new ArgumentException($"The trainer '{trainer}' is not handled currently.");
                }
            }
            throw new ArgumentException($"The trainer '{node.Name}' is not handled currently.");
        }
    }
}
