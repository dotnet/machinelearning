// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Auto;
using static Microsoft.ML.CLI.TrainerGenerators;

namespace Microsoft.ML.CLI
{
    internal interface ITrainerGenerator
    {
        string GenerateTrainer();
        string GenerateUsings();
    }
    internal static class TrainerGeneratorFactory
    {
        internal static ITrainerGenerator GetInstance(Pipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));
            var node = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer).First();
            if (node == null)
                return null;
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
                    case TrainerName.LogisticRegressionBinary:
                        return new LogisticRegressionBinary(node);
                    case TrainerName.OnlineGradientDescentRegression:
                        return new OnlineGradientDescentRegression(node);
                    case TrainerName.OrdinaryLeastSquaresRegression:
                        return new OrdinaryLeastSquaresRegression(node);
                    case TrainerName.PoissonRegression:
                        return new PoissonRegression(node);
                    case TrainerName.SdcaBinary:
                        return new StochasticDualCoordinateAscentBinary(node);
                    case TrainerName.SdcaRegression:
                        return new StochasticDualCoordinateAscentRegression(node);
                    case TrainerName.StochasticGradientDescentBinary:
                        return new StochasticGradientDescentClassification(node);
                    case TrainerName.SymSgdBinary:
                        return new SymbolicStochasticGradientDescent(node);
                    default:
                        return null;
                }
            }
            return null;
        }
    }
}
