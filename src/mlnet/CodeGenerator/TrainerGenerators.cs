// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI
{
    internal static class TrainerGenerators
    {
        internal class LightGbm : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LightGbm";

            //ClassName of the options to trainer
            internal override string OptionsName => "LightGbm.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"NumLeaves","numLeaves" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"MinDataPerLeaf","minDataPerLeaf" },
                        {"LearningRate","learningRate" },
                        {"NumBoostRound","numBoostRound" }
                    };
                }
            }

            public LightGbm(PipelineNode node) : base(node)
            {
            }
        }

        internal class AveragedPerceptron : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "AveragedPerceptron";

            //ClassName of the options to trainer
            internal override string OptionsName => "AveragedPerceptron.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"Weights","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"LossFunction","lossFunction" },
                        {"LearningRate","learningRate" },
                        {"DecreaseLearningRate","decreaseLearningRate" },
                        {"L2RegularizerWeight","l2RegularizerWeight" },
                        {"NumIterations","numIterations" }
                        };
                }
            }

            public AveragedPerceptron(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastForest : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastForest";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastForest.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"Weights","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"LearningRate","learningRate" },
                        {"NumLeaves","numLeaves" },
                        {"NumTrees","numTrees" },
                        {"MinDatapointsInLeaves","minDatapointsInLeaves" },
                        };
                }
            }

            public FastForest(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastTree : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTree";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTree.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"LearningRate","learningRate" },
                        {"NumLeaves","numLeaves" },
                        {"NumTrees","numTrees" },
                        {"MinDatapointsInLeaves","minDatapointsInLeaves" },
                        };
                }
            }

            public FastTree(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastTreeTweedie : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTreeTweedie";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTreeTweedie.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"Weights","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"LearningRate","learningRate" },
                        {"NumLeaves","numLeaves" },
                        {"NumTrees","numTrees" },
                        {"MinDatapointsInLeaves","minDatapointsInLeaves" },
                        };
                }
            }

            public FastTreeTweedie(PipelineNode node) : base(node)
            {
            }
        }

        internal class LinearSvm : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LinearSupportVectorMachines";

            //ClassName of the options to trainer
            internal override string OptionsName => "LinearSvm.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"InitialWeights","weightsColumn" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"NumIterations","numIterations" },
                    };
                }
            }

            public LinearSvm(PipelineNode node) : base(node)
            {
            }
        }

        internal class LogisticRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LogisticRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "LogisticRegression.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"L1Weight","l1Weight" },
                        {"L2Weight","l2Weight" },
                        {"OptTol","optimizationTolerance" },
                        {"MemorySize","memorySize" },
                        {"EnforceNoNNegativity","enforceNoNegativity" },
                    };
                }
            }

            public LogisticRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class OnlineGradientDescentRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "OnlineGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "OnlineGradientDescent.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"LearningRate" , "learningRate" },
                        {"DecreaseLearningRate" , "decreaseLearningRate" },
                        {"L2RegularizerWeight" , "l2RegularizerWeight" },
                        {"NumIterations" , "numIterations" },
                        {"LabelColumn" , "labelColumn" },
                        {"FeatureColumn" , "featureColumn" },
                        {"InitialWeights" ,"weightsColumn" },
                        {"LossFunction" ,"lossFunction" },

                    };
                }
            }

            public OnlineGradientDescentRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class OrdinaryLeastSquaresRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "OrdinaryLeastSquares";

            //ClassName of the options to trainer
            internal override string OptionsName => "OrdinaryLeastSquares.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                    };
                }
            }

            public OrdinaryLeastSquaresRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class PoissonRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "PoissonRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "PoissonRegression.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"L1Weight","l1Weight" },
                        {"L2Weight","l2Weight" },
                        {"OptTol","optimizationTolerance" },
                        {"MemorySize","memorySize" },
                        {"EnforceNoNNegativity","enforceNoNegativity" },
                    };
                }
            }

            public PoissonRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscent : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "StochasticDualCoordinateAscent";

            //ClassName of the options to trainer
            internal override string OptionsName => "StochasticDualCoordinateAscent.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"Loss","loss" },
                        {"L2Const","l2Const" },
                        {"L1Threshold","l1Threshold" },
                        {"MaxIterations","maxIterations" }
                    };
                }
            }

            public StochasticDualCoordinateAscent(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticGradientDescent : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "StochasticGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "StochasticGradientDescent.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","weights" },
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                        {"NumIterations","numIterations" },
                        {"MaxIterations","maxIterations" },
                        {"InitLearningRate","initLearningRate" },
                        {"L2Weight","l2Weight" }
                    };
                }
            }

            public StochasticGradientDescent(PipelineNode node) : base(node)
            {
            }
        }

        internal class SymbolicStochasticGradientDescent : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "SymbolicStochasticGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "SymbolicStochasticGradientDescent.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"LabelColumn","labelColumn" },
                        {"FeatureColumn","featureColumn" },
                    };
                }
            }

            public SymbolicStochasticGradientDescent(PipelineNode node) : base(node)
            {
            }
        }

    }
}
