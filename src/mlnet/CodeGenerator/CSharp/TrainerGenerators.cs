// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal static class TrainerGenerators
    {
        internal class LightGbm : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LightGbm";

            //ClassName of the options to trainer
            internal override string OptionsName => "Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"NumLeaves","numLeaves" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"MinDataPerLeaf","minDataPerLeaf" },
                        {"LearningRate","learningRate" },
                        {"NumBoostRound","numBoostRound" },
                        {"WeightColumn","exampleWeightColumnName" }
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.LightGBM;\r\n";

            public LightGbm(PipelineNode node) : base(node)
            {
            }
        }

        internal class AveragedPerceptron : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "AveragedPerceptron";

            //ClassName of the options to trainer
            internal override string OptionsName => "AveragedPerceptronTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"LossFunction","lossFunction" },
                        {"LearningRate","learningRate" },
                        {"DecreaseLearningRate","decreaseLearningRate" },
                        {"L2RegularizerWeight","l2RegularizerWeight" },
                        {"NumberOfIterations","numIterations" }
                        };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers.Online;\r\n ";

            public AveragedPerceptron(PipelineNode node) : base(node)
            {
            }
        }

        #region FastTree
        internal abstract class FastTreeBase : TrainerGeneratorBase
        {
            internal override string Usings => "using Microsoft.ML.Trainers.FastTree;\r\n";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"LearningRate","learningRate" },
                        {"NumLeaves","numLeaves" },
                        {"NumTrees","numTrees" },
                        {"MinDatapointsInLeaves","minDatapointsInLeaves" },
                        };
                }
            }

            public FastTreeBase(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastForestClassification : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastForest";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastForestClassification.Options";

            public FastForestClassification(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastForestRegression : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastForest";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastForestRegression.Options";

            public FastForestRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastTreeClassification : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTree";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTreeBinaryClassificationTrainer.Options";

            public FastTreeClassification(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastTreeRegression : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTree";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTreeRegressionTrainer.Options";

            public FastTreeRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class FastTreeTweedie : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTreeTweedie";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTreeTweedieTrainer.Options";

            public FastTreeTweedie(PipelineNode node) : base(node)
            {
            }
        }
        #endregion

        internal class LinearSvm : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LinearSupportVectorMachines";

            //ClassName of the options to trainer
            internal override string OptionsName => "LinearSvmTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn", "exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"NumberOfIterations","numIterations" },
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers.Online;\r\n ";

            public LinearSvm(PipelineNode node) : base(node)
            {
            }
        }

        internal class LogisticRegressionBinary : TrainerGeneratorBase
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
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"L1Weight","l1Weight" },
                        {"L2Weight","l2Weight" },
                        {"OptTol","optimizationTolerance" },
                        {"MemorySize","memorySize" },
                        {"EnforceNonNegativity","enforceNoNegativity" },
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers;\r\n";

            public LogisticRegressionBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class OnlineGradientDescentRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "OnlineGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "OnlineGradientDescentTrainer.Options";

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
                        {"LabelColumn" , "labelColumnName" },
                        {"FeatureColumn" , "featureColumnName" },
                        {"LossFunction" ,"lossFunction" },

                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers.Online;\r\n";

            public OnlineGradientDescentRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class OrdinaryLeastSquaresRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "OrdinaryLeastSquares";

            //ClassName of the options to trainer
            internal override string OptionsName => "OlsLinearRegressionTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers.HalLearners;\r\n";

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
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"L1Weight","l1Weight" },
                        {"L2Weight","l2Weight" },
                        {"OptTol","optimizationTolerance" },
                        {"MemorySize","memorySize" },
                        {"EnforceNonNegativity","enforceNoNegativity" },
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers;\r\n";

            public PoissonRegression(PipelineNode node) : base(node)
            {
            }
        }

        #region SDCA
        internal abstract class StochasticDualCoordinateAscentBase : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "StochasticDualCoordinateAscent";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"Loss","loss" },
                        {"L2Const","l2Const" },
                        {"L1Threshold","l1Threshold" },
                        {"MaxIterations","maxIterations" }
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers;\r\n";

            public StochasticDualCoordinateAscentBase(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscentBinary : StochasticDualCoordinateAscentBase
        {
            //ClassName of the options to trainer
            internal override string OptionsName => "SdcaBinaryTrainer.Options";

            public StochasticDualCoordinateAscentBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscentRegression : StochasticDualCoordinateAscentBase
        {
            //ClassName of the options to trainer
            internal override string OptionsName => "SdcaRegressionTrainer.Options";

            public StochasticDualCoordinateAscentRegression(PipelineNode node) : base(node)
            {
            }
        }
        #endregion

        internal class StochasticGradientDescentClassification : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "StochasticGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "SgdBinaryTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"WeightColumn","exampleWeightColumnName" },
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"NumIterations","numIterations" },
                        {"MaxIterations","maxIterations" },
                        {"InitLearningRate","initLearningRate" },
                        {"L2Weight","l2Weight" }
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers;\r\n";

            public StochasticGradientDescentClassification(PipelineNode node) : base(node)
            {
            }
        }

        internal class SymbolicStochasticGradientDescent : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "SymbolicStochasticGradientDescent";

            //ClassName of the options to trainer
            internal override string OptionsName => "SymSgdClassificationTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"LabelColumn","labelColumnName" },
                        {"FeatureColumn","featureColumnName" },
                        {"NumberOfIterations","numberOfIterations" }
                    };
                }
            }

            internal override string Usings => "using Microsoft.ML.Trainers.HalLearners;\r\n";

            public SymbolicStochasticGradientDescent(PipelineNode node) : base(node)
            {
            }
        }

    }
}
