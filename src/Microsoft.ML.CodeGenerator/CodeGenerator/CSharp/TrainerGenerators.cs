// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Text;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal static class TrainerGenerators
    {
        internal abstract class LightGbmBase : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LightGbm";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"NumberOfLeaves","numberOfLeaves" },
                        {"LabelColumnName","labelColumnName" },
                        {"RowGroupColumnName","rowGroupColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"MinimumExampleCountPerLeaf","minimumExampleCountPerLeaf" },
                        {"LearningRate","learningRate" },
                        {"NumberOfIterations","numberOfIterations" },
                        {"ExampleWeightColumnName","exampleWeightColumnName" }
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers.LightGbm;\r\n" };

            public LightGbmBase(PipelineNode node) : base(node)
            {
            }
        }

        internal class LightGbmBinary : LightGbmBase
        {
            internal override string OptionsName => "LightGbmBinaryTrainer.Options";

            public LightGbmBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class LightGbmMulti : LightGbmBase
        {
            internal override string OptionsName => "LightGbmMulticlassTrainer.Options";

            public LightGbmMulti(PipelineNode node) : base(node)
            {
            }
        }

        internal class LightGbmRegression : LightGbmBase
        {
            internal override string OptionsName => "LightGbmRegressionTrainer.Options";

            public LightGbmRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class LightGbmRanking : LightGbmBase
        {
            internal override string OptionsName => "LightGbmRankingTrainer.Options";

            public LightGbmRanking(PipelineNode node) : base(node)
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
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"LossFunction","lossFunction" },
                        {"LearningRate","learningRate" },
                        {"DecreaseLearningRate","decreaseLearningRate" },
                        {"L2Regularization","l2Regularization" },
                        {"NumberOfIterations","numberOfIterations" }
                        };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n " };

            public AveragedPerceptron(PipelineNode node) : base(node)
            {
            }
        }

        #region FastTree
        internal abstract class FastTreeBase : TrainerGeneratorBase
        {
            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers.FastTree;\r\n" };

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"RowGroupColumnName","rowGroupColumnName" },
                        {"LearningRate","learningRate" },
                        {"NumberOfLeaves","numberOfLeaves" },
                        {"NumberOfTrees","numberOfTrees" },
                        {"MinimumExampleCountPerLeaf","minimumExampleCountPerLeaf" },
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
            internal override string OptionsName => "FastTreeBinaryTrainer.Options";

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

        internal class FastTreeRanking : FastTreeBase
        {
            //ClassName of the trainer
            internal override string MethodName => "FastTree";

            //ClassName of the options to trainer
            internal override string OptionsName => "FastTreeRankingTrainer.Options";

            public FastTreeRanking(PipelineNode node) : base(node)
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
            internal override string MethodName => "LinearSvm";

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
                        {"ExampleWeightColumnName", "exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"NumberOfIterations","numIterations" },
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n " };

            public LinearSvm(PipelineNode node) : base(node)
            {
            }
        }

        #region Logistic Regression

        internal abstract class LbfgsLogisticRegressionBase : TrainerGeneratorBase
        {
            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"L1Regularization","l1Regularization" },
                        {"L2Regularization","l2Regularization" },
                        {"OptimizationTolerance","optimizationTolerance" },
                        {"HistorySize","historySize" },
                        {"EnforceNonNegativity","enforceNonNegativity" },
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public LbfgsLogisticRegressionBase(PipelineNode node) : base(node)
            {
            }
        }
        internal class LbfgsLogisticRegressionBinary : LbfgsLogisticRegressionBase
        {
            internal override string MethodName => "LbfgsLogisticRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "LbfgsLogisticRegressionBinaryTrainer.Options";

            public LbfgsLogisticRegressionBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class LbfgsMaximumEntropyMulti : LbfgsLogisticRegressionBase
        {
            internal override string MethodName => "LbfgsMaximumEntropy";

            //ClassName of the options to trainer
            internal override string OptionsName => "LbfgsMaximumEntropyMulticlassTrainer.Options";

            public LbfgsMaximumEntropyMulti(PipelineNode node) : base(node)
            {
            }
        }
        #endregion

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
                        {"L2Regularization" , "l2Regularization" },
                        {"NumberOfIterations" , "numberOfIterations" },
                        {"LabelColumnName" , "labelColumnName" },
                        {"FeatureColumnName" , "featureColumnName" },
                        {"LossFunction" ,"lossFunction" },
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public OnlineGradientDescentRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class OlsRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "Ols";

            //ClassName of the options to trainer
            internal override string OptionsName => "OlsTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public OlsRegression(PipelineNode node) : base(node)
            {
            }
        }

        internal class LbfgsPoissonRegression : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "LbfgsPoissonRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "LbfgsPoissonRegressionTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"L1Regularization","l1Regularization" },
                        {"L2Regularization","l2Regularization" },
                        {"OptimizationTolerance","optimizationTolerance" },
                        {"HistorySize","historySize" },
                        {"EnforceNonNegativity","enforceNonNegativity" },
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public LbfgsPoissonRegression(PipelineNode node) : base(node)
            {
            }
        }

        #region SDCA
        internal abstract class StochasticDualCoordinateAscentBase : TrainerGeneratorBase
        {
            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"Loss","loss" },
                        {"L2Regularization","l2Regularization" },
                        {"L1Regularization","l1Regularization" },
                        {"MaximumNumberOfIterations","maximumNumberOfIterations" }
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public StochasticDualCoordinateAscentBase(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscentBinary : StochasticDualCoordinateAscentBase
        {
            internal override string MethodName => "SdcaLogisticRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "SdcaLogisticRegressionBinaryTrainer.Options";

            public StochasticDualCoordinateAscentBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscentMulti : StochasticDualCoordinateAscentBase
        {
            internal override string MethodName => "SdcaMaximumEntropy";

            //ClassName of the options to trainer
            internal override string OptionsName => "SdcaMaximumEntropyMulticlassTrainer.Options";

            public StochasticDualCoordinateAscentMulti(PipelineNode node) : base(node)
            {
            }
        }

        internal class StochasticDualCoordinateAscentRegression : StochasticDualCoordinateAscentBase
        {
            internal override string MethodName => "Sdca";

            //ClassName of the options to trainer
            internal override string OptionsName => "SdcaRegressionTrainer.Options";

            public StochasticDualCoordinateAscentRegression(PipelineNode node) : base(node)
            {
            }
        }
        #endregion

        internal class SgdCalibratedBinary : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "SgdCalibrated";

            //ClassName of the options to trainer
            internal override string OptionsName => "SgdCalibratedTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"ExampleWeightColumnName","exampleWeightColumnName" },
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"NumberOfIterations","numberOfIterations" },
                        {"LearningRate","learningRate" },
                        {"L2Regularization","l2Regularization" }
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public SgdCalibratedBinary(PipelineNode node) : base(node)
            {
            }
        }

        internal class SymbolicSgdLogisticRegressionBinary : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "SymbolicSgdLogisticRegression";

            //ClassName of the options to trainer
            internal override string OptionsName => "SymbolicSgdLogisticRegressionBinaryTrainer.Options";

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        {"LabelColumnName","labelColumnName" },
                        {"FeatureColumnName","featureColumnName" },
                        {"NumberOfIterations","numberOfIterations" }
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public SymbolicSgdLogisticRegressionBinary(PipelineNode node) : base(node)
            {

            }
        }

        internal class OneVersusAll : TrainerGeneratorBase
        {
            private readonly PipelineNode _node;
            private string[] _binaryTrainerUsings;

            //ClassName of the trainer
            internal override string MethodName => "OneVersusAll";

            //ClassName of the options to trainer
            internal override string OptionsName => null;

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters => null;

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public OneVersusAll(PipelineNode node) : base(node)
            {
                _node = node;
            }

            public override string GenerateTrainer()
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(MethodName);
                sb.Append("(");
                sb.Append("mlContext.BinaryClassification.Trainers."); // This is dependent on the name of the MLContext object in template.
                var trainerGenerator = TrainerGeneratorFactory.GetInstance((PipelineNode)_node.Properties["BinaryTrainer"]);
                _binaryTrainerUsings = trainerGenerator.GenerateUsings();
                sb.Append(trainerGenerator.GenerateTrainer());
                sb.Append(",");
                sb.Append("labelColumnName:");
                sb.Append("\"");
                sb.Append(_node.Properties["LabelColumnName"]);
                sb.Append("\"");
                sb.Append(")");
                return sb.ToString();
            }

            public override string[] GenerateUsings()
            {
                return _binaryTrainerUsings;
            }
        }

        internal sealed class ImageClassificationTrainer : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "ImageClassification";
            internal override string OptionsName => "ImageClassificationTrainer.Options";
            internal override string[] Usings => new string[] { "using Microsoft.ML.Vision;\r\n" };

            public ImageClassificationTrainer(PipelineNode node) : base(node)
            {
            }
            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>();
                }
            }
        }

        internal class MatrixFactorization : TrainerGeneratorBase
        {
            //ClassName of the trainer
            internal override string MethodName => "MatrixFactorization";

            internal override string OptionsName => "MatrixFactorizationTrainer.Options";
            protected override bool IncludeFeatureColumnName => false;

            //The named parameters to the trainer.
            internal override IDictionary<string, string> NamedParameters
            {
                get
                {
                    return
                    new Dictionary<string, string>()
                    {
                        { "MatrixColumnIndexColumnName","matrixColumnIndexColumnName" },
                        { "MatrixRowIndexColumnName","matrixRowIndexColumnName" },
                        { "LabelColumnName","labelColumnName" }
                    };
                }
            }

            internal override string[] Usings => new string[] { "using Microsoft.ML.Trainers;\r\n" };

            public MatrixFactorization(PipelineNode node) : base(node)
            {
            }
        }
    }
}
