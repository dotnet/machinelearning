// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.FastTree;

namespace Microsoft.ML.Runtime.RunTests
{
    //=========================== Binary classifiers ====================
    public class PredictorAndArgs
    {
        public SubComponent Trainer { get; set; }
        public SubComponent Scorer { get; set; }
        public SubComponent Tester { get; set; }
        public string Tag { get; set; }
        public string[] MamlArgs;

        // Whether to write the progress to the output console file or not.
        public bool BaselineProgress { get; set; }

        public PredictorAndArgs()
        {
        }

        public PredictorAndArgs(SubComponent trainer, string tag = null)
        {
            Trainer = trainer;
            Tag = tag;
        }
    }

    public /*static*/ class TestLearnersBase
    {
        // This ensures that the needed assemblies are loaded!
        static TestLearnersBase()
        {
            bool ok = true;
            //ok &= typeof(BinaryNeuralNetwork) != null;
            ok &= typeof(FastTreeBinaryClassificationTrainer) != null;
            //ok &= typeof(OneClassSvmTrainer) != null;
            //ok &= typeof(LDSvmTrainer) != null;
            Contracts.Check(ok, "Missing assemblies!");
        }

        // New.
        public static PredictorAndArgs binarySdca = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCA", "maxIterations=5 checkFreq=9 nt=1"),
            Tag = "BinarySDCA"
        };

        // New.
        public static PredictorAndArgs binarySdcaL1 = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCA", "l2=1e-06 l1=0.5 maxIterations=5 checkFreq=9 nt=1"),
            Tag = "BinarySDCA-L1"
        };

        // New.
        public static PredictorAndArgs binarySdcaSmoothedHinge = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCA", "l2=1e-06 loss=SmoothedHinge l1=0.5 maxIterations=5 checkFreq=9 nt=1"),
            Tag = "BinarySDCA-SmoothedHinge"
        };

        // New.
        public static PredictorAndArgs binarySgd = new PredictorAndArgs
        {
            Trainer = new SubComponent("SGD", "maxIterations=2 checkFreq=9 nt=1"),
            Tag = "BinarySGD"
        };

        // New.
        public static PredictorAndArgs binarySgdHinge = new PredictorAndArgs
        {
            Trainer = new SubComponent("SGD", "loss=Hinge maxIterations=2 checkFreq=9 nt=1"),
            Tag = "BinarySGD-Hinge"
        };

        // New.
        public static PredictorAndArgs multiclassSdca = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAMC", "maxIterations=20 checkFreq=0 nt=1"),
            Tag = "MultiClassSDCA"
        };

        // New.
        public static PredictorAndArgs multiclassSdcaL1 = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAMC", "l2=1e-04 l1=0.5 maxIterations=20 checkFreq=0 nt=1"),
            Tag = "MultiClassSDCA-L1"
        };

        // New.
        public static PredictorAndArgs multiclassSdcaSmoothedHinge = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAMC", "l2=1e-04 loss=SmoothedHinge maxIterations=20 checkFreq=0 nt=1"),
            Tag = "MultiClassSDCA-SmoothedHinge"
        };

        // Old.
        public static PredictorAndArgs perceptron_tlOld = new PredictorAndArgs
        {
            Trainer = new SubComponent("AveragedPerceptron", "lr=0.01 iter=100 numnorm=500 lazy+"),
        };

        public static PredictorAndArgs perceptron = new PredictorAndArgs
        {
            Trainer = new SubComponent("AveragedPerceptron", "lr=0.01 iter=100 lazy+"),
        };

        // New.
        public static PredictorAndArgs perceptron_reg = new PredictorAndArgs
        {
            Trainer = new SubComponent("AveragedPerceptron", "lr=0.01 iter=100 lazy+ reg=0.002"),
            Tag = "AveragedPerceptron-Reg"
        };

        // Old.
        public static PredictorAndArgs perceptronDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent("AveragedPerceptron"),
        };

        // Old.
        public static PredictorAndArgs perceptronNotNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("AveragedPerceptron", "norm={}"),
        };

        // New and Old.
        public static PredictorAndArgs linearSVM = new PredictorAndArgs
        {
            Trainer = new SubComponent("LinearSVM", "iter=100 lambda=0.03"),
        };

        // Old.
        public static PredictorAndArgs linearSVMNotNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("LinearSVM", "iter=100 lambda=0.03 norm={}"),
        };

        // Old.
        public static PredictorAndArgs linearSVMNotNormOne = new PredictorAndArgs
        {
            Trainer = new SubComponent("LinearSVM", "iter=1 lambda=0.03 norm={}"),
        };

        // Old.
        public static PredictorAndArgs logisticRegression_tlOld = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 norm={} nt=1"),
        };

        // New.
        public static PredictorAndArgs logisticRegression = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 nt=1"),
            MamlArgs = new[] { "norm=no" },
            BaselineProgress = true
        };

        // New.
        public static PredictorAndArgs logisticRegressionNonNegative = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-4 nt=1 nn=+"),
            MamlArgs = new[] { "norm=no" },
            Tag = "LogisticRegression-non-negative",
            BaselineProgress = true
        };

        // New.
        public static PredictorAndArgs logisticRegressionNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 nt=1"),
            Tag = "LogisticRegression-norm",
            BaselineProgress = true
        };

        // New.
        public static PredictorAndArgs logisticRegressionBinNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 nt=1"),
            MamlArgs = new[] { "xf=BinNormalizer{col=Features numBins=5}" },
            Tag = "LogisticRegression-bin-norm",
            BaselineProgress = true
        };

        // Old.
        public static PredictorAndArgs logisticRegressionSGD = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 sgd=0.1 nt=1"),
            Tag = "SGD",
        };

        // New.
        public static PredictorAndArgs logisticRegressionGaussianNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("LogisticRegression", "l1=1.0 l2=0.1 ot=1e-3 nt=1"),
            MamlArgs = new[] { "xf=MeanVarNormalizer{col=Features}" },
            Tag = "LogisticRegression-GaussianNorm",
            BaselineProgress = true
        };

        // New.
        public static PredictorAndArgs multiclassLogisticRegression = new PredictorAndArgs
        {
            Trainer = new SubComponent("MulticlassLogisticRegression", "l1=0.001 l2=0.1 ot=1e-3 nt=1"),
            MamlArgs = new[] { "norm=no" },
            BaselineProgress = true,
        };

        // New.
        public static PredictorAndArgs multiclassLogisticRegressionNonNegative = new PredictorAndArgs
        {
            Trainer = new SubComponent("MulticlassLogisticRegression", "l1=0.001 l2=0.1 ot=1e-3 nt=1 nn=+"),
            MamlArgs = new[] { "norm=no" },
            Tag = "LogisticRegression-Non-Negative",
            BaselineProgress = true,
        };

        // Old.
        public static PredictorAndArgs multiclassLogisticRegressionRegularized = new PredictorAndArgs
        {
            Trainer = new SubComponent("MulticlassLogisticRegression", "nt=1"),
            BaselineProgress = true,
        };

        // New.
        public static PredictorAndArgs FastForestClassification = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastForestClassification", "nl=5 mil=10 iter=10"),
            BaselineProgress = true,
        };

        // New and Old.
        public static PredictorAndArgs fastRankClassification = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRank", "nl=5 mil=5 lr=0.25 iter=20"),
            BaselineProgress = true,
        };

        // New.
        public static PredictorAndArgs fastRankClassificationPruning = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRank", "nl=5 mil=5 lr=0.25 iter=20 pruning=+"),
            Tag = "FastRank-Pruning",
            BaselineProgress = true,
        };

        public static PredictorAndArgs fastRankClassificationWeighted = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRank", "nl=5 mil=20 lr=0.25 iter=20"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeClassfier = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "nl=5 mil=5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTree",
            BaselineProgress = true,
        };

        public static PredictorAndArgs LightGBMClassifier = new PredictorAndArgs
        {
            Trainer = new SubComponent("LightGBMBinary", "nl=5 mil=5 lr=0.25 iter=20 mb=255"),
            Tag = "LightGBM",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeWithCategoricalClassfier = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "cat=+ nl=5 mil=5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTreeCategorical",
            BaselineProgress = true
        };

        public static PredictorAndArgs FastTreeWithCategoricalClassfierDisk = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "cat=+ nl=5 mil=5 lr=0.25 iter=20 mb=255 dt+"),
            Tag = "FastTreeCategoricalDisk",
            BaselineProgress = true
        };

        public static PredictorAndArgs FastTreeClassfierHighMinDocs = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "mil=10000 iter=5"),
            Tag = "FastTreeHighMinDocs",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeClassfierDisk = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "nl=5 mil=5 lr=0.25 iter=20 mb=255 dt+"),
            Tag = "FastTreeDisk",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeDropoutClassfier = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "nl=5 mil=5 tdrop=0.5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTreeDrop",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeBsrClassfier = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeBinaryClassification", "nl=5 mil=5 bsr+ lr=0.25 iter=20 mb=255"),
            Tag = "FastTreeBsr",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeRanker = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRanking", "iter=10 nl=10 mil=32 lr=0.30 mb=255"),
            Tag = "FastTree",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeRankerCustomGains = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRanking", "iter=10 nl=10 mil=32 lr=0.30 mb=255 gains=0,2,3,5,6"),
            MamlArgs = new[] { "eval=Ranking{gains=0,2,3,5,6}" },
            Tag = "FastTreeCustomGains",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeRegressor = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRegression", "nl=5 mil=5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTree",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeRegressorCategorical = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRegression", "cat=+ nl=5 mil=5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTreeCategorical",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeUnderbuiltRegressor = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRegression", "nl=30 mil=30 lr=0.25 iter=10 mb=255"),
            Tag = "FastTree",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeDropoutRegressor = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeRegression", "nl=5 mil=5 tdrop=0.5 lr=0.25 iter=20 mb=255"),
            Tag = "FastTreeDrop",
            BaselineProgress = true,
        };

        public static PredictorAndArgs FastTreeTweedieRegressor = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastTreeTweedieRegression", "nl=5 mil=5 lr=0.25 iter=10 mb=255 index=1.5"),
            Tag = "FastTreeTweedie",
            MamlArgs = new[] { "Eval=regression{loss=tweedie{index=1.5}}" },
            BaselineProgress = true,
        };

        public static PredictorAndArgs RegressionGamTrainer = new PredictorAndArgs
        {
            Trainer = new SubComponent("RegressionGamTrainer", ""),
            Tag = "RegressionGamTrainer",
            BaselineProgress = true,
        };

        public static PredictorAndArgs BinaryClassificationGamTrainer = new PredictorAndArgs
        {
            Trainer = new SubComponent("BinaryClassificationGamTrainer", ""),
            Tag = "BinaryClassificationGamTrainer",
            BaselineProgress = true,
        };

        public static PredictorAndArgs BinaryClassificationGamTrainerDiskTranspose = new PredictorAndArgs
        {
            Trainer = new SubComponent("BinaryClassificationGamTrainer", "dt+"),
            Tag = "BinaryClassificationGamTrainerDiskTranspose",
            BaselineProgress = true,
        };

        // New.
        public static PredictorAndArgs QuantileRegressionScorer = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastForestRegression", "nl=5 mil=5 iter=20"),
            Scorer = new SubComponent("QuantileRegression", "quantiles = 0.25,0.5,0.75"),
            Tag = "QuantileRegressorTester",
            BaselineProgress = true,
        };

        // New.
        public static PredictorAndArgs FastForestRegression = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastForestRegression", "nl=5 mil=5 iter=20"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs fastRankRegression = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRankRegression", "nl=5 mil=5 lr=0.25 iter=20"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs fastRankRegressionWeighted = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRankRegression", "nl=5 mil=20 lr=0.25 iter=20"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs OGD = new PredictorAndArgs
        {
            Trainer = new SubComponent("OGD", "iter=100"),
            MamlArgs = new[] { "xf=MinMax{col=Features zero-}" }
        };

        public static PredictorAndArgs Ols = new PredictorAndArgs
        {
            Trainer = new SubComponent("OLS"),
            MamlArgs = new[] { "norm=no" },
        };

        public static PredictorAndArgs OlsNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("OLS"),
            MamlArgs = new[] { "xf=MinMax{col=Features}" },
            Tag = "OLSNorm"
        };

        public static PredictorAndArgs OlsReg = new PredictorAndArgs
        {
            Trainer = new SubComponent("OLS", "l2=0.1"),
            MamlArgs = new[] { "norm=no" },
            Tag = "OLSReg"
        };

        public static PredictorAndArgs Sdcar = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAR", "nt=1"),
            Tag = "SDCAR"
        };

        public static PredictorAndArgs SdcarNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAR", "l2=0.01 l1=0 iter=10 checkFreq=-1 nt=1"),
            Tag = "SDCARNorm"
        };

        public static PredictorAndArgs SdcarReg = new PredictorAndArgs
        {
            Trainer = new SubComponent("SDCAR", "l2=0.1 l1=0 iter=10 checkFreq=-1 nt=1"),
            Tag = "SDCARReg"
        };

        public static PredictorAndArgs fastRankRanking = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRankRanking", "iter=10 nl=10 mil=32 lr=0.30"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs fastRankRankingWeighted = new PredictorAndArgs
        {
            Trainer = new SubComponent("FastRankRanking", "iter=10 nl=10 mil=32 lr=0.30"),
            BaselineProgress = true,
        };

        public static PredictorAndArgs poissonRegression = new PredictorAndArgs
        {
            Trainer = new SubComponent("PoissonRegression", "l1=1.0 ot=1e-2 nt=1"),
            Tester = new SubComponent("Regression", "loss=PoissonLoss"),
            MamlArgs = new[] { "xf=MinMax{col=Features zero-}" },
            BaselineProgress = true
        };

        public static PredictorAndArgs poissonRegressionNonNegative = new PredictorAndArgs
        {
            Trainer = new SubComponent("PoissonRegression", "l1=1.0 ot=1e-2 nt=1 nn=+"),
            Tester = new SubComponent("Regression", "loss=PoissonLoss"),
            MamlArgs = new[] { "xf=MinMax{col=Features zero-}" },
            Tag = "PoissonRegression-Non-Negative",
            BaselineProgress = true
        };

        public static PredictorAndArgs NnBinDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent("BinaryNeuralNetwork", "accel=sse"),
            Tag = "Default",
        };

        public static PredictorAndArgs NnBinNoNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("BinaryNeuralNetwork", "accel=sse"),
            MamlArgs = new[] { "norm=no" },
            Tag = "NoNorm",
        };

        public static PredictorAndArgs NnBinMomentum = new PredictorAndArgs
        {
            Trainer = new SubComponent("BinaryNeuralNetwork", "momentum=0.5 accel=sse"),
            Tag = "Momentum",
        };

        public static PredictorAndArgs NnBinCustom(string path)
        {
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("BinaryNeuralNetwork", "filename={" + path + "} lr=0.01 accel=sse"),
                Tag = "Custom",
            };
        }
        public static PredictorAndArgs NnBinInline(string netDefinition)
        {
            var quoted = new StringBuilder();
            CmdQuoter.QuoteValue(netDefinition, quoted);
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("BinaryNeuralNetwork", "net=" + quoted + " accel=sse"),
                Tag = "Inline",
            };
        }

        public static PredictorAndArgs NnMultiDefault(int numOutputs)
        {
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("MultiClassNeuralNetwork", string.Format("lr=0.1 accel=sse output={0}", numOutputs)),
                Tag = "Default",
            };
        }

        public static PredictorAndArgs NnMultiCustom(string path)
        {
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("MultiClassNeuralNetwork", "filename={" + path + "} lr=0.1 accel=sse"),
                Tag = "Default",
            };
        }

        public static PredictorAndArgs NnMultiCustomLred(string path)
        {
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("MultiClassNeuralNetwork", "filename={" + path + "} lr=0.1 lred=0.9 lredfreq=5 lrederror=0.02 accel=sse"),
                Tag = "Lred",
            };
        }

        public static PredictorAndArgs NnMultiMomentum(int numOutputs, bool useNag = false)
        {
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("MultiClassNeuralNetwork",
                        string.Format("lr=0.1 momentum=0.5 accel=sse output={0} nag{1}", numOutputs, useNag ? "+" : "-")),
                Tag = "Momentum" + (useNag ? "Nag" : string.Empty),
            };
        }

        public static PredictorAndArgs DssmDefault(int qryFeaturesCount, int docFeaturesCount, int negativeDocsCount, int numIterations, Float gamma)
        {
            string settings = string.Format("qfeats={0} dfeats={1} negdocs={2} iter={3} gamma={4} accel=sse",
                qryFeaturesCount, docFeaturesCount, negativeDocsCount, numIterations, gamma);
            return new PredictorAndArgs
            {
                Trainer = new SubComponent("DssmNet", settings),
                Scorer = new SubComponent("MultiClass"),
                Tag = "DefaultDssm",
            };
        }

        public static PredictorAndArgs OneClassSvmRbf = new PredictorAndArgs
        {
            Trainer = new SubComponent("OneClassSVM"),
        };

        public static PredictorAndArgs OneClassSvmSigmoid = new PredictorAndArgs
        {
            Trainer = new SubComponent("OneClassSVM", "ker=SigmoidKernel {g=0.005}"),
        };

        public static PredictorAndArgs OneClassSvmLinear = new PredictorAndArgs
        {
            Trainer = new SubComponent("OneClassSVM", "ker=LinearKernel"),
        };

        public static PredictorAndArgs OneClassSvmPoly = new PredictorAndArgs
        {
            Trainer = new SubComponent("OneClassSVM", "ker=PolynomialKernel {b=1}"),
        };

        public static PredictorAndArgs PCAAnomalyDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent("pcaAnomaly"),
            Tag = "Default"
        };

        public static PredictorAndArgs PCAAnomalyNoNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("pcaAnomaly"),
            MamlArgs = new[] { "norm=no" },
            Tag = "NoNorm"
        };

        public static PredictorAndArgs LDSVMDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent("LDSVM", "iter=1000"),
            Tag = "LDSVM-def"
        };

        public static PredictorAndArgs LDSVMNoBias = new PredictorAndArgs
        {
            Trainer = new SubComponent("LDSVM", "iter=1000 noBias=+"),
            Tag = "LDSVM-nob"
        };

        public static PredictorAndArgs LDSvmNoNorm = new PredictorAndArgs
        {
            Trainer = new SubComponent("LDSVM", "iter=1000"),
            MamlArgs = new[] { "norm=no" },
            Tag = "LDSVM-non"
        };

        public static PredictorAndArgs LDSvmNoCalib = new PredictorAndArgs
        {
            Trainer = new SubComponent("LDSVM", "iter=1000"),
            MamlArgs = new[] { "cali={}" },
            Tag = "LDSVM-noc"
        };

        public static PredictorAndArgs KMeansDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent("KM", "nt=1"),
            Tag = "KMeans-def"
        };

        public static PredictorAndArgs KMeansDefaultNoNormalizer = new PredictorAndArgs
        {
            Trainer = new SubComponent("KM", "norm={} nt=1"),
            Tag = "KMeans-def-nonorm"
        };

        public static PredictorAndArgs KMeansInitPlusPlus = new PredictorAndArgs
        {
            Trainer = new SubComponent("KM", "nt=1 init=KMeansPlusPlus"),
            Tag = "KMeans-initplusplus"
        };

        public static PredictorAndArgs KMeansInitRandom = new PredictorAndArgs
        {
            Trainer = new SubComponent("KM", "nt=1 init=Random"),
            Tag = "KMeans-initrandom"
        };

        public static string BinaryTrembleTrainer = "BinaryTremble";
        public static string MultiClassTrembleTrainer = "MultiClassTremble";

        public static PredictorAndArgs BinaryTrembleDecisionTreeLR = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, "inp=SingleFeaturePredictor inp=LogisticRegression{maxiter=10 quiet=+ t-} lnp=LogisticRegression{maxiter=100 quiet=+ t-} nl=8"),
            Tag = "BinaryTremble-lr"
        };

        public static PredictorAndArgs BinaryDecisionTreeDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, ""),
            Tag = "BinaryDT-def"
        };

        public static PredictorAndArgs BinaryDecisionTreePruning = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, "prune=+"),
            Tag = "BinaryDT-prune"
        };

        public static PredictorAndArgs BinaryDecisionTreeGini = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, "imp=Gini"),
            Tag = "BinaryDT-gini"
        };

        public static PredictorAndArgs BinaryDecisionTreeModified = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, "mil=2 inp=SingleFeaturePredictor{nb=FIFTEEN ff=0.7} nl=40"),
            Tag = "BinaryDT-mod"
        };

        public static PredictorAndArgs BinaryDecisionTreeRewt = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryTrembleTrainer, "mil=2 inp=SingleFeaturePredictor{nb=FIFTEEN ff=0.7} nl=40 rewt=+"),
            Tag = "BinaryDT-rewt"
        };

        public static PredictorAndArgs MultiClassTrembleDecisionTreeLR = new PredictorAndArgs
        {
            Trainer = new SubComponent(MultiClassTrembleTrainer, "inp=SingleFeaturePredictor inp=LogisticRegression{maxiter=10 quiet=+ t-} lnp=MultiClassLogisticRegression{maxiter=100 quiet=+ t-} nl=8"),
            Tag = "MultiClassTremble-lr"
        };

        public static PredictorAndArgs MultiClassDecisionTreeDefault = new PredictorAndArgs
        {
            Trainer = new SubComponent(MultiClassTrembleTrainer, "nl=20"),
            Tag = "MultiClassDT-def"
        };

        public static PredictorAndArgs MultiClassDecisionTreePruning = new PredictorAndArgs
        {
            Trainer = new SubComponent(MultiClassTrembleTrainer, "nl=20 prune=+"),
            Tag = "MultiClassDT-prune"
        };

        public static PredictorAndArgs MultiClassDecisionTreeGini = new PredictorAndArgs
        {
            Trainer = new SubComponent(MultiClassTrembleTrainer, "nl=20 imp=Gini"),
            Tag = "MultiClassDT-gini"
        };

        public static PredictorAndArgs MultiClassDecisionTreeModified = new PredictorAndArgs
        {
            Trainer = new SubComponent(MultiClassTrembleTrainer, "mil=2 inp=SingleFeaturePredictor{nb=FIFTEEN ff=0.7} nl=40"),
            Tag = "MultiClassDT-mod"
        };

        public const string BinaryBPMTrainer = "BinaryBPM";

        public static PredictorAndArgs BinaryBayesPointMachine = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryBPMTrainer),
            MamlArgs = new[] { "norm=no" }
        };

        public static PredictorAndArgs BinaryBayesPointMachineNoBias = new PredictorAndArgs
        {
            Trainer = new SubComponent(BinaryBPMTrainer, "bias=-"),
            MamlArgs = new[] { "norm=no" },
            Tag = "BinaryBPM-NoBias"
        };

        public static PredictorAndArgs MulticlassNaiveBayesClassifier = new PredictorAndArgs
        {
            Trainer = new SubComponent("MultiClassNaiveBayes", ""),
            BaselineProgress = true,
        };

        public static PredictorAndArgs FieldAwareFactorizationMachine = new PredictorAndArgs
        {
            Trainer = new SubComponent("FieldAwareFactorizationMachine", "d=5 shuf- norm-"),
            MamlArgs = new[] { "xf=Copy{col=DupFeatures:Features} xf=MinMax{col=Features col=DupFeatures} norm=No", "col[Feature]=DupFeatures" },
            BaselineProgress = true
        };
    }
}