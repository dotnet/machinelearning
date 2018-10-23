// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Calibration;

[assembly: LoadableClass(typeof(TreeEnsembleCombiner), null, typeof(SignatureModelCombiner), "Fast Tree Model Combiner", "FastTreeCombiner")]

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public sealed class TreeEnsembleCombiner : IModelCombiner
    {
        private readonly IHost _host;
        private readonly PredictionKind _kind;

        public TreeEnsembleCombiner(IHostEnvironment env, PredictionKind kind)
        {
            _host = env.Register("TreeEnsembleCombiner");
            switch (kind)
            {
                case PredictionKind.BinaryClassification:
                case PredictionKind.Regression:
                case PredictionKind.Ranking:
                    _kind = kind;
                    break;
                default:
                    throw _host.ExceptUserArg(nameof(kind), $"Tree ensembles can be either of type {nameof(PredictionKind.BinaryClassification)}, " +
                        $"{nameof(PredictionKind.Regression)} or {nameof(PredictionKind.Ranking)}");
            }
        }

        public IPredictor CombineModels(IEnumerable<IPredictor> models)
        {
            _host.CheckValue(models, nameof(models));

            var ensemble = new Ensemble();
            int modelCount = 0;
            int featureCount = -1;
            bool binaryClassifier = false;
            foreach (var model in models)
            {
                modelCount++;

                var predictor = model;
                _host.CheckValue(predictor, nameof(models), "One of the models is null");

                var calibrated = predictor as CalibratedPredictorBase;
                double paramA = 1;
                if (calibrated != null)
                {
                    _host.Check(calibrated.Calibrator is PlattCalibrator,
                        "Combining FastTree models can only be done when the models are calibrated with Platt calibrator");
                    predictor = calibrated.SubPredictor;
                    paramA = -(calibrated.Calibrator as PlattCalibrator).ParamA;
                }
                var tree = predictor as FastTreePredictionWrapper;
                if (tree == null)
                    throw _host.Except("Model is not a tree ensemble");
                foreach (var t in tree.TrainedEnsemble.Trees)
                {
                    var bytes = new byte[t.SizeInBytes()];
                    int position = -1;
                    t.ToByteArray(bytes, ref position);
                    position = -1;
                    var tNew = new RegressionTree(bytes, ref position);
                    if (paramA != 1)
                    {
                        for (int i = 0; i < tNew.NumLeaves; i++)
                            tNew.SetOutput(i, tNew.LeafValues[i] * paramA);
                    }
                    ensemble.AddTree(tNew);
                }

                if (modelCount == 1)
                {
                    binaryClassifier = calibrated != null;
                    featureCount = tree.InputType.ValueCount;
                }
                else
                {
                    _host.Check((calibrated != null) == binaryClassifier, "Ensemble contains both calibrated and uncalibrated models");
                    _host.Check(featureCount == tree.InputType.ValueCount, "Found models with different number of features");
                }
            }

            var scale = 1 / (double)modelCount;

            foreach (var t in ensemble.Trees)
            {
                for (int i = 0; i < t.NumLeaves; i++)
                    t.SetOutput(i, t.LeafValues[i] * scale);
            }

            switch (_kind)
            {
                case PredictionKind.BinaryClassification:
                    if (!binaryClassifier)
                        return new FastTreeBinaryPredictor(_host, ensemble, featureCount, null);

                    var cali = new PlattCalibrator(_host, -1, 0);
                    return new FeatureWeightsCalibratedPredictor(_host, new FastTreeBinaryPredictor(_host, ensemble, featureCount, null), cali);
                case PredictionKind.Regression:
                    return new FastTreeRegressionPredictor(_host, ensemble, featureCount, null);
                case PredictionKind.Ranking:
                    return new FastTreeRankingPredictor(_host, ensemble, featureCount, null);
                default:
                    _host.Assert(false);
                    throw _host.ExceptNotSupp();
            }
        }
    }
}
