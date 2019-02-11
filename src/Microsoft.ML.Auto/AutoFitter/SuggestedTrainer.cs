// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Training;

namespace Microsoft.ML.Auto
{
    internal class SuggestedTrainer
    {
        public IEnumerable<SweepableParam> SweepParams { get; }
        public TrainerName TrainerName { get; }
        public ParameterSet HyperParamSet { get; set; }

        private readonly MLContext _mlContext;
        private readonly ITrainerExtension _trainerExtension;

        internal SuggestedTrainer(MLContext mlContext, ITrainerExtension trainerExtension,
            ParameterSet hyperParamSet = null)
        {
            _mlContext = mlContext;
            _trainerExtension = trainerExtension;
            SweepParams = _trainerExtension.GetHyperparamSweepRanges();
            TrainerName = TrainerExtensionCatalog.GetTrainerName(_trainerExtension);
            SetHyperparamValues(hyperParamSet);
        }

        public void SetHyperparamValues(ParameterSet hyperParamSet)
        {
            HyperParamSet = hyperParamSet;
            PropagateParamSetValues();
        }

        public SuggestedTrainer Clone()
        {
            return new SuggestedTrainer(_mlContext, _trainerExtension, HyperParamSet?.Clone());
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> BuildTrainer(MLContext env)
        {
            IEnumerable<SweepableParam> sweepParams = null;
            if (HyperParamSet != null)
            {
                sweepParams = SweepParams;
            }
            return _trainerExtension.CreateInstance(_mlContext, sweepParams);
        }

        public override string ToString()
        {
            var paramsStr = string.Empty;
            if (SweepParams != null)
            {
                paramsStr = string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"));
            }
            return $"{TrainerName}{{{paramsStr}}}";
        }

        public PipelineNode ToPipelineNode()
        {
            var hyperParams = SweepParams.Where(p => p != null && p.RawValue != null);
            var elementProperties = TrainerExtensionUtil.BuildPipelineNodeProps(TrainerName, hyperParams);
            return new PipelineNode(TrainerName.ToString(), PipelineNodeType.Trainer, 
                new[] { "Features" }, new[] { "Score" }, elementProperties);
        }

        /// <summary>
        /// make sure sweep params and param set are consistent
        /// </summary>
        private void PropagateParamSetValues()
        {
            if (HyperParamSet == null)
            {
                return;
            }

            var spMap = SweepParams.ToDictionary(sp => sp.Name);

            foreach (var hp in HyperParamSet)
            {
                if (spMap.ContainsKey(hp.Name))
                {
                    var sp = spMap[hp.Name];
                    sp.SetUsingValueText(hp.ValueText);
                }
            }
        }
    }
}
