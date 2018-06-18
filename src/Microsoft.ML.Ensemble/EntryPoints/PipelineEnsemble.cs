// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;

[assembly: EntryPointModule(typeof(PipelineEnsemble))]

namespace Microsoft.ML.Runtime.Ensemble.EntryPoints
{
    public static class PipelineEnsemble
    {
        public sealed class SummaryOutput
        {
            [TlcModule.Output(Desc = "The summaries of the individual predictors")]
            public IDataView[] Summaries;

            [TlcModule.Output(Desc = "The model statistics of the individual predictors")]
            public IDataView[] Stats;
        }

        [TlcModule.EntryPoint(Name = "Models.EnsembleSummary", Desc = "Summarize a pipeline ensemble predictor.")]
        public static SummaryOutput Summarize(IHostEnvironment env, SummarizePredictor.Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("PipelineEnsemblePredictor");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            input.PredictorModel.PrepareData(host, new EmptyDataView(host, input.PredictorModel.TransformModel.InputSchema), out RoleMappedData rmd, out IPredictor predictor);

            var calibrated = predictor as CalibratedPredictorBase;
            while (calibrated != null)
            {
                predictor = calibrated.SubPredictor;
                calibrated = predictor as CalibratedPredictorBase;
            }
            var ensemble = predictor as SchemaBindablePipelineEnsembleBase;
            host.CheckUserArg(ensemble != null, nameof(input.PredictorModel.Predictor), "Predictor is not a pipeline ensemble predictor");

            var summaries = new IDataView[ensemble.PredictorModels.Length];
            var stats = new IDataView[ensemble.PredictorModels.Length];
            for (int i = 0; i < ensemble.PredictorModels.Length; i++)
            {
                var pm = ensemble.PredictorModels[i];

                pm.PrepareData(host, new EmptyDataView(host, pm.TransformModel.InputSchema), out rmd, out IPredictor pred);
                summaries[i] = SummarizePredictor.GetSummaryAndStats(host, pred, rmd.Schema, out stats[i]);
            }
            return new SummaryOutput() { Summaries = summaries, Stats = stats };
        }
    }
}
