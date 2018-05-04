// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: EntryPointModule(typeof(SummarizePredictor))]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class SummarizePredictor
    {
        public abstract class InputBase
        {
            [Argument(ArgumentType.Required, ShortName = "predictorModel", HelpText = "The predictor to summarize")]
            public IPredictorModel PredictorModel;
        }

        public sealed class Input : InputBase
        {
        }

        [TlcModule.EntryPoint(Name = "Models.Summarizer", Desc = "Summarize a linear regression predictor.")]
        public static CommonOutputs.SummaryOutput Summarize(IHostEnvironment env, SummarizePredictor.Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("LinearRegressionPredictor");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            RoleMappedData rmd;
            IPredictor predictor;
            input.PredictorModel.PrepareData(host, new EmptyDataView(host, input.PredictorModel.TransformModel.InputSchema), out rmd, out predictor);

            var output = new CommonOutputs.SummaryOutput();
            output.Summary = GetSummaryAndStats(host, predictor, rmd.Schema, out output.Stats);
            return output;
        }

        public static IDataView GetSummaryAndStats(IHostEnvironment env, IPredictor predictor, RoleMappedSchema schema, out IDataView stats)
        {
            var calibrated = predictor as CalibratedPredictorBase;
            while (calibrated != null)
            {
                predictor = calibrated.SubPredictor;
                calibrated = predictor as CalibratedPredictorBase;
            }

            IDataView summary = null;
            stats = null;
            var dvGetter = predictor as ICanGetSummaryAsIDataView;
            var rowGetter = predictor as ICanGetSummaryAsIRow;
            if (dvGetter != null)
                summary = dvGetter.GetSummaryDataView(schema);
            if (rowGetter != null)
            {
                var row = rowGetter.GetSummaryIRowOrNull(schema);
                env.Check(dvGetter == null || row == null,
                    "Predictor outputs two summary data views, don't know which one to choose");
                if (row != null)
                    summary = RowCursorUtils.RowAsDataView(env, row);
                var statsRow = rowGetter.GetStatsIRowOrNull(schema);
                if (statsRow != null)
                    stats = RowCursorUtils.RowAsDataView(env, statsRow);
            }
            if (dvGetter == null && rowGetter == null)
            {
                var bldr = new ArrayDataViewBuilder(env);
                var summaryModel = predictor as ICanSaveSummary;

                // Save a data view containing one row and one column with the model summary.
                if (summaryModel != null)
                {
                    var sb = new StringBuilder();
                    using (StringWriter sw = new StringWriter(sb))
                        summaryModel.SaveSummary(sw, schema);
                    bldr.AddColumn("Summary", sb.ToString());
                }
                else
                    bldr.AddColumn("PredictorName", predictor.GetType().ToString());
                summary = bldr.GetDataView();
            }
            env.AssertValue(summary);
            return summary;
        }
    }
}
