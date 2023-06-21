// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.Fairlearn
{
    /// <summary>
    /// 
    /// 1, generate cost column from lamda parameter
    /// 2. insert cost column into dataset
    /// 3. restore trainable pipeline
    /// 4. train
    /// 5. calculate metric = observe loss + fairness loss
    /// </summary>
    public class GridSearchTrailRunner : ITrialRunner
    {
        private readonly MLContext _context;
        private readonly string _labelColumn;
        private readonly string _sensitiveColumn;
        private readonly SweepablePipeline _pipeline;
        private readonly ClassificationMoment _moment;
        private readonly ITrainValidateDatasetManager _datasetManager;

        public GridSearchTrailRunner(MLContext context, ITrainValidateDatasetManager datasetManager, string labelColumn, string sensitiveColumn, SweepablePipeline pipeline, ClassificationMoment moment)
        {
            _context = context;
            this._datasetManager = datasetManager;
            this._labelColumn = labelColumn;
            this._sensitiveColumn = sensitiveColumn;
            _pipeline = pipeline;
            _moment = moment;
        }

        public void Dispose()
        {
        }

        public Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            //DataFrameColumn signedWeights = null;
            var pipeline = _pipeline.BuildFromOption(_context, settings.Parameter["_pipeline_"]);
            // get lambda 
            var lambdas = settings.Parameter["_lambda_search_space"];
            var key = lambdas.Keys;
            // (sign, group, value)
            var lambdasValue = key.Select(x =>
            {
                var sign = x.Split('_')[1] == "pos" ? "+" : "-";
                var e = x.Split('_')[0];
                var value = lambdas[x].AsType<float>();

                return (sign, e, value);
            });

            var trainDataset = _datasetManager.LoadTrainDataset(_context, settings);
            var validateDataset = _datasetManager.LoadValidateDataset(_context, settings);

            var df = new DataFrame();
            df["sign"] = DataFrameColumn.Create("sign", lambdasValue.Select(x => x.sign));
            df["group_id"] = DataFrameColumn.Create("group_id", lambdasValue.Select(x => x.e));
            df["value"] = DataFrameColumn.Create("value", lambdasValue.Select(x => x.value));
            _moment.LoadData(trainDataset, DataFrameColumn.Create("y", trainDataset.GetColumn<bool>(this._labelColumn)), DataFrameColumn.Create("group_id", trainDataset.GetColumn<string>(this._sensitiveColumn)));
            var signWeightColumn = _moment.SignedWeights(df);
            trainDataset = ZipDataView.Create(_context, new IDataView[] { trainDataset, new DataFrame(signWeightColumn) });
            var model = pipeline.Fit(trainDataset);
            // returns an IDataview object that contains the predictions
            var eval = model.Transform(validateDataset);
            // extract the predicted label and convert it to 1.0f and 0.0 so that we can feed that into the gamma function
            var predictedLabel = eval.GetColumn<bool>("PredictedLabel").Select(b => b ? 1f : 0f).ToArray();
            var column = DataFrameColumn.Create<float>("pred", predictedLabel);
            //Get the gamma based on the predicted label of the testDataset
            _moment.LoadData(validateDataset, DataFrameColumn.Create("y", eval.GetColumn<bool>(this._labelColumn)), DataFrameColumn.Create("group_id", validateDataset.GetColumn<string>(this._sensitiveColumn)));
            var gamma = _moment.Gamma(column);
            double fairnessLost = Convert.ToSingle(gamma["value"].Max());
            var metrics = _context.BinaryClassification.EvaluateNonCalibrated(eval, this._labelColumn);
            // the metric should be the combination of the observed loss from the model and the fairness loss
            double metric = 0.0f;
            metric = metrics.Accuracy - fairnessLost;

            stopWatch.Stop();

            return Task.FromResult<TrialResult>(new FairnessTrialResult()
            {
                FairnessMetric = fairnessLost,
                Metric = metric,
                Model = model,
                Loss = -metric,
                TrialSettings = settings,
                DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
            });
        }
    }
}
