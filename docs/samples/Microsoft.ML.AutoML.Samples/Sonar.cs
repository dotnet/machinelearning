using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.AutoML.Samples
{
    /// <summary>
    /// Time series forecasting using Sonar dataset.
    /// </summary>
    public static class Sonar
    {
        public static void Run()
        {
            // Load file
            var trainDataPath = @"C:\Users\xiaoyuz\Desktop\forecasting\-0401_load.csv";
            var evaluatePath = @"C:\Users\xiaoyuz\Desktop\forecasting\0401_0415_load.csv";
            var predictedPath = @"C:\Users\xiaoyuz\Desktop\forecasting\predicted.csv";
            var trainDf = DataFrame.LoadCsv(trainDataPath);
            var evaluateDf = DataFrame.LoadCsv(evaluatePath);

            var mlContext = new MLContext();
            var searchSpace = new SearchSpace<ForecastBySsaSearchSpace>();
            var runner = new CustomRunner(mlContext);

            Console.WriteLine($"train data Length: {trainDf.Rows.Count}");
            var pipeline = mlContext.Transforms.CopyColumns("newLoad", "load")
                .Append(mlContext.Auto().CreateSweepableEstimator((context, ss) =>
                {
                    return mlContext.Forecasting.ForecastBySsa("predict", "load", ss.WindowSize, ss.SeriesLength, Convert.ToInt32(trainDf.Rows.Count), ss.Horizon, rank: ss.Rank, variableHorizon: true);
                }, searchSpace));

            var autoMLExperiment = mlContext.Auto().CreateExperiment();

            autoMLExperiment.SetPipeline(pipeline)
                            .SetTrialRunner(runner)
                            .SetTrainingTimeInSeconds(600)
                            .SetEvaluateMetric(RegressionMetric.RootMeanSquaredError)
                            .SetDataset(trainDf, evaluateDf);

            mlContext.Log += (e, o) =>
            {
                if (o.Source.StartsWith("AutoMLExperiment"))
                {
                    Console.WriteLine(o.RawMessage);
                }
            };

            var res = autoMLExperiment.Run().Result;
            var bestModel = res.Model;
            Console.WriteLine($"best model rmse: {res.Metric}");

            // evaluate
            var predictEngine = bestModel.CreateTimeSeriesEngine<ForecastInput, ForecastOutnput>(mlContext);

            var predictLoads1H = new List<float>();
            var predictLoads2H = new List<float>();
            predictLoads2H.Add(0);
            foreach (var load in evaluateDf.GetColumn<Single>("load"))
            {
                // firstly, get next n predict where n is horizon
                var predict = predictEngine.Predict();

                predictLoads1H.Add(predict.Predict[0]);
                predictLoads2H.Add(predict.Predict[1]);

                // update model with truth value
                predictEngine.Predict(new ForecastInput()
                {
                    Load = load,
                });
            }

            evaluateDf["predict_load_1h"] = DataFrameColumn.Create("predict_load_1h", predictLoads1H);
            evaluateDf["predict_load_2h"] = DataFrameColumn.Create("predict_load_2h", predictLoads2H.SkipLast(1));
            DataFrame.WriteCsv(evaluateDf, predictedPath);
        }
    }

    public class ForecastInput
    {
        [ColumnName("load")]
        public float Load { get; set; }
    }

    public class ForecastOutnput
    {
        [ColumnName("predict")]
        public float[] Predict { get; set; }
    }

    public class CustomRunner : ITrialRunner
    {
        private MLContext _context;

        public CustomRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            var datasetManager = provider.GetService<IDatasetManager>();
            if (datasetManager is TrainTestDatasetManager trainTestDatasetManager)
            {
                try
                {
                    var trainDataset = trainTestDatasetManager.TrainDataset;
                    var testDataset = trainTestDatasetManager.TestDataset;

                    var stopWatch = new Stopwatch();
                    stopWatch.Start();
                    var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                    var model = pipeline.Fit(trainDataset);

                    var predictEngine = model.CreateTimeSeriesEngine<ForecastInput, ForecastOutnput>(this._context);

                    // check point
                    predictEngine.CheckPoint(this._context, "origin");

                    var predictedLoad1H = new List<float>();
                    var predictedLoad2H = new List<float>();
                    var N = testDataset.GetRowCount();

                    // evaluate
                    foreach (var load in testDataset.GetColumn<Single>("load"))
                    {
                        // firstly, get next n predict where n is horizon
                        var predict = predictEngine.Predict();

                        predictedLoad1H.Add(predict.Predict[0]);
                        predictedLoad2H.Add(predict.Predict[1]);

                        // update model with truth value
                        predictEngine.Predict(new ForecastInput()
                        {
                            Load = load,
                        });
                    }

                    var rmse1H = Enumerable.Zip(testDataset.GetColumn<float>("load"), predictedLoad1H)
                                           .Select(x => Math.Pow(x.First - x.Second, 2))
                                           .Average();
                    rmse1H = Math.Sqrt(rmse1H);

                    var rmse2H = Enumerable.Zip(testDataset.GetColumn<float>("load").Skip(1), predictedLoad2H)
                                           .Select(x => Math.Pow(x.First - x.Second, 2))
                                           .Average();
                    rmse2H = Math.Sqrt(rmse2H);

                    stopWatch.Stop();
                    var rmse = (rmse1H + rmse2H) / 2;

                    return new TrialResult()
                    {
                        Metric = rmse,
                        Model = model,
                        TrialSettings = settings,
                        DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    };

                }
                catch (Exception)
                {
                    return new TrialResult()
                    {
                        Metric = double.MaxValue,
                        Model = null,
                        TrialSettings = settings,
                        DurationInMilliseconds = 0,
                    };
                }
            }

            throw new ArgumentException();
        }
    }
    public class ForecastBySsaSearchSpace
    {
        [Range(2, 24 * 7 * 30)]
        public int WindowSize { get; set; } = 2;

        [Range(2, 24 * 7 * 30)]
        public int SeriesLength { get; set; } = 2;

        [Range(1, 24 * 7 * 30)]
        public int Rank { get; set; } = 1;

        [Range(2, 50)]
        public int Horizon { get; set; } = 2;
    }
}
