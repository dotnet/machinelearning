using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto.APINew
{
    public static class MLContextExtension
    {
        public static AutoInfereceCataglog AutoInference(this MLContext mlContext)
        {
            return new AutoInfereceCataglog();
        }
    }

    public class ExperimentSettings
    {
        public uint MaxInferenceTimeInSeconds;
        public bool EnableCaching;
        public CancellationToken CancellationToken;
    }

    public class RegressionExperimentSettings : ExperimentSettings
    {
        public IProgress<Data.RegressionMetrics> ProgressCallback;
        public Data.RegressionMetrics OptimizingMetrics;
        public RegressionTrainer[] WhitelistedTrainers;
    }

    public enum RegressionMetric
    {
        RSquared
    }

    public enum RegressionTrainer
    {
        LightGbm
    }

    public class ColumnInfereceResults
    {
        public TextLoader.Arguments TextLoaderArgs;
        public ColumnInformation ColumnInformation;
    }

    public class ColumnInformation
    {
        public string LableColumn;
        public string WeightColumn;
        public IEnumerable<string> CategoricalColumns;
    }

    public class RegressionExperiment
    {
        public RunResult<RegressionMetric> Execute(IDataView testData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        public RunResult<RegressionMetric> Execute(IDataView testData, IDataView validationData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        public RunResult<RegressionMetric> Execute(IDataView testData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }
    }

    public class AutoInfereceCataglog
    {
        RegressionExperiment CreateRegressionExperiment(uint maxInferenceTimeInSeconds)
        {
            return new RegressionExperiment();
        }

        RegressionExperiment CreateRegressionExperiment(RegressionExperimentSettings experimentSettings)
        {
            return new RegressionExperiment();
        }

        public ColumnInfereceResults InferColumns()
        {
            throw new NotImplementedException();
        }
    }

    public class RunResult<T>
    {
        public readonly T Metrics;
        public readonly ITransformer Model;
        public readonly Exception Exception;
        public readonly string TrainerName;
        public readonly int RuntimeInSeconds;

        internal readonly Pipeline Pipeline;
        internal readonly int PipelineInferenceTimeInSeconds;

        internal RunResult(
            ITransformer model,
            T metrics,
            Pipeline pipeline,
            Exception exception,
            int runtimeInSeconds,
            int pipelineInferenceTimeInSeconds)
        {
            Model = model;
            Metrics = metrics;
            Pipeline = pipeline;
            Exception = exception;
            RuntimeInSeconds = runtimeInSeconds;
            PipelineInferenceTimeInSeconds = pipelineInferenceTimeInSeconds;

            TrainerName = pipeline?.Nodes.Where(n => n.NodeType == PipelineNodeType.Trainer).Last().Name;
        }
    }
}
