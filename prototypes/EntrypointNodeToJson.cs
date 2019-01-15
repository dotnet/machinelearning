using System;
using System.Collections.Generic;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Samples.Dynamic;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            //FeatureContributionCalculationTransform_RegressionExample.FeatureContributionCalculationTransform_Regression();

            var mlContext = new MLContext();
            var iHostEnv = mlContext as IHostEnvironment;

            iHostEnv.ComponentCatalog.RegisterAssembly(typeof(SdcaBinaryTrainer).Assembly);
            iHostEnv.ComponentCatalog.RegisterAssembly(typeof(LogLossFactory).Assembly);

            var arg = new SdcaBinaryTrainer.Arguments();
            arg.L2Const = 0.02f;

            var entrypointNode = EntryPointNode.Create(mlContext, "Trainers.StochasticDualCoordinateAscentBinaryClassifier",
                arg,
                iHostEnv.ComponentCatalog,
                new RunContext(new ExceptionContext()),
                new Dictionary<string, string>() { { "TrainingData", "TrainData" } },
                new Dictionary<string, string>());

            var json = entrypointNode.ToJson();
            Console.WriteLine(json);

            Console.ReadLine();
        }
    }

    public class ExceptionContext : IExceptionContext
    {
        public string ContextDescription => throw new NotImplementedException();

        public TException Process<TException>(TException ex) where TException : Exception
        {
            throw new NotImplementedException();
        }
    }
}
