using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.BinaryClassification.Calibrators
{
    public static class Platt2
    {

        class ModelInput
        {
            public bool Label { get; set; }
            public float Score { get; set; }
        }

        class ModelInput2
        {
            public float Score { get; set; }
            public bool Label { get; set; }
        }

        class ModelInput3
        {
            public float ScoreX { get; set; }
            public bool Label { get; set; }
        }

        public static void Example()
        {
            var mlContext = new MLContext(seed: 0);

            IDataView data = mlContext.Data.LoadFromEnumerable<ModelInput>(
                new ModelInput[]
                {
                                new ModelInput { Score = 10, Label = true },
                                new ModelInput { Score = 15, Label = false },
                }
            );

            var calibratorEstimator = mlContext.BinaryClassification.Calibrators
                .Platt();

            var calibratorTransformer = calibratorEstimator.Fit(data);
            var finalData = calibratorTransformer.Transform(data);
            var prev = finalData.Preview();


            // EXAMPLE 2
            //IDataView data2 = mlContext.Data.LoadFromEnumerable<ModelInput2>(
            //    new ModelInput2[]
            //    {
            //                    new ModelInput2 { Score = 10, Label = true },
            //                    new ModelInput2 { Score = 15, Label = false },
            //    }
            //);

            //calibratorEstimator = mlContext.BinaryClassification.Calibrators
            //    .Platt();

            //calibratorTransformer = calibratorEstimator.Fit(data2);
            //finalData = calibratorTransformer.Transform(data2);
            //prev = finalData.Preview();

            // EXAMPLE 3
            IDataView data3 = mlContext.Data.LoadFromEnumerable<ModelInput3>(
                new ModelInput3[]
                {
                                new ModelInput3 { ScoreX = 10, Label = true },
                                new ModelInput3 { ScoreX = 15, Label = false },
                }
            );

            calibratorEstimator = mlContext.BinaryClassification.Calibrators
                .Platt(scoreColumnName: "ScoreX");

            calibratorTransformer = calibratorEstimator.Fit(data3);
            finalData = calibratorTransformer.Transform(data3);
            prev = finalData.Preview();
        }
    }
}
