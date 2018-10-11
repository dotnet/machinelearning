// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Samples
{
    public static class DatasetCreator
    {
        public static (string trainPath, string testPath) CreateRegressionDataset()
        {
            // creating a small sample dataset, and writting it to file
            string trainDataPath = @"RegressionTrainDataset.txt";
            string testDataPath = @"RegressionTestDataset.txt";

            string header = "feature_a, feature_b, target";

            int a = 0;
            int b = 0;
            int target = 0;

            var csvTrain = new StringBuilder().AppendLine(header);
            var csvTest = new StringBuilder().AppendLine(header);

            Random rnd = new Random();
            for (int i = 0; i < 1000; i++)
            {
                a = rnd.Next(i - 5, i + 5);
                b = rnd.Next(0, 10);

                target = 2*a + b;

                if (i % 15 == 0)
                    csvTest.AppendLine($"{a}, {b}, {target}");
                else
                    csvTrain.AppendLine($"{a}, {b} , {target}");
            }


            if (!File.Exists(trainDataPath))
                File.WriteAllText(trainDataPath, csvTrain.ToString());
            else
            {
                new Exception("Train dataset file already exists");
            }

            if (!File.Exists(testDataPath))
                File.WriteAllText(testDataPath, csvTest.ToString());
            else
            {
                new Exception("Test dataset file already exists");
            }

            return (trainDataPath, testDataPath);
        }
    }
}
