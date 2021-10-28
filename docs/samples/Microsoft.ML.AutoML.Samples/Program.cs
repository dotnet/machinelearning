// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML.Samples
{
    public class Program
    {
        public static void Main(string[] args)
        {
            try
            {
                RecommendationExperiment.Run();
                Console.Clear();

                RegressionExperiment.Run();
                Console.Clear();

                BinaryClassificationExperiment.Run();
                Console.Clear();

                MulticlassClassificationExperiment.Run();
                Console.Clear();

                RankingExperiment.Run();
                Console.Clear();

                Console.WriteLine("Done");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception {ex}");
            }

            Console.ReadLine();
        }
    }
}
