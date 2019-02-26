// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Samples
{
    public class Program
    {
        public static void Main(string[] args)
        {
            try
            {
                AutoTrainRegression.Run();
                Console.Clear();

                AutoTrainBinaryClassification.Run();
                Console.Clear();

                AutoTrainMulticlassClassification.Run();
                Console.Clear();

                CustomizeTraining.Run();
                Console.Clear();

                ObserveProgress.Run();
                Console.Clear();

                Cancellation.Run();
                Console.Clear();

                Console.WriteLine("Done");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            Console.ReadLine();
        }
    }
}
