using System;

namespace Microsoft.ML.AutoML.Samples
{
    public class Program
    {
        public static void Main(string[] args)
        {
            try
            {
                /*RecommendationExperiment.Run();
                Console.Clear();

                RegressionExperiment.Run();
                Console.Clear();

                BinaryClassificationExperiment.Run();
                Console.Clear();*/

                MulticlassClassificationExperiment.Run();
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
