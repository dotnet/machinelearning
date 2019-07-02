using System;
using System.Reflection;
using Samples.Dynamic;
using Samples.Dynamic.Trainers.BinaryClassification;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll();

        internal static void RunAll()
        {
            int samples = 0;
            foreach (var type in Assembly.GetExecutingAssembly().GetTypes())
            {
                var sample = type.GetMethod("Example", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);
                String[] test = {"PairwiseCoupling", "PermutationFeatureImportance", "SdcaMaximumEntropy",
                "SdcaMaximumEntropyWithOptions", "SdcaNonCalibrated", "SdcaNonCalibratedWithOptions"};
                if (sample != null && Array.IndexOf(test, type.Name) > -1)//type.Name.Equals("ConvertType"))

                {
                    Console.WriteLine(type.Name);
                    sample.Invoke(null, null);
                    samples++;
                }
            }

            Console.WriteLine("Number of samples that ran without any exception: " + samples);
        }
    }
}
