using System;
using System.Reflection;
using Samples.Dynamic.Trainers.MulticlassClassification;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine(nameof(LbfgsMaximumEntropy));
            LbfgsMaximumEntropy.Example();
            Console.WriteLine(nameof(LbfgsMaximumEntropyWithOptions));
            LbfgsMaximumEntropyWithOptions.Example();
            Console.WriteLine(nameof(LightGbm));
            LightGbm.Example();
            Console.WriteLine(nameof(LightGbmWithOptions));
            LightGbmWithOptions.Example();
            Console.WriteLine(nameof(NaiveBayes));
            NaiveBayes.Example();
            Console.WriteLine(nameof(OneVersusAll));
            OneVersusAll.Example();
            Console.WriteLine(nameof(PairwiseCoupling));
            PairwiseCoupling.Example();
            Console.WriteLine(nameof(SdcaMaximumEntropy));
            SdcaMaximumEntropy.Example();
            Console.WriteLine(nameof(SdcaMaximumEntropyWithOptions));
            SdcaMaximumEntropyWithOptions.Example();
            Console.WriteLine(nameof(SdcaNonCalibrated));
            SdcaNonCalibrated.Example();
            Console.WriteLine(nameof(SdcaNonCalibratedWithOptions));
            SdcaNonCalibratedWithOptions.Example();

        }

        internal static void RunAll()
        {
            int samples = 0;
            foreach (var type in Assembly.GetExecutingAssembly().GetTypes())
            {
                var sample = type.GetMethod("Example", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

                if (sample != null)
                {
                    sample.Invoke(null, null);
                    samples++;
                }
            }

            Console.WriteLine("Number of samples that ran without any exception: " + samples);
        }
    }
}
