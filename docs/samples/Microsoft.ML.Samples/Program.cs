using System;
using System.Reflection;
using System.Text.RegularExpressions;
using Samples.Dynamic;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll("LocalizeRootCauseMultipleDimensions");

        internal static void RunAll(string pattern = null)
        {
            int samples = 0;
            foreach (var type in Assembly.GetExecutingAssembly().GetTypes())
            {
                if (pattern?.Equals(type.Name) ?? false)
                {
                    var sample = type.GetMethod("Example", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

                    if (sample != null)
                    {
                        Console.WriteLine(type.Name);
                        sample.Invoke(null, null);
                        samples++;
                    }
                }
            }

            Console.WriteLine("Number of samples that ran without any exception: " + samples);
        }
    }
}
