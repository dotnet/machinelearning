using System;
using System.Reflection;
using Samples.Dynamic;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            CalculateFeatureContribution.Example();

            if (args.Length > 0 && args[0] == "-runall")
                RunAll();
        }

        internal static void RunAll()
        {
            foreach (var type in Assembly.GetExecutingAssembly().GetTypes())
            {
                var method = type.GetMethod("Example", BindingFlags.NonPublic | BindingFlags.Public
                    | BindingFlags.Static | BindingFlags.FlattenHierarchy);

                if (method != null)
                {
                    try
                    {
                        method.Invoke(null, null);
                    }
                    catch (Exception ex)
                    {
                        // Print the exception for debugging.
                        Console.Write(type);
                        Console.Write(ex);

                        // Throw to fail.
                        throw ex;
                    }
                }
            }
        }
    }
}
