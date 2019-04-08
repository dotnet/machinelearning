using System;
using System.Collections.Generic;
using System.Reflection;
using Samples.Dynamic;

namespace Microsoft.ML.Samples
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            //DataViewEnumerable.Example();

            //if (args[1] == "-runall")
                RunAll();
        }

        internal static void RunAll()
        {
            List<Type> failures = new List<Type>();
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
                        Console.Write(ex);
                        failures.Add(type);
                    }
                }
            }
        }
    }
}
