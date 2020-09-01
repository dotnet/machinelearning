using System;
using System.Reflection;
using Microsoft.ML.Data;
using Samples.Dynamic;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll(args == null || args.Length == 0 ? null : args[0]);

        internal static void RunAll(string name = null)
        {
            var mlContext = new MLContext();

            var loader = mlContext.Data.CreateTextLoader<ModelInput>();

            //THIS WORK
            var d = loader.Load("./xyz/*");

            //THIS DOESN'T
            var data = mlContext.Data.LoadFromTextFile<ModelInput>("./xyz/*");
        }

        public class ModelInput
        {
            [LoadColumn(0)]
            public bool Label { get; set; }
            [LoadColumn(1)]
            public string Workclass { get; set; }
        }
    }
}
