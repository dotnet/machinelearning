using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Samples.Dynamic;
using System.Collections.Generic;
using System;
using System.IO;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Linq;

namespace Microsoft.ML.Samples
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            //MakeBinarySearchTree(10);
            TestGam();
        }


        private static int[] MakeBinarySearchTree(int numInternalNodes)
        {
            var binIndices = Enumerable.Range(0, numInternalNodes - 1).ToArray();
            var bstIndices = new List<int>();

            MakeBinarySearchTreeRecursive(binIndices, 0, binIndices.Length - 1, bstIndices);
            var ret = bstIndices.ToArray();
            return ret;
        }

        private static void MakeBinarySearchTreeRecursive(
            int[] array, int lower, int upper, List<int> bstIndices)
        {
            if (lower > upper)
            {
                var mid = (lower + upper) / 2;
                bstIndices.Add(array[mid]);
                MakeBinarySearchTreeRecursive(array, lower, mid - 1, bstIndices);
                MakeBinarySearchTreeRecursive(array, mid + 1, upper, bstIndices);
            }
        }

        private static void TestGam()
        {
            var mlContext = new MLContext(seed: 0, conc: 0);

            var idv = mlContext.Data.CreateTextReader(
                    new TextLoader.Arguments()
                    {
                        HasHeader = false,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("Features", DataKind.R4, 1, 9),
                        }
                    }).Read(@"F:\temp\ini\data\breast-cancer-noNan.txt");

            //var pipeline = mlContext.Transforms.Normalize("Features")
            //    .Append(mlContext.Regression.Trainers.GeneralizedAdditiveModels());

            var pipeline = mlContext.Regression.Trainers.GeneralizedAdditiveModels();

            var model = pipeline.Fit(idv);
            var data = model.Transform(idv);

            var roleMappedSchema = new RoleMappedSchema(data.Schema, false,
                new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, "Features"),
                new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, "Label"));

            //using (var fs = File.Create(@"F:\temp\ini\model3.zip"))
            //    mlContext.Model.Save(model, fs);

            using (StreamWriter writer = new StreamWriter(@"F:\temp\ini\model5.ini"))
                model.Model.SaveAsIni(writer, roleMappedSchema);

            var results = mlContext.Regression.Evaluate(data);

            //using (var fs = File.OpenRead(@"F:\temp\ini\model3.zip"))
            //{
            //    var loadedModel = mlContext.Model.Load(fs);
            //    var testOut = loadedModel.Transform(idv);
            //}

            //var modelPath = @"F:\temp\model.zip";
            //using (var fs = File.Create(modelPath))
            //    mlContext.Model.Save(model, fs);

            //var savePredCommand = new SavePredictorCommand(
            //    mlContext,
            //    new SavePredictorCommand.Arguments
            //    {
            //        InputModelFile = modelPath,
            //    });

            //savePredCommand.Run();
        }
    }
}
