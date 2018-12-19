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
        static IHostEnvironment Host = new MLContext();

        static void Main(string[] args)
        {
            //MakeBinarySearchTree(10);
            TestGam();
        }


        private static (int[], int[], int[]) MakeBinarySearchTree(int numInternalNodes)
        {
            var binIndices = Enumerable.Range(0, numInternalNodes).ToArray();
            var bstIndices = new List<int>();
            var lteChild = new List<int>();
            var gtChild = new List<int>();
            var internalNodeId = numInternalNodes;

            MakeBinarySearchTreeRecursive(binIndices, 0, binIndices.Length - 1, bstIndices, lteChild, gtChild, ref internalNodeId);
            var ret = (bstIndices.ToArray(), lteChild.ToArray(), gtChild.ToArray());
            return ret;
        }

        private static int MakeBinarySearchTreeRecursive(
            int[] array, int lower, int upper,
            List<int> bstIndices, List<int> lteChild, List<int> gtChild, ref int internalNodeId)
        {
            if (lower > upper)
            {
                // Base case: we've reached a leaf node
                Assert(lower == upper + 1);
                return lower + 100;
            }
            else
            {
                var mid = (lower + upper) / 2;
                var left = MakeBinarySearchTreeRecursive(
                    array, lower, mid - 1, bstIndices, lteChild, gtChild, ref internalNodeId);
                var right = MakeBinarySearchTreeRecursive(
                    array, mid + 1, upper, bstIndices, lteChild, gtChild, ref internalNodeId);
                bstIndices.Insert(0, array[mid]);
                lteChild.Insert(0, left);
                gtChild.Insert(0, right);
                return --internalNodeId;
            }
        }

        private static void Assert(bool v)
        {
            if (!v)
                throw new NotImplementedException();
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
                            new TextLoader.Column("F1", DataKind.R4, 1),
                            new TextLoader.Column("F3", DataKind.R4, 3),
                            new TextLoader.Column("F6", DataKind.R4, 6),
                            new TextLoader.Column("F7", DataKind.R4, 7),
                            new TextLoader.Column("F9", DataKind.R4, 9),
                        }
                    }).Read(@"F:\temp\ini\data\breast-cancer-noNan.txt");

            var pipeline = mlContext.Transforms.Concatenate("Features", "F1", "F9", "F7", "F6")
                .Append(mlContext.Regression.Trainers.GeneralizedAdditiveModels());

            var model = pipeline.Fit(idv);
            var data = model.Transform(idv);

            var roleMappedSchema = new RoleMappedSchema(data.Schema, false,
                new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, "Features"),
                new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, "Label"));

            using (StreamWriter writer = new StreamWriter(@"F:\temp\ini\model2.ini"))
                model.LastTransformer.Model.SaveAsIni(writer, roleMappedSchema);

            var results = mlContext.Regression.Evaluate(data);
        }
    }
}
