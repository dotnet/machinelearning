using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Transforms
{
    public static class Expression
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(0.5f, new[] { 1f, 0.2f }, 3, "hi", true, new[] { "zero", "one" }),
                new InputData(-2.7f, new[] { 3.5f, -0.1f }, 2, "bye", false, new[] { "a", "b" }),
                new InputData(1.3f, new[] { 1.9f, 3.3f }, 39, "hi", false, new[] { "0", "1" }),
                new InputData(3, new[] { 3f, 3f }, 4, "hello", true, new[] { "c", "d" }),
                new InputData(0, new[] { 1f, 1f }, 1, "hi", true, new[] { "zero", "one" }),
                new InputData(30.4f, new[] { 10f, 4f }, 9, "bye", true, new[] { "e", "f" }),
                new InputData(5.6f, new[] { 1.1f, 2.2f }, 0, "hey", false, new[] { "g", "h" }),
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline that applies various expressions to the input columns.
            var pipeline = mlContext.Transforms.Expression("Expr1", "(x,y)=>log(y)+x",
                    nameof(InputData.FloatColumn), nameof(InputData.FloatVectorColumn))
                .Append(mlContext.Transforms.Expression("Expr2", "(b,s,i)=>b ? len(s) : i",
                    nameof(InputData.BooleanColumn), nameof(InputData.StringVectorColumn), nameof(InputData.IntColumn)))
                .Append(mlContext.Transforms.Expression("Expr3", "(s,f1,f2,i)=>len(concat(s,\"a\"))+f1+f2+i",
                    nameof(InputData.StringColumn), nameof(InputData.FloatVectorColumn), nameof(InputData.FloatColumn), nameof(InputData.IntColumn)))
                .Append(mlContext.Transforms.Expression("Expr4", "(x,y)=>cos(x+pi())*y",
                    nameof(InputData.FloatColumn), nameof(InputData.IntColumn)));

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this concatenation did.
            // We can extract the newly created column as an IEnumerable of
            // TransformedData.
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
            {
                Console.Write(string.Join(" ", featureRow.Expr1));
                Console.Write(" ");
                Console.Write(string.Join(" ", featureRow.Expr2));
                Console.Write(" ");
                Console.Write(string.Join(" ", featureRow.Expr3));
                Console.Write(" ");
                Console.WriteLine(featureRow.Expr4);
            }

            // Expected output:
            //  Features column obtained post-transformation.
            //  0.5 - 1.109438 4 3 7.5 6.7 - 2.63274768567112
            //  - 1.447237 NaN 2 2 6.8 3.2 1.80814432479224
            //  1.941854 2.493922 39 39 45.2 46.6 - 10.4324561082543
            //  4.098612 4.098612 1 1 16 16 3.95996998640178
            //  0 0 4 3 5 5 - 1
            //  32.70258 31.78629 1 1 53.4 47.4 - 4.74149076052604
            //  5.69531 6.388457 0 0 10.7 11.8 0
        }

        private class InputData
        {
            public float FloatColumn;
            [VectorType(3)]
            public float[] FloatVectorColumn;
            public int IntColumn;
            public string StringColumn;
            public bool BooleanColumn;
            [VectorType(2)]
            public string[] StringVectorColumn;

            public InputData(float f, float[] fv, int i, string s, bool b, string[] sv)
            {
                FloatColumn = f;
                FloatVectorColumn = fv;
                IntColumn = i;
                StringColumn = s;
                BooleanColumn = b;
                StringVectorColumn = sv;
            }
        }

        private sealed class TransformedData
        {
            public float[] Expr1 { get; set; }
            public int[] Expr2 { get; set; }
            public float[] Expr3 { get; set; }
            public double Expr4 { get; set; }
        }
    }
}
