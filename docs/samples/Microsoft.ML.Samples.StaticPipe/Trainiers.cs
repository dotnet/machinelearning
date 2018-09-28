// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Samples.StaticPipe
{
    public class Trainers
    {
        public static void Main()
        {
            SdcaRegression();
        }

        public static void SdcaRegression()
        {
            // writting a small sample dataset to file
            string dataPath = @"c:\temp\MyTest.txt";

            float a = 0;
            float b = 0;
            float d = 0;
            float target = 0;

            var csvContent = new StringBuilder("feature_a, feature_b, feature_c, target");
            Random rnd = new Random();

            for (int i=0; i< 1000; i++)
            {
                a = rnd.Next(i-3, i+3);
                b = 2 * rnd.Next(i - 2, i + 2);
                d = rnd.Next(i - 5, i + 5);

                var newLine = string.Format($"{a}, {b}, {d}, {target}");
                csvContent.AppendLine(newLine);
            }

            if (!File.Exists(dataPath))
                File.WriteAllText(dataPath, csvContent.ToString());
            else
                Console.WriteLine("Change the dataPath, a file with that path already exists.");

            var env = new ConsoleEnvironment(seed: 0);
            // the file 
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.Sdca(r.label, r.features, maxIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            var model = pipe.Fit(dataSource);
            // 11 input features, so we ought to have 11 weights.
            // Assert.Equal(11, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int cc = 0; cc < schema.ColumnCount; ++cc)
                Console.WriteLine($"{schema.GetColumnName(cc)}, {schema.GetColumnType(cc)}");
        }
    }
}
