using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.ML.IsolationForest.Sample
{
    class Program
    {
        static void Main()
        {
            var ml = new MLContext(seed: 0);
            var rnd = new Random(0);

            var data = Enumerable.Range(0, 1000).Select(i =>
            {
                var x1 = (float)(rnd.NextDouble() * 0.5 - 0.25);
                var x2 = (float)(rnd.NextDouble() * 0.5 - 0.25);
                if (i % 200 == 0) { x1 = 6; x2 = 6; } // outliers
                return new Input { X1 = x1, X2 = x2 };
            }).ToList();

            var dv = ml.Data.LoadFromEnumerable(data);

            var pipeline = ml.Transforms.Concatenate("Features", nameof(Input.X1), nameof(Input.X2))
                .Append(new IsolationForestTrainer(ml, new IsolationForestTrainer.Options
                {
                    Trees = 100,
                    SampleSize = 256,
                    Contamination = 0.01
                }));

            var model = pipeline.Fit(dv);
            var scored = model.Transform(dv);

            var rows = ml.Data.CreateEnumerable<Output>(scored, reuseRowObject: false).ToList();

            Console.WriteLine($"avg score={rows.Average(r => r.Score):F4}  max score={rows.Max(r => r.Score):F4}");
            for (var i = 0; i < Math.Min(5, rows.Count); i++)
            {
                var r = rows[i];
                Console.WriteLine($"{r.X1:F2},{r.X2:F2} -> {r.Score:F4}  pred={r.PredictedLabel}");
            }
        }

        public class Input
        {
            public float X1 { get; set; }
            public float X2 { get; set; }
        }

        // Bind to emitted columns: Score, PredictedLabel (+ we keep original inputs)
        public class Output : Input
        {
            public float Score { get; set; }
            public bool PredictedLabel { get; set; }
        }
    }
}
