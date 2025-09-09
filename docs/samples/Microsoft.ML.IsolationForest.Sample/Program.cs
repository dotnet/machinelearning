// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

#nullable enable

namespace Microsoft.ML.IsolationForest.Sample
{
    internal static class Program
    {
        private static void Main()
        {
            Console.WriteLine("=== Microsoft.ML.IsolationForest Samples ===");

            RunWebLatencySample();
            RunIoTSensorSample();
            RunRetailPurchasesSample();
            RunManufacturingProcessSample();
            RunSupportTicketsSample();

            Console.WriteLine("Done.");
        }

        // --------------------------------------------------------------------
        // 1) Website performance: P95 latency (ms) + error rate (%)
        //    Higher values should push the anomaly score higher.
        // --------------------------------------------------------------------
        private static void RunWebLatencySample()
        {
            Console.WriteLine("\n[Web Latency + Error Rate]");
            var ml = new MLContext(seed: 42);

            // 1) Synthetic data: mostly calm, a couple of spikes
            var data = new[]
            {
                new { P95Ms = 220f, ErrorRatePct = 0.2f },
                new { P95Ms = 180f, ErrorRatePct = 0.1f },
                new { P95Ms = 240f, ErrorRatePct = 0.3f },
                new { P95Ms = 210f, ErrorRatePct = 0.2f },
                new { P95Ms = 950f, ErrorRatePct = 5f   }, // anomaly
                new { P95Ms = 1200f, ErrorRatePct = 10f }  // anomaly
            };
            var dv = ml.Data.LoadFromEnumerable(data);

            // 2) Feature engineering
            var features = ml.Transforms.Concatenate("Features", nameof(WebOut.P95Ms), nameof(WebOut.ErrorRatePct));
            var dvFeats = features.Fit(dv).Transform(dv);

            // 3) Train Isolation Forest (custom output names to match WebOut)
            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                Trees = 100,
                SampleSize = 64,
                ScoreColumnName = nameof(WebOut.IF_Score),
                PredictedLabelColumnName = nameof(WebOut.IF_Label),
                MatchThreshold = 50f
            });

            // Directionality: both metrics "should be low" (i.e., high is suspicious)
            // We pass indices of features that are expected to be high/low relative to medians.
            var shouldBeHigh = new HashSet<int> { 0, 1 }; // P95Ms, ErrorRatePct
            trainer.SetDirectionality(shouldBeHigh, shouldBeLow: null);

            trainer.Fit(dvFeats, ml);
            var pipeline = trainer.CreatePipeline(ml);
            var model = pipeline.Fit(dvFeats);
            var scored = model.Transform(dvFeats);

            var rows = ml.Data.CreateEnumerable<WebOut>(scored, reuseRowObject: false).ToList();
            foreach (var r in rows)
                Console.WriteLine($"P95={r.P95Ms,6:0} ms  Err={r.ErrorRatePct,4:0.0}%  IF_Score={r.IF_Score,6:0.0}  IF_Label={r.IF_Label}");
        }

        // --------------------------------------------------------------------
        // 2) IoT / sensors: temperature, humidity, pressure, vibration
        //    Higher temperature & vibration are suspicious.
        // --------------------------------------------------------------------
        private static void RunIoTSensorSample()
        {
            Console.WriteLine("\n[IoT Sensors]");
            var ml = new MLContext(seed: 123);

            var rng = new Random(1);
            var normal = Enumerable.Range(0, 30).Select(_ => new
            {
                TemperatureC = 22f + (float)(rng.NextDouble() * 1.2 - 0.6),
                HumidityPct = 45f + (float)(rng.NextDouble() * 5 - 2.5),
                PressureKPa = 101.0f + (float)(rng.NextDouble() * 0.8 - 0.4),
                VibrationMmS = 1.2f + (float)(rng.NextDouble() * 0.4 - 0.2),
            });

            var anomalies = new[]
            {
                new { TemperatureC = 35f, HumidityPct = 30f, PressureKPa = 99.0f, VibrationMmS = 4.5f },
                new { TemperatureC = 38f, HumidityPct = 65f, PressureKPa = 102.5f, VibrationMmS = 5.0f },
            };

            var all = normal.Concat(anomalies).ToList();
            var dv = ml.Data.LoadFromEnumerable(all);

            var features = ml.Transforms.Concatenate("Features",
                nameof(IoTOut.TemperatureC), nameof(IoTOut.HumidityPct),
                nameof(IoTOut.PressureKPa), nameof(IoTOut.VibrationMmS));
            var dvFeats = features.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                Trees = 100,
                SampleSize = 64,
                ScoreColumnName = nameof(IoTOut.IF_Score),
                PredictedLabelColumnName = nameof(IoTOut.IF_Label),
                MatchThreshold = 60f
            });

            // Indices 0:TemperatureC, 3:VibrationMmS should be high when anomalous
            trainer.SetDirectionality([0, 3]);

            trainer.Fit(dvFeats, ml);
            var pipeline = trainer.CreatePipeline(ml);
            var model = pipeline.Fit(dvFeats);
            var scored = model.Transform(dvFeats);

            var rows = ml.Data.CreateEnumerable<IoTOut>(scored, reuseRowObject: false).ToList();
            foreach (var r in rows.Take(10))
                Console.WriteLine($"T={r.TemperatureC,5:0.0}C  H={r.HumidityPct,5:0.0}%  Vib={r.VibrationMmS,4:0.0}  IF_Score={r.IF_Score,6:0.0}  IF_Label={r.IF_Label}");
        }

        // --------------------------------------------------------------------
        // 3) Retail purchases: amount, item count, customer tenure days
        //    Amount & item count high are suspicious. Tenure low may be suspicious,
        //    but we’ll keep it neutral in this simple example.
        // --------------------------------------------------------------------
        private static void RunRetailPurchasesSample()
        {
            Console.WriteLine("\n[Retail Purchases]");
            var ml = new MLContext(seed: 77);

            var data = new List<PurchaseIn>();
            for (var i = 0; i < 50; i++)
            {
                data.Add(new PurchaseIn
                {
                    Amount = 20f + i * 0.3f,
                    ItemCount = 1 + (i % 4),
                    CustomerTenureDays = 200 + i
                });
            }
            // Anomalies
            data.Add(new PurchaseIn { Amount = 950f, ItemCount = 35f, CustomerTenureDays = 10f });
            data.Add(new PurchaseIn { Amount = 1200f, ItemCount = 1f, CustomerTenureDays = 3f });

            var dv = ml.Data.LoadFromEnumerable(data);

            var features = ml.Transforms.Concatenate("Features",
                nameof(PurchaseOut.Amount), nameof(PurchaseOut.ItemCount), nameof(PurchaseOut.CustomerTenureDays));
            var dvFeats = features.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                Trees = 120,
                SampleSize = 64,
                ScoreColumnName = nameof(PurchaseOut.IF_Score),
                PredictedLabelColumnName = nameof(PurchaseOut.IF_Label),
                MatchThreshold = 50f
            });

            // High: Amount (0), ItemCount (1)
            trainer.SetDirectionality([0, 1]);

            trainer.Fit(dvFeats, ml);
            var pipe = trainer.CreatePipeline(ml);
            var model = pipe.Fit(dvFeats);
            var scored = model.Transform(dvFeats);

            var rows = ml.Data.CreateEnumerable<PurchaseOut>(scored, reuseRowObject: false).ToList();
            foreach (var r in rows.Skip(rows.Count - 5))
                Console.WriteLine($"Amount={r.Amount,7:0.0}  Items={r.ItemCount,4:0}  Tenure={r.CustomerTenureDays,4:0}d  IF_Score={r.IF_Score,6:0.0}  IF_Label={r.IF_Label}");
        }

        // --------------------------------------------------------------------
        // 4) Manufacturing process: speed, torque, temp, humidity
        //    High temperature + torque spikes are suspicious.
        // --------------------------------------------------------------------
        private static void RunManufacturingProcessSample()
        {
            Console.WriteLine("\n[Manufacturing / Process Control]");
            var ml = new MLContext(seed: 88);

            var normal = Enumerable.Range(0, 40).Select(i => new ProcessIn
            {
                SpeedRpm = 1500f + (i % 4) * 5f,
                TorqueNm = 45f + (i % 3) * 0.3f,
                TempC = 60f + (i % 5) * 0.2f,
                HumidityPct = 35f + (i % 2) * 0.4f
            });

            var anomalies = new[]
            {
                new ProcessIn { SpeedRpm = 1600f, TorqueNm = 80f, TempC = 90f, HumidityPct = 50f },
                new ProcessIn { SpeedRpm = 1400f, TorqueNm = 5f, TempC = 95f, HumidityPct = 20f }
            };

            var all = normal.Concat(anomalies).ToList();
            var dv = ml.Data.LoadFromEnumerable(all);

            var features = ml.Transforms.Concatenate("Features",
                nameof(ProcessOut.SpeedRpm), nameof(ProcessOut.TorqueNm),
                nameof(ProcessOut.TempC), nameof(ProcessOut.HumidityPct));
            var dvFeats = features.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                Trees = 100,
                SampleSize = 64,
                ScoreColumnName = nameof(ProcessOut.IF_Score),
                PredictedLabelColumnName = nameof(ProcessOut.IF_Label),
                MatchThreshold = 55f
            });

            // High: Torque (1), Temp (2)
            trainer.SetDirectionality([1, 2]);

            trainer.Fit(dvFeats, ml);
            var pipe = trainer.CreatePipeline(ml);
            var model = pipe.Fit(dvFeats);
            var scored = model.Transform(dvFeats);

            var rows = ml.Data.CreateEnumerable<ProcessOut>(scored, reuseRowObject: false).ToList();
            foreach (var r in rows.Skip(rows.Count - 6))
                Console.WriteLine($"Speed={r.SpeedRpm,6:0} rpm  Torque={r.TorqueNm,5:0.0}  Temp={r.TempC,5:0.0}C  IF_Score={r.IF_Score,6:0.0}  IF_Label={r.IF_Label}");
        }

        // --------------------------------------------------------------------
        // 5) Support tickets: length, sentiment, hour-of-day
        //    Very long + very negative may be suspicious (workload spikes, etc.)
        // --------------------------------------------------------------------
        private static void RunSupportTicketsSample()
        {
            Console.WriteLine("\n[Support Tickets]");
            var ml = new MLContext(seed: 5);

            var normal = Enumerable.Range(0, 25).Select(i => new
            {
                Words = 40f + (float)(i % 6) * 5f,         // ~40-65 words
                Sentiment = 0.6f + (float)(i % 5) * 0.05f, // mildly positive
                HourOfDay = (float)((i * 3) % 24)
            });

            var anomalies = new[]
            {
                new { Words = 450f, Sentiment = -0.9f, HourOfDay = 2f  },
                new { Words = 520f, Sentiment = -0.8f, HourOfDay = 23f }
            };

            var all = normal.Concat(anomalies).ToList();
            var dv = ml.Data.LoadFromEnumerable(all);

            var features = ml.Transforms.Concatenate("Features",
                nameof(TicketOut.Words), nameof(TicketOut.Sentiment), nameof(TicketOut.HourOfDay));
            var dvFeats = features.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                Trees = 80,
                SampleSize = 64,
                ScoreColumnName = nameof(TicketOut.IF_Score),
                PredictedLabelColumnName = nameof(TicketOut.IF_Label),
                MatchThreshold = 60f
            });

            // High words (0) suspicious; low sentiment handled implicitly by tree structure
            trainer.SetDirectionality([0]);

            trainer.Fit(dvFeats, ml);
            var pipe = trainer.CreatePipeline(ml);
            var model = pipe.Fit(dvFeats);
            var scored = model.Transform(dvFeats);

            var rows = ml.Data.CreateEnumerable<TicketOut>(scored, reuseRowObject: false).ToList();
            foreach (var r in rows.TakeLast(6))
                Console.WriteLine($"Words={r.Words,5:0}  Sent={r.Sentiment,5:0.00}  Hour={r.HourOfDay,2:0}  IF_Score={r.IF_Score,6:0.0}  IF_Label={r.IF_Label}");
        }

        // --------------------------------------------------------------------
        // Input schemas for samples (kept minimal and local)
        // --------------------------------------------------------------------
        private sealed class PurchaseIn
        {
            public float Amount { get; set; }
            public float ItemCount { get; set; }
            public float CustomerTenureDays { get; set; }
        }

        private sealed class ProcessIn
        {
            public float SpeedRpm { get; set; }
            public float TorqueNm { get; set; }
            public float TempC { get; set; }
            public float HumidityPct { get; set; }
        }
    }
}
