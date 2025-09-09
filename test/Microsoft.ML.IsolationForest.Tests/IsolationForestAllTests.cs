// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Xunit;

#nullable enable

namespace Microsoft.ML.IsolationForest.Tests
{
    public sealed class IsolationForestAllTests
    {
        // ---------------------------
        // Helpers
        // ---------------------------

        private static IReadOnlyList<double[]> MakeToyData(int n = 64, int d = 3, int seed = 1, double outlierShift = 6.0)
        {
            var rng = new Random(seed);
            var data = new List<double[]>(n);
            // Mostly “normal”
            for (int i = 0; i < n - 2; i++)
            {
                var row = new double[d];
                for (int j = 0; j < d; j++) row[j] = rng.NextDouble() * 0.4 - 0.2; // ~[-0.2, 0.2]
                data.Add(row);
            }
            // A couple of outliers
            data.Add(Enumerable.Repeat(outlierShift, d).ToArray());
            data.Add(Enumerable.Repeat(outlierShift * 1.2, d).ToArray());
            return data;
        }

        private static double[] ColumnMedians(IReadOnlyList<double[]> x)
        {
            int d = x[0].Length;
            var med = new double[d];
            var col = new double[x.Count];
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < x.Count; i++) col[i] = x[i][j];
                Array.Sort(col);
                int n = col.Length;
                med[j] = (n % 2 == 1) ? col[n / 2] : 0.5 * (col[n / 2 - 1] + col[n / 2]);
            }
            return med;
        }

        private static IReadOnlyList<double[]> TinyData_AllEqual(int n = 5, int d = 3, double v = 0.0)
        {
            var row = Enumerable.Repeat(v, d).ToArray();
            return Enumerable.Range(0, n).Select(_ => (double[])row.Clone()).ToArray();
        }

        private static bool IsFinite(double v) => !(double.IsNaN(v) || double.IsInfinity(v));

        // ---------------------------
        // Core model tests
        // ---------------------------

        [Fact]
        public void Model_Fit_Score_Deterministic_WithSeed()
        {
            var x = MakeToyData();
            var m1 = new IsolationForestModel(nTrees: 64, sampleSize: 32, seed: 7);
            var m2 = new IsolationForestModel(nTrees: 64, sampleSize: 32, seed: 7);

            m1.Fit(x, parallel: true);
            m2.Fit(x, parallel: true);

            // Compare scores for a few rows – must be identical with same seed.
            for (int i = 0; i < 5; i++)
            {
                var s1 = m1.Score(x[i]);
                var s2 = m2.Score(x[i]);
                Assert.Equal(s1, s2, precision: 12);
            }
        }

        [Fact]
        public void Model_Throws_BeforeFit_And_OnEmpty()
        {
            var model = new IsolationForestModel();

            Assert.Throws<InvalidOperationException>(() => model.Score(new double[] { 0, 0 }));
            Assert.Throws<ArgumentException>(() => model.Fit(Array.Empty<double[]>(), parallel: false));
        }

        [Fact]
        public void Model_ScaledScore_ClampedAndUnclamped_Behavior()
        {
            var x = MakeToyData(n: 32, d: 2, seed: 3);
            var model = new IsolationForestModel(nTrees: 50, sampleSize: 16, seed: 3);
            model.Fit(x, parallel: false);

            var row = x[0];
            var sClamp = model.ScaledScore0To100(row, clipTo100: true);
            var sNoClamp = model.ScaledScore0To100(row, clipTo100: false);

            Assert.InRange(sClamp, 0.0, 100.0);
            Assert.True(IsFinite(sNoClamp));
        }

        [Fact]
        public void Model_Directionality_Boosts_When_Satisfied()
        {
            var x = MakeToyData(n: 48, d: 3, seed: 11, outlierShift: 4.5);
            var model = new IsolationForestModel(nTrees: 60, sampleSize: 24, seed: 11);
            model.Fit(x, parallel: true);

            var med = ColumnMedians(x);
            var hi = new HashSet<int> { 0, 1, 2 };

            var rowHi = x[x.Count - 1];

            var baseOnly = model.ScaledScore0To100(rowHi, clipTo100: true);
            var withDir = model.ScaledScore0To100WithDirectionality(rowHi, med, hi, shouldBeLowIdx: null, clipTo100: true);

            Assert.True(withDir >= baseOnly - 1e-9);
        }

        [Fact]
        public void Model_ShapScaler_IsFinite_AndBounded()
        {
            var x = MakeToyData(n: 40, d: 4, seed: 5, outlierShift: 5.5);
            var model = new IsolationForestModel(nTrees: 80, sampleSize: 32, seed: 5);
            model.Fit(x, parallel: true);

            var med = ColumnMedians(x);
            var hi = new HashSet<int> { 0, 1 };
            var lo = new HashSet<int> { 2 }; // mixed

            foreach (var row in x.Take(5))
            {
                var y = model.ScaledScore0To100FromShap(row, med, hi, lo, clipTo100: true);
                Assert.InRange(y, 0.0, 100.0);
            }
        }

        [Fact]
        public void Model_ParallelFalse_ProducesReasonableScores()
        {
            var x = MakeToyData(n: 24, d: 2, seed: 13);
            var model = new IsolationForestModel(nTrees: 40, sampleSize: 16, seed: 13);
            model.Fit(x, parallel: false);

            var scores = x.Select(model.Score).ToArray();
            Assert.All(scores, s => Assert.True(s > 0.0 && s <= 1.0));
        }

        // ---------------------------
        // Trainer end-to-end tests
        // ---------------------------

        [Fact]
        public void Trainer_EndToEnd_DefaultOutputs_Work()
        {
            var ml = new MLContext(seed: 123);
            var rows = new[]
            {
                new { A = 0f, B = 0f },
                new { A = 0.1f, B = -0.1f },
                new { A = 5.0f, B = 5.0f }
            };

            var dv = ml.Data.LoadFromEnumerable(rows);

            // Build features vector “Features”
            var concat = ml.Transforms.Concatenate("Features", "A", "B");
            var dv2 = concat.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer();
            trainer.Fit(dv2, ml);

            var pipe = trainer.CreatePipeline(ml);
            var model = pipe.Fit(dv2);
            var scored = model.Transform(dv2);

            var outRows = ml.Data.CreateEnumerable<OutRow>(scored, reuseRowObject: false).ToList();

            Assert.Equal(3, outRows.Count);
            Assert.All(outRows, r => Assert.True(r.Score >= 0f)); // sample check
        }

        [Fact]
        public void Trainer_RenamedOutputs_And_Thresholding_Work()
        {
            var ml = new MLContext(seed: 321);
            var rows = Enumerable.Range(0, 20)
                                 .Select(i => new { A = (float)(i < 18 ? i * 0.01 : 5.0), B = (float)(i < 18 ? i * -0.01 : 5.0) })
                                 .ToList();
            var dv = ml.Data.LoadFromEnumerable(rows);

            var concat = ml.Transforms.Concatenate("Features", "A", "B");
            var dv2 = concat.Fit(dv).Transform(dv);

            var trainer = new IsolationForestTrainer(new IsolationForestTrainer.Options
            {
                ScoreColumnName = "IF_Score",
                PredictedLabelColumnName = "IF_Label",
                Trees = 50,
                SampleSize = 32,
                MatchThreshold = 50f // 0..100 scale
            });

            trainer.Fit(dv2, ml);
            var pipe = trainer.CreatePipeline(ml);
            var model = pipe.Fit(dv2);
            var scored = model.Transform(dv2);

            var ifScore = scored.GetColumn<float>(ml, "IF_Score").ToArray();
            var ifLabel = scored.GetColumn<bool>(ml, "IF_Label").ToArray();

            Assert.Equal(rows.Count, ifScore.Length);
            Assert.Equal(rows.Count, ifLabel.Length);
            Assert.All(ifScore, s => Assert.InRange(s, 0f, 100f));
            Assert.Contains(ifLabel, b => b);
        }

        // ---------------------------
        // Micro-tests to tick psi-floor paths & tiny-set edges
        // ---------------------------

        [Fact]
        public void PsiFloor_LinearScaler_Differs_WhenPsiSmall()
        {
            // nTrain (psi) intentionally small (< 20) to tick PsiMinForFirstSplitLinear logic.
            var x = MakeToyData(n: 6, d: 3, seed: 9, outlierShift: 3.5);
            var model = new IsolationForestModel(nTrees: 32, sampleSize: 6, seed: 9);
            model.Fit(x, parallel: true);

            var med = ColumnMedians(x);
            var emptyHi = new HashSet<int>();   // totalMarked == 0 => returns the "base score" (NO psi floor)
            var row = x[0];

            // Base (NO psi floor) via directionality path with no marked features.
            var baseNoFloor = model.ScaledScore0To100WithDirectionality(
                row, med, shouldBeHighIdx: emptyHi, shouldBeLowIdx: null, clipTo100: false);

            // Linear scaler WITH psi floor.
            var linearWithFloor = model.ScaledScore0To100(row, clipTo100: false);

            // With psi < 20 we expect a measurable difference.
            Assert.NotEqual(baseNoFloor, linearWithFloor, precision: 6);
        }

        [Fact]
        public void PsiFloor_LinearScaler_Matches_WhenPsiLarge()
        {
            // nTrain (psi) >= 20 so floor should not change the result materially.
            var x = MakeToyData(n: 64, d: 2, seed: 10);
            var model = new IsolationForestModel(nTrees: 64, sampleSize: 64, seed: 10);
            model.Fit(x, parallel: false);

            var med = ColumnMedians(x);
            var emptyHi = new HashSet<int>();
            var row = x[1];

            var baseNoFloor = model.ScaledScore0To100WithDirectionality(
                row, med, shouldBeHighIdx: emptyHi, shouldBeLowIdx: null, clipTo100: false);

            var linearWithFloor = model.ScaledScore0To100(row, clipTo100: false);

            // Should be very close when psi >= 20
            Assert.Equal(baseNoFloor, linearWithFloor, precision: 6);
        }

        [Fact]
        public void PsiFloor_ShapScaler_Bounded_WhenPsiSmall()
        {
            // Small psi will force SHAP scaler to use interpolation between C(18) and C(19).
            var x = MakeToyData(n: 8, d: 4, seed: 15, outlierShift: 4.0);
            var model = new IsolationForestModel(nTrees: 40, sampleSize: 8, seed: 15);
            model.Fit(x, parallel: true);

            var med = ColumnMedians(x);
            var hi = new HashSet<int> { 0, 2 };
            var lo = new HashSet<int> { 1 };

            var y = model.ScaledScore0To100FromShap(x[0], med, hi, lo, clipTo100: true);
            Assert.InRange(y, 0.0, 100.0);
        }

        [Fact]
        public void Degenerate_AllEqualData_ProducesFiniteScores()
        {
            var x = TinyData_AllEqual(n: 5, d: 3, v: 0.0);
            var model = new IsolationForestModel(nTrees: 20, sampleSize: 5, seed: 21);
            model.Fit(x, parallel: false);

            foreach (var row in x)
            {
                var s = model.ScaledScore0To100(row, clipTo100: true);
                var aph = model.NormalizedAveragePathLength(row);
                Assert.InRange(s, 0.0, 100.0);
                Assert.True(IsFinite(aph));
            }
        }

        [Fact]
        public void PsiTiny_ParallelAndSerial_Comparable()
        {
            var x = MakeToyData(n: 10, d: 3, seed: 4, outlierShift: 3.0);

            var mPar = new IsolationForestModel(nTrees: 32, sampleSize: 10, seed: 4);
            mPar.Fit(x, parallel: true);

            var mSer = new IsolationForestModel(nTrees: 32, sampleSize: 10, seed: 4);
            mSer.Fit(x, parallel: false);

            // Compare a handful of rows; allow tiny drift.
            for (int i = 0; i < Math.Min(5, x.Count); i++)
            {
                var a = mPar.ScaledScore0To100(x[i], clipTo100: true);
                var b = mSer.ScaledScore0To100(x[i], clipTo100: true);
                Assert.InRange(Math.Abs(a - b), 0.0, 1e-6);
            }
        }
    }

    internal static class DataViewExtensions
    {
        public static IEnumerable<T> GetColumn<T>(this IDataView dv, MLContext ml, string columnName)
        {
            var col = dv.Schema[columnName];
            using var cursor = dv.GetRowCursor(dv.Schema);
            var getter = cursor.GetGetter<T>(col);
            T buffer = default!; // Use null-forgiving operator to suppress CS8601
            while (cursor.MoveNext())
            {
                getter(ref buffer);
                yield return buffer!;
            }
        }
    }

    sealed class OutRow
    {
        public float Score { get; set; }
        public bool PredictedLabel { get; set; }
    }
}
