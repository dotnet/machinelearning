// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using System;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.StaticPipelineTesting
{
    public sealed class Training : BaseTestClassWithConsole
    {
        public Training(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void SdcaRegression()
        {
            var env = new TlcEnvironment(seed: 0);
            var dataPath = GetDataPath("external", "winequality-white.csv");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: r.label.PredictSdcaRegression(r.features, maxIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            Assert.Equal(11, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = RegressionEvaluator.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.L1, 0, double.PositiveInfinity);
            Assert.InRange(metrics.L2, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Rms, 0, double.PositiveInfinity);
            Assert.Equal(metrics.Rms * metrics.Rms, metrics.L2, 5);
            Assert.InRange(metrics.LossFn, 0, double.PositiveInfinity);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");
        }

        [Fact]
        public void SdcaRegressionNameCollision()
        {
            var env = new TlcEnvironment(seed: 0);
            var dataPath = GetDataPath("external", "winequality-white.csv");
            var dataSource = new MultiFileSource(dataPath);

            // Here we introduce another column called "Score" to collide with the name of the default output. Heh heh heh...
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10), Score: c.LoadText(2)),
                separator: ';', hasHeader: true);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.Score, score: r.label.PredictSdcaRegression(r.features, maxIterations: 2)));

            var pipe = reader.Append(est);

            var model = pipe.Fit(dataSource);
            var data = model.Read(dataSource);

            // Now, let's see if that column is still there, and still text!
            var schema = data.AsDynamic.Schema;
            Assert.True(schema.TryGetColumnIndex("Score", out int scoreCol), "Score column not present!");
            Assert.Equal(TextType.Instance, schema.GetColumnType(scoreCol));

            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");
        }

        [Fact]
        public void SdcaBinaryClassification()
        {
            var env = new TlcEnvironment(seed: 0);
            var dataPath = GetDataPath("breast-cancer.txt");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryPredictor pred = null;
            ParameterMixingCalibratedPredictor cali = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: r.label.PredictSdcaBinaryClassification(r.features,
                    maxIterations: 2,
                    onFit: (p, c) => { pred = p; cali = c; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            Assert.Null(cali);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            Assert.NotNull(cali);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = BinaryClassifierEvaluator.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
            Assert.InRange(metrics.LogLoss, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Entropy, 0, double.PositiveInfinity);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");
        }

        [Fact]
        public void SdcaBinaryClassificationNoClaibration()
        {
            var env = new TlcEnvironment(seed: 0);
            var dataPath = GetDataPath("breast-cancer.txt");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryPredictor pred = null;

            var loss = new HingeLoss(new HingeLoss.Arguments() { Margin = 1 });

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: r.label.PredictSdcaBinaryClassification(r.features,
                maxIterations: 2,
                loss: loss, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = BinaryClassifierEvaluator.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");
        }
    }
}
