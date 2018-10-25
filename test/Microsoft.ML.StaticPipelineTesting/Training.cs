// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using System;
using System.Linq;
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
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.Sdca(r.label, r.features, maxIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            Assert.Equal(11, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
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
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new RegressionContext(env);

            // Here we introduce another column called "Score" to collide with the name of the default output. Heh heh heh...
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10), Score: c.LoadText(2)),
                separator: ';', hasHeader: true);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.Score, score: ctx.Trainers.Sdca(r.label, r.features, maxIterations: 2)));

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
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryPredictor pred = null;
            ParameterMixingCalibratedPredictor cali = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(r.label, r.features,
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

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
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
        public void SdcaBinaryClassificationNoCalibration()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryPredictor pred = null;

            var loss = new HingeLoss(new HingeLoss.Arguments() { Margin = 1 });

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(r.label, r.features,
                maxIterations: 2,
                loss: loss, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");
        }

        [Fact]
        public void AveragePerceptronNoCalibration()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryPredictor pred = null;

            var loss = new HingeLoss(new HingeLoss.Arguments() { Margin = 1 });

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.AveragedPerceptron(r.label, r.features, lossFunction: loss,
                numIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights2.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [Fact]
        public void FfmBinaryClassification()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features1: c.LoadFloat(1, 4), features2: c.LoadFloat(5, 9)));

            FieldAwareFactorizationMachinePredictor pred = null;

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.FieldAwareFactorizationMachine(r.label, new[] { r.features1, r.features2 }, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [Fact]
        public void SdcaMulticlass()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MulticlassLogisticRegressionPredictor pred = null;

            var loss = new HingeLoss(new HingeLoss.Arguments() { Margin = 1 });

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(
                    r.label,
                    r.features,
                    maxIterations: 2,
                    loss: loss, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            VBuffer<float>[] weights = default;
            pred.GetWeights(ref weights, out int n);
            Assert.True(n == 3 && n == weights.Length);
            foreach (var w in weights)
                Assert.True(w.Length == 4);

            var biases = pred.GetBiases();
            Assert.True(biases.Count() == 3);

            var data = model.Read(dataSource);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void CrossValidate()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(
                    r.label,
                    r.features,
                    maxIterations: 2)));

            var results = ctx.CrossValidate(reader.Read(dataSource), est, r => r.label)
                .Select(x => x.metrics).ToArray();
            Assert.Equal(5, results.Length);
            Assert.True(results.All(x => x.LogLoss > 0));
        }

        [Fact]
        public void FastTreeBinaryClassification()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.FastTree(r.label, r.features,
                    numTrees: 10,
                    numLeaves: 5,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            // 9 input features, so we ought to have 9 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(9, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [Fact]
        public void FastTreeRegression()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            FastTreeRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.FastTree(r.label, r.features,
                    numTrees: 10,
                    numLeaves: 5,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(11, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.L1, 0, double.PositiveInfinity);
            Assert.InRange(metrics.L2, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Rms, 0, double.PositiveInfinity);
            Assert.Equal(metrics.Rms * metrics.Rms, metrics.L2, 5);
            Assert.InRange(metrics.LossFn, 0, double.PositiveInfinity);
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmBinaryClassification()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.LightGbm(r.label, r.features,
                    numBoostRound: 10,
                    numLeaves: 5,
                    learningRate: 0.01,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            // 9 input features, so we ought to have 9 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(9, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmRegression()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LightGbmRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.LightGbm(r.label, r.features,
                    numBoostRound: 10,
                    numLeaves: 5,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(11, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.L1, 0, double.PositiveInfinity);
            Assert.InRange(metrics.L2, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Rms, 0, double.PositiveInfinity);
            Assert.Equal(metrics.Rms * metrics.Rms, metrics.L2, 5);
            Assert.InRange(metrics.LossFn, 0, double.PositiveInfinity);
        }

        [Fact]
        public void PoissonRegression()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            PoissonRegressionPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.PoissonRegression(r.label, r.features,
                    l1Weight: 2,
                    enoforceNoNegativity: true,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(11, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.L1, 0, double.PositiveInfinity);
            Assert.InRange(metrics.L2, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Rms, 0, double.PositiveInfinity);
            Assert.Equal(metrics.Rms * metrics.Rms, metrics.L2, 5);
            Assert.InRange(metrics.LossFn, 0, double.PositiveInfinity);
        }

        [Fact]
        public void LogisticRegressionBinaryClassification()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.LogisticRegressionBinaryClassifier(r.label, r.features,
                    l1Weight: 10,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            // 9 input features, so we ought to have 9 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(9, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [Fact]
        public void MulticlassLogisticRegression()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MulticlassLogisticRegressionPredictor pred = null;

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.MultiClassLogisticRegression(
                    r.label,
                    r.features, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            VBuffer<float>[] weights = default;
            pred.GetWeights(ref weights, out int n);
            Assert.True(n == 3 && n == weights.Length);
            foreach (var w in weights)
                Assert.True(w.Length == 4);

            var data = model.Read(dataSource);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void OnlineGradientDescent()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionPredictor pred = null;

            var loss = new SquaredLoss();

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.OnlineGradientDescent(r.label, r.features,
                lossFunction:loss,
                onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(11, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.score, new PoissonLoss());
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.L1, 0, double.PositiveInfinity);
            Assert.InRange(metrics.L2, 0, double.PositiveInfinity);
            Assert.InRange(metrics.Rms, 0, double.PositiveInfinity);
            Assert.Equal(metrics.Rms * metrics.Rms, metrics.L2, 5);
            Assert.InRange(metrics.LossFn, 0, double.PositiveInfinity);
        }

        [Fact]
        public void KMeans()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new ClusteringContext(env);
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            KMeansPredictor pred = null;

            var est = reader.MakeNewEstimator()
                 .Append(r => (label: r.label.ToKey(), r.features))
                 .Append(r => (r.label, r.features, preds: ctx.Trainers.KMeans(r.features, clustersCount: 3, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            VBuffer<float>[] centroids = default;
            int k;
            pred.GetClusterCentroids(ref centroids, out k);

            Assert.True(k == 3);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.preds.score, r => r.label, r => r.features);
            Assert.NotNull(metrics);

            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.InRange(metrics.Nmi, 0.73, 0.77);
            Assert.InRange(metrics.Dbi, 0.662, 0.667);

            metrics = ctx.Evaluate(data, r => r.preds.score, label: r => r.label);
            Assert.NotNull(metrics);

            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.True(metrics.Dbi == 0.0);

            metrics = ctx.Evaluate(data, r => r.preds.score, features: r => r.features);
            Assert.True(double.IsNaN(metrics.Nmi));

            metrics = ctx.Evaluate(data, r => r.preds.score);
            Assert.NotNull(metrics);
            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.True(double.IsNaN(metrics.Nmi));
            Assert.True(metrics.Dbi == 0.0);

        }

        [Fact]
        public void FastTreeRanking()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.adultRanking.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RankingContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(0), features: c.LoadFloat(9, 14), groupId: c.LoadText(1)),
                separator: '\t', hasHeader: true);

            FastTreeRankingPredictor pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.features, groupId: r.groupId.ToKey()))
                .Append(r => (r.label, r.groupId, score: ctx.Trainers.FastTree(r.label, r.features, r.groupId, onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.groupId, r => r.score);
            Assert.NotNull(metrics);

            Assert.True(metrics.Ndcg.Length == metrics.Dcg.Length && metrics.Dcg.Length == 3);

            Assert.InRange(metrics.Dcg[0], 1.4, 1.6);
            Assert.InRange(metrics.Dcg[1], 1.4, 1.8);
            Assert.InRange(metrics.Dcg[2], 1.4, 1.8);

            Assert.InRange(metrics.Ndcg[0], 36.5, 37);
            Assert.InRange(metrics.Ndcg[1], 36.5, 37);
            Assert.InRange(metrics.Ndcg[2], 36.5, 37);
        }

        [Fact]
        public void MultiClassNaiveBayesTrainer()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MultiClassNaiveBayesPredictor pred = null;

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.MultiClassNaiveBayesTrainer(
                    r.label,
                    r.features, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            int[] labelHistogram = default;
            int[][] featureHistogram = default;
            pred.GetLabelHistogram(ref labelHistogram, out int labelCount1);
            pred.GetFeatureHistogram(ref featureHistogram, out int labelCount2, out int featureCount);
            Assert.True(labelCount1 == 3 && labelCount1 == labelCount2 && labelCount1 <= labelHistogram.Length);
            for (int i = 0; i < labelCount1; i++)
                Assert.True(featureCount == 4 && (featureCount <= featureHistogram[i].Length));

            var data = model.Read(dataSource);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
                Console.WriteLine($"{schema.GetColumnName(c)}, {schema.GetColumnType(c)}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void HogwildSGDBinaryClassification()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.StochasticGradientDescentClassificationTrainer(r.label, r.features,
                    l2Weight: 0,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            // 9 input features, so we ought to have 9 weights.
            VBuffer<float> weights = new VBuffer<float>();
            pred.GetFeatureWeights(ref weights);
            Assert.Equal(9, weights.Length);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }
    }
}
