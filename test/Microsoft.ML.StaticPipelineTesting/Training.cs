// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.FactorizationMachine;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Learners;
using Microsoft.ML.LightGBM;
using Microsoft.ML.LightGBM.StaticPipe;
using Microsoft.ML.RunTests;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Trainers.Recommender;
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
            var env = new MLContext(seed: 0, conc: 1);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.Sdca(r.label, r.features, maxIterations: 2,
                onFit: p => pred = p, advancedSettings: s => s.NumThreads = 1)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 11 input features, so we ought to have 11 weights.
            Assert.Equal(11, pred.Weights.Count);

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
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");
        }

        [Fact]
        public void SdcaRegressionNameCollision()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new RegressionContext(env);

            // Here we introduce another column called "Score" to collide with the name of the default output. Heh heh heh...
            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10), Score: c.LoadText(2)),
                separator: ';', hasHeader: true);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.Score, score: ctx.Trainers.Sdca(r.label, r.features, maxIterations: 2, advancedSettings: s => s.NumThreads = 1)));

            var pipe = reader.Append(est);

            var model = pipe.Fit(dataSource);
            var data = model.Read(dataSource);

            // Now, let's see if that column is still there, and still text!
            var schema = data.AsDynamic.Schema;
            Assert.True(schema.TryGetColumnIndex("Score", out int scoreCol), "Score column not present!");
            Assert.Equal(TextType.Instance, schema[scoreCol].Type);

            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");
        }

        [Fact]
        public void SdcaBinaryClassification()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryModelParameters pred = null;
            ParameterMixingCalibratedPredictor cali = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(r.label, r.features,
                    maxIterations: 2,
                    onFit: (p, c) => { pred = p; cali = c; },
                    advancedSettings: s => s.NumThreads = 1)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            Assert.Null(cali);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            Assert.NotNull(cali);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights.Count);

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
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");
        }

        [Fact]
        public void SdcaBinaryClassificationNoCalibration()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryModelParameters pred = null;

            var loss = new HingeLoss(1);

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.Sdca(r.label, r.features,
                maxIterations: 2,
                loss: loss, onFit: p => pred = p,
                advancedSettings: s => s.NumThreads = 1)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");
        }

        [Fact]
        public void AveragePerceptronNoCalibration()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryModelParameters pred = null;

            var loss = new HingeLoss(1);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.AveragedPerceptron(r.label, r.features, lossFunction: loss,
                numIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights.Count);

            var data = model.Read(dataSource);

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds);
            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.Auc, 0, 1);
            Assert.InRange(metrics.Auprc, 0, 1);
        }

        [Fact]
        public void AveragePerceptronCalibration()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            LinearBinaryModelParameters pred = null;

            var loss = new HingeLoss(1);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.AveragedPerceptron(r.label, r.features, lossFunction: loss,
                numIterations: 2, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);
            // 9 input features, so we ought to have 9 weights.
            Assert.Equal(9, pred.Weights.Count);

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features1: c.LoadFloat(1, 4), features2: c.LoadFloat(5, 9)));

            FieldAwareFactorizationMachineModelParameters pred = null;

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MulticlassLogisticRegressionModelParameters pred = null;

            var loss = new HingeLoss(1);

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
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void CrossValidate()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoaderStatic.CreateReader(env,
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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            FastTreeRegressionModelParameters pred = null;

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LightGbmRegressionModelParameters pred = null;

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            PoissonRegressionModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.PoissonRegression(r.label, r.features,
                    l1Weight: 2,
                    enoforceNoNegativity: true,
                    onFit: (p) => { pred = p; },
                    advancedSettings: s => s.NumThreads = 1)));

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.LogisticRegressionBinaryClassifier(r.label, r.features,
                    l1Weight: 10,
                    onFit: (p) => { pred = p; },
                    advancedSettings: s => s.NumThreads = 1)));

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
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MulticlassLogisticRegressionModelParameters pred = null;

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.MultiClassLogisticRegression(
                    r.label,
                    r.features, onFit: p => pred = p,
                    advancedSettings: s => s.NumThreads = 1)));

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
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void OnlineGradientDescent()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RegressionContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            LinearRegressionModelParameters pred = null;

            var loss = new SquaredLoss();

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: ctx.Trainers.OnlineGradientDescent(r.label, r.features,
                lossFunction: loss,
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
            var env = new MLContext(seed: 0, conc: 1);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            KMeansModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .AppendCacheCheckpoint()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, r.features, preds: env.Clustering.Trainers.KMeans(r.features, clustersCount: 3, onFit: p => pred = p, advancedSettings: s => s.NumThreads = 1)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            VBuffer<float>[] centroids = default;
            int k;
            pred.GetClusterCentroids(ref centroids, out k);

            Assert.True(k == 3);

            var data = model.Read(dataSource);

            var metrics = env.Clustering.Evaluate(data, r => r.preds.score, r => r.label, r => r.features);
            Assert.NotNull(metrics);

            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.InRange(metrics.Nmi, 0.73, 0.77);
            Assert.InRange(metrics.Dbi, 0.662, 0.667);

            metrics = env.Clustering.Evaluate(data, r => r.preds.score, label: r => r.label);
            Assert.NotNull(metrics);

            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.True(metrics.Dbi == 0.0);

            metrics = env.Clustering.Evaluate(data, r => r.preds.score, features: r => r.features);
            Assert.True(double.IsNaN(metrics.Nmi));

            metrics = env.Clustering.Evaluate(data, r => r.preds.score);
            Assert.NotNull(metrics);
            Assert.InRange(metrics.AvgMinScore, 0.5262, 0.5264);
            Assert.True(double.IsNaN(metrics.Nmi));
            Assert.True(metrics.Dbi == 0.0);

        }

        [Fact]
        public void FastTreeRanking()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.adultRanking.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RankingContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(0), features: c.LoadFloat(9, 14), groupId: c.LoadText(1)),
                separator: '\t', hasHeader: true);

            FastTreeRankingModelParameters pred = null;

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRanking()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.adultRanking.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new RankingContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(0), features: c.LoadFloat(9, 14), groupId: c.LoadText(1)),
                separator: '\t', hasHeader: true);

            LightGbmRankingModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.features, groupId: r.groupId.ToKey()))
                .Append(r => (r.label, r.groupId, score: ctx.Trainers.LightGbm(r.label, r.features, r.groupId, onFit: (p) => { pred = p; })));

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void MultiClassLightGBM()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            OvaModelParameters pred = null;

            // With a custom loss function we no longer get calibrated predictions.
            var est = reader.MakeNewEstimator()
                .Append(r => (label: r.label.ToKey(), r.features))
                .Append(r => (r.label, preds: ctx.Trainers.LightGbm(
                    r.label,
                    r.features, onFit: p => pred = p)));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var data = model.Read(dataSource);

            // Just output some data on the schema for fun.
            var schema = data.AsDynamic.Schema;
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void MultiClassNaiveBayesTrainer()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new MulticlassClassificationContext(env);
            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadText(0), features: c.LoadFloat(1, 4)));

            MultiClassNaiveBayesModelParameters pred = null;

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
            for (int c = 0; c < schema.Count; ++c)
                Console.WriteLine($"{schema[c].Name}, {schema[c].Type}");

            var metrics = ctx.Evaluate(data, r => r.label, r => r.preds, 2);
            Assert.True(metrics.LogLoss > 0);
            Assert.True(metrics.TopKAccuracy > 0);
        }

        [Fact]
        public void HogwildSGDBinaryClassification()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);
            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            IPredictorWithFeatureWeights<float> pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, preds: ctx.Trainers.StochasticGradientDescentClassificationTrainer(r.label, r.features,
                    l2Weight: 0,
                    onFit: (p) => { pred = p; },
                    advancedSettings: s => s.NumThreads = 1)));

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

        [ConditionalFact(typeof(BaseTestBaseline), nameof(BaseTestBaseline.LessThanNetCore30OrNotNetCoreAnd64BitProcess))] // netcore3.0 and x86 output differs from Baseline. This test is being fixed as part of issue #1441.
        public void MatrixFactorization()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Specify where to find data file
            var dataPath = GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            // Read data file. The file contains 3 columns, label (float value), matrixColumnIndex (unsigned integer key), and matrixRowIndex (unsigned integer key).
            // More specifically, LoadKey(1, 0, 19) means that the matrixColumnIndex column is read from the 2nd (indexed by 1) column in the data file and as
            // a key type (stored as 32-bit unsigned integer) ranged from 0 to 19 (aka the training matrix has 20 columns).
            var reader = mlContext.Data.CreateTextReader(ctx => (label: ctx.LoadFloat(0), matrixColumnIndex: ctx.LoadKey(1, 0, 19), matrixRowIndex: ctx.LoadKey(2, 0, 39)), hasHeader: true);

            // The parameter that will be into the onFit method below. The obtained predictor will be assigned to this variable
            // so that we will be able to touch it.
            MatrixFactorizationPredictor pred = null;

            // Create a statically-typed matrix factorization estimator. The MatrixFactorization's input and output defined in MatrixFactorizationStatic
            // tell what (aks a Scalar<float>) is expected. Notice that only one thread is used for deterministic outcome.
            var matrixFactorizationEstimator = reader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.MatrixFactorization(r.label, r.matrixRowIndex, r.matrixColumnIndex, onFit: p => pred = p,
                advancedSettings: args => { args.NumThreads = 1; })));

            // Create a pipeline from the reader (the 1st step) and the matrix factorization estimator (the 2nd step).
            var pipe = reader.Append(matrixFactorizationEstimator);

            // pred will be assigned by the onFit method once the training process is finished, so pred must be null before training.
            Assert.Null(pred);

            // Train the pipeline on the given data file. Steps in the pipeline are sequentially fitted (by calling their Fit function).
            var model = pipe.Fit(dataSource);

            // pred got assigned so that one can inspect the predictor trained in pipeline.
            Assert.NotNull(pred);

            // Feed the data file into the trained pipeline. The data would be loaded by TextLoader (the 1st step) and then the output of the
            // TextLoader would be fed into MatrixFactorizationEstimator.
            var estimatedData = model.Read(dataSource);

            // After the training process, the metrics for regression problems can be computed.
            var metrics = mlContext.Regression.Evaluate(estimatedData, r => r.label, r => r.score);

            // Naive test. Just make sure the pipeline runs.
            Assert.InRange(metrics.L2, 0, 0.5);
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void MultiClassLightGbmStaticPipelineWithInMemoryData()
        {
            // Create a general context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Create in-memory examples as C# native class.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert native C# class to IDataView, a consumble format to ML.NET functions.
            var dataView = ComponentCreation.CreateDataView(mlContext, examples);

            // IDataView is the data format used in dynamic-typed pipeline. To use static-typed pipeline, we need to convert
            // IDataView to DataView by calling AssertStatic(...). The basic idea is to specify the static type for each column
            // in IDataView in a lambda function.
            var staticDataView = dataView.AssertStatic(mlContext, c => (
                         Features: c.R4.Vector,
                         Label: c.Text.Scalar));

            // Create static pipeline. First, we make an estimator out of static DataView as the starting of a pipeline.
            // Then, we append necessary transforms and a classifier to the starting estimator.
            var pipe = staticDataView.MakeNewEstimator()
                    .Append(mapper: r => (
                        r.Label,
                        // Train multi-class LightGBM. The trained model maps Features to Label and probability of each class.
                        // The call of ToKey() is needed to convert string labels to integer indexes.
                        Predictions: mlContext.MulticlassClassification.Trainers.LightGbm(r.Label.ToKey(), r.Features)
                    ))
                    .Append(r => (
                        // Actual label.
                        r.Label,
                        // Labels are converted to keys when training LightGBM so we convert it here again for calling evaluation function.
                        LabelIndex: r.Label.ToKey(),
                        // Used to compute metrics such as accuracy.
                        r.Predictions,
                        // Assign a new name to predicted class index.
                        PredictedLabelIndex: r.Predictions.predictedLabel,
                        // Assign a new name to class probabilities.
                        Scores: r.Predictions.score
                    ));

            // Split the static-typed data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var (trainingData, testingData) = mlContext.MulticlassClassification.TrainTestSplit(staticDataView, testFraction: 0.5);

            // Train the model.
            var model = pipe.Fit(trainingData);

            // Do prediction on the test set.
            var prediction = model.Transform(testingData);

            // Evaluate the trained model is the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(prediction, r => r.LabelIndex, r => r.Predictions);

            // Check if metrics are resonable.
            Assert.Equal(0.863482146891263, metrics.AccuracyMacro, 6);
            Assert.Equal(0.86309523809523814, metrics.AccuracyMicro, 6);

            // Convert prediction in ML.NET format to native C# class.
            var nativePredictions = new List<DatasetUtils.MulticlassClassificationExample>(prediction.AsDynamic.AsEnumerable<DatasetUtils.MulticlassClassificationExample>(mlContext, false));

            // Get schema object of the prediction. It contains metadata such as the mapping from predicted label index
            // (e.g., 1) to its actual label (e.g., "AA").
            var schema = prediction.AsDynamic.Schema;

            // Retrieve the mapping from labels to label indexes.
            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            schema[nameof(DatasetUtils.MulticlassClassificationExample.PredictedLabelIndex)].Metadata.GetValue("KeyValues", ref labelBuffer);
            var nativeLabels = labelBuffer.DenseValues().ToList(); // nativeLabels[nativePrediction.PredictedLabelIndex-1] is the original label indexed by nativePrediction.PredictedLabelIndex.

            // Show prediction result for the 3rd example.
            var nativePrediction = nativePredictions[2];
            var expectedProbabilities = new float[] { 0.922597349f, 0.07508608f, 0.00221699756f, 9.95488E-05f };
            // Scores and nativeLabels are two parallel attributes; that is, Scores[i] is the probability of being nativeLabels[i].
            for (int i = 0; i < labelBuffer.Length; ++i)
                Assert.Equal(expectedProbabilities[i], nativePrediction.Scores[i], 6);

            // The predicted label below should be  with probability 0.922597349.
            Console.WriteLine("Our predicted label to this example is {0} with probability {1}",
                nativeLabels[(int)nativePrediction.PredictedLabelIndex-1],
                nativePrediction.Scores[(int)nativePrediction.PredictedLabelIndex-1]);
        }
    }
}