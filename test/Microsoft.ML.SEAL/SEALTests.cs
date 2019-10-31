// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.Research.SEAL;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.SEAL
{
    public sealed class SEALTests : TestDataPipeBase
    {
        public SEALTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        private class TestClass
        {
            public double[] plaintext;
            public Ciphertext[] ciphertext;
        }

        [Fact]
        public void EncryptFeaturesTest()
        {
            var encParams = new EncryptionParameters(SchemeType.CKKS)
            {
                PolyModulusDegree = 8192,
                CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree: 8192)
            };
            var context = new SEALContext(encParams);
            var keygen = new KeyGenerator(context);

            using (var fs = File.Open("secret.key", FileMode.OpenOrCreate))
            {
                keygen.SecretKey.Save(fs);
            }

            using (var fs2 = File.Open("public.key", FileMode.OpenOrCreate))
            {
                keygen.PublicKey.Save(fs2);
            }

            var scale = 1152921504606846976;
            var polyModDegree = 8192;
            var bitSizes = new[] {
                36028797005856769,
                36028797001138177,
                18014398492704769,
                18014398491918337
            };

            var encryptPipeline = ML.Transforms.EncryptFeatures(true,
                scale,
                polyModDegree,
                "public.key",
                bitSizes,
                "ciphertext",
                "plaintext");

            var decryptPipeline = encryptPipeline.Append(ML.Transforms.EncryptFeatures(false,
                scale,
                polyModDegree,
                "private.key",
                bitSizes,
                "plaintext",
                "ciphertext"));

            var data = new[] { new TestClass() { plaintext = new[] { 1.1, 2.2, 3.3, 4.4 }}};
            var dataView = ML.Data.LoadFromEnumerable(data);
            var model = encryptPipeline.Fit(dataView);
            var engine = model.CreatePredictionEngine<TestClass, TestClass>(ML);
            var prediction = engine.Predict(data[0]);
            Assert.Equal(data[0].plaintext, prediction.plaintext);
        }

        /*
        [Fact]
        public void EncryptedEvaluation()
        {
            // Generate C# objects as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(100);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is very helpful when working with iterative
            // algorithms which needs many data passes. Since SDCA is the case, we cache.
            data = mlContext.Data.Cache(data);

            var context = GlobalContext.CKKSContext;
            var keygen = new KeyGenerator(context);

            using (var fs = File.Open("secret.key", FileMode.OpenOrCreate))
            {
                keygen.SecretKey.Save(fs);
            }

            using (var fs2 = File.Open("public.key", FileMode.OpenOrCreate))
            {
                keygen.PublicKey.Save(fs2);
            }

            var scale = 1152921504606846976;
            var polyModDegree = 8192;
            var bitSizes = new[] {
                36028797005856769,
                36028797001138177,
                18014398492704769,
                18014398491918337
            };

            var encryptPipeline = Microsoft.ML.SEAL.SealCatalog.EncryptFeatures(true, scale, polyModDegree, "public.key", bitSizes, "ciphertext", "plaintext");

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var esdcaPipeline = encryptPipeline.Append(mlContext.BinaryClassification.Trainers.EncryptedSdcaLogisticRegression(polyModDegree: polyModDegree,
                bitSizes: bitSizes, scale: scale, labelColumnName: "Label", featureColumnName: "Features", l2Regularization: 0.001f));

            var decryptPipeline = esdcaPipeline.Append(Microsoft.ML.SEAL.SealCatalog.EncryptFeatures(false, scale, polyModDegree, "private.key", bitSizes,
                "plaintext", "ciphertext"));

            // Step 3: Train the pipeline created.
            var model = decryptPipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);
            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Check a few metrics to make sure the trained model is ok.
            Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
            Assert.InRange(metrics.LogLoss, 0, 0.5);

            var rawPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.CalibratedBinaryClassifierOutput>(prediction, false);

            // Step 5: Inspect the prediction of the first example.
            var first = rawPrediction.First();
            // This is a positive example.
            Assert.True(first.Label);
            // Positive example should have non-negative score.
            Assert.True(first.Score > 0);
            // Positive example should have high probability of belonging the positive class.
            Assert.InRange(first.Probability, 0.8, 1);
        }
        */
    }
}