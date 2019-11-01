// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.RunTests;
using Microsoft.ML.SEAL;
using Microsoft.Research.SEAL;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.SEAL
{
    public sealed class SEALTests : TestDataPipeBase, IDisposable
    {
        private readonly TextWriter _originalOut;
        private readonly TextWriter _textWriter;

        public SEALTests(ITestOutputHelper helper)
            : base(helper)
        {
            _originalOut = System.Console.Out;
            _textWriter = new StringWriter();
            System.Console.SetOut(_textWriter);
        }

        public void Dispose()
        {
            Output.WriteLine(_textWriter.ToString());
            System.Console.SetOut(_originalOut);
        }

        private class TestClass
        {
            public double[] plaintext;
        }

        [Fact]
        public void EncryptFeaturesTest()
        {
            var coeffModuli = new[] {
                new SmallModulus(36028797005856769),
                new SmallModulus(36028797001138177),
                new SmallModulus(18014398492704769),
                new SmallModulus(18014398491918337)
            };
            var scale = 1152921504606846976;
            var polyModDegree = 8192UL;
            var encParams = new EncryptionParameters(SchemeType.CKKS)
            {
                PolyModulusDegree = polyModDegree,
                CoeffModulus = coeffModuli
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

            var encryptPipeline = ML.Transforms.EncryptFeatures(true,
                scale,
                polyModDegree,
                "public.key",
                coeffModuli,
                "ciphertext",
                "plaintext");

            var decryptPipeline = encryptPipeline.Append(ML.Transforms.EncryptFeatures(false,
                scale,
                polyModDegree,
                "secret.key",
                coeffModuli,
                "plaintext",
                "ciphertext"));

            var data = new[] { new TestClass() { plaintext = new[] { 1.1, 2.2, 3.3, 4.4 }}};
            var dataView = ML.Data.LoadFromEnumerable(data);
            var model = encryptPipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var rawPredictionEnumerable = ML.Data.CreateEnumerable<TestClass>(prediction, false);
            foreach (var rawPrediction in rawPredictionEnumerable)
            {
                Assert.Equal(data[0].plaintext, rawPrediction.plaintext);
            }
        }

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

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var pipeline = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features", l2Regularization: 0.001f);

            // Step 3: Train the pipeline created.
            var unencryptedModel = pipeline.Fit(data);

            var coeffModuli = new[] {
                new SmallModulus(36028797005856769),
                new SmallModulus(36028797001138177),
                new SmallModulus(18014398492704769),
                new SmallModulus(18014398491918337)
            };
            var scale = 1152921504606846976;
            var polyModDegree = 8192UL;
            var encParams = new EncryptionParameters(SchemeType.CKKS)
            {
                PolyModulusDegree = polyModDegree,
                CoeffModulus = coeffModuli
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

            var encryptPipeline = ML.Transforms.EncryptFeatures(true,
                scale,
                polyModDegree,
                "public.key",
                coeffModuli,
                "Ciphertext",
                "Features");

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var esdcaPipeline = encryptPipeline.Append(mlContext.BinaryClassification.Trainers.EncryptedSdcaLogisticRegression(polyModulusDegree: polyModDegree,
                coeffModuli: coeffModuli, scale: scale, labelColumnName: "Label", featureColumnName: "Ciphertext", l2Regularization: 0.001f));

            var decryptPipeline = esdcaPipeline.Append(encryptPipeline.Append(ML.Transforms.EncryptFeatures(false,
                scale,
                polyModDegree,
                "secret.key",
                coeffModuli,
                "Plaintext",
                "Label")));

            // Step 3: Train the pipeline created.
            var encryptedModel = decryptPipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var unencryptedPrediction = unencryptedModel.Transform(data);
            var encryptedPrediction = encryptedModel.Transform(data);
            var rawUnencryptedPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.CalibratedBinaryClassifierOutput>(unencryptedPrediction, false);
            var rawEncryptedPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.CalibratedBinaryClassifierOutput>(encryptedPrediction, false);
            var bothPredictions = rawUnencryptedPrediction.Zip(rawEncryptedPrediction, (u, e) => new { unencrypted = u, encrypted = e });

            foreach (var rawPrediction in bothPredictions)
            {
                Assert.Equal(rawPrediction.unencrypted, rawPrediction.encrypted);
            }
        }
    }
}