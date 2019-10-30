// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
using Microsoft.Research.SEALNet;
using Xunit;

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

            var encryptPipeline = Microsoft.ML.SEAL.SealCatalog.EncryptFeatures(true,
                1152921504606846976,
                8192,
                "public.key",
                new[] {
                    36028797005856769,
                    36028797001138177,
                    18014398492704769,
                    18014398491918337
                },
                "ciphertext",
                "plaintext");

            var decryptPipeline = encryptPipeline.Append(Microsoft.ML.SEAL.SealCatalog.EncryptFeatures(false,
                1152921504606846976,
                8192,
                "private.key",
                new int[] {
                    36028797005856769,
                    36028797001138177,
                    18014398492704769,
                    18014398491918337
                },
                "plaintext",
                "ciphertext"));

            var data = new[] { 1.1, 2.2, 3.3, 4.4 };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var model = encryptPipeline.Fit(dataView);
            var engine = model.CreatePredictionEngine<TestClass, TestClass>(ML);
        }
    }
}