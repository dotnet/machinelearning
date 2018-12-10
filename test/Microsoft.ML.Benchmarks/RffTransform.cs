// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Transforms.Projections;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    public class RffTransformTrain
    {
        private string _dataPath_Digits;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _dataPath_Digits = Path.GetFullPath(TestDatasets.Digits.trainFilename);

            if (!File.Exists(_dataPath_Digits))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _dataPath_Digits));
        }

        [Benchmark]
        public void CV_Multiclass_Digits_RffTransform_OVAAveragedPerceptron()
        {
            string cmd = @"CV k=5 data={" + _dataPath_Digits + "}" +
                " loader=TextLoader{col=Label:R4:64 col=Features:R4:0-63 sep=,}" +
                " xf=RffTransform{col=FeaturesRFF:Features}" +
                " xf=Concat{col=Features:FeaturesRFF}" +
                " tr=OVA{p=AveragedPerceptron{iter=10}}";

            var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, RandomFourierFeaturizingTransformer, AveragedPerceptronTrainer>();
            Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
        }
    }
}
