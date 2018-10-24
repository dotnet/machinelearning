// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms.CategoricalTransforms;
using Microsoft.ML.Transforms.Normalizers;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Meta-components: Meta-components (for example, components that themselves instantiate components) should not be booby-trapped.
        /// When specifying what trainer OVA should use, a user will be able to specify any binary classifier.
        /// If they specify a regression or multi-class classifier ideally that should be a compile error.
        /// </summary>
        [Fact]
        public void Metacomponents()
        {
            using (var env = new LocalEnvironment())
            {
                var loader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.irisData.trainFilename)));
                var term = TermTransform.Create(env, loader, "Label");
                var concat = new ConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth").Transform(term);
                var trainer = new Ova(env, new Ova.Arguments
                {
                    PredictorType = ComponentFactoryUtils.CreateFromFunction(
                        e => new AveragedPerceptronTrainer(env, "Label", "Features", lossFunction: new SmoothedHingeLoss.Arguments())
                    )
                });

                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, concat, prefetch: null) : concat;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");

                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new TrainContext(trainRoles));
            }
        }
    }
}