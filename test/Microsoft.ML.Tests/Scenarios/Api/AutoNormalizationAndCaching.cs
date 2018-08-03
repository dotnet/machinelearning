using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Auto-normalization and caching: It should be relatively easy for normalization 
        /// and caching to be introduced for training, if the trainer supports or would benefit
        /// from that.
        /// </summary>
        [Fact]
        public void AutoNormalizationAndCaching()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(false), loader);

                // Train.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });

                // Auto-caching.
                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, trans, prefetch: null) : trans;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");
                
                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));
            }

        }
    }
}
