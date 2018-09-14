// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Cross-validation: Have a mechanism to do cross validation, that is, you come up with
        /// a data source (optionally with stratification column), come up with an instantiable transform
        /// and trainer pipeline, and it will handle (1) splitting up the data, (2) training the separate
        /// pipelines on in-fold data, (3) scoring on the out-fold data, (4) returning the set of
        /// evaluations and optionally trained pipes. (People always want metrics out of xfold,
        /// they sometimes want the actual models too.)
        /// </summary>
        [Fact]
        void New_CrossValidation()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {

                var data = new TextLoader(env, MakeSentimentTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));
                // Pipeline.
                var pipeline = new TextTransform(env, "SentimentText", "Features")
                        .Append(new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                        {
                            NumThreads = 1,
                            ConvergenceTolerance = 1f
                        }, "Features", "Label"));

                var cv = new MyCrossValidation.BinaryCrossValidator(env)
                {
                    NumFolds = 2
                };

                var cvResult = cv.CrossValidate(data, pipeline);
            }
        }
    }
}
