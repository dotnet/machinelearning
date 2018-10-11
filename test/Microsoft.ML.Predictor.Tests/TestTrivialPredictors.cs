// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Internallearn.Test;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.RunTests
{
#if OLD_TESTS // REVIEW: Port these tests.
    /// <summary>
    /// The trivial tests are meant to test for the correctness of our handling of "trivial" predictors,
    /// that is, predictors that have possibly no model complexity (e.g., a linear predictor with no weights,
    /// or a multiclass predictor with one class).
    /// </summary>
    public class TestTrivialPredictors
    {
        private static void CheckOutput(Float expected, Float actual)
        {
            Assert.AreEqual(expected, actual, "difference between original/serialized models output");
        }

        private static void CheckOutput(Float[] expected, Float[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length, "difference between original/serialized models output length");
            for (int i = 0; i < expected.Length; ++i)
                Assert.AreEqual(expected[i], actual[i], "difference between original/serialized models output index {0}", i);
        }

        /// <summary>
        /// Train a model on a single example,
        /// </summary>
        /// <typeparam name="TOutput"></typeparam>
        /// <param name="trainerMaker"></param>
        /// <param name="checker"></param>
        private static void TrivialHelper<TOutput>(Func<ITrainerHost, ITrainer<Instances, IPredictorProducing<TOutput>>> trainerMaker, Action<TOutput, TOutput> checker)
        {
            // The following simple instance should result in a "trivial" predictor for binary classification, regression, and multiclass, I think.
            ListInstances instances = new ListInstances();
            instances.AddInst(new Float[] { (Float)0.0 }, (Float)0);
            instances.CopyMetadata(null);
            ITrainerHost host = new TrainHost(new Random(1), 0);

            var trainer = trainerMaker(host);
            trainer.Train(instances);
            IPredictor<Instance, TOutput> predictor = (IPredictor<Instance, TOutput>)trainer.CreatePredictor();
            IPredictor<Instance, TOutput> loadedPredictor = default(IPredictor<Instance, TOutput>);

            using (Stream stream = new MemoryStream())
            {
                using (RepositoryWriter writer = RepositoryWriter.CreateNew(stream, false))
                {
                    ModelSaveContext.SaveModel(writer, predictor, "foo");
                    writer.Commit();
                }
                stream.Position = 0;
                using (RepositoryReader reader = RepositoryReader.Open(stream, false))
                {
                    ModelLoadContext.LoadModel(out loadedPredictor, reader, "foo");
                }
                Assert.AreNotEqual(default(IPredictor<Instance, TOutput>), loadedPredictor, "did not load expected model");
            }

            TOutput result = predictor.Predict(instances[0]);
            TOutput loadedResult = loadedPredictor.Predict(instances[0]);
            checker(result, loadedResult);
        }

        [Fact, TestCategory("Trivial")]
        public void TrivialAveragedPerceptron()
        {
            TrivialHelper<Float>(host => new AveragedPerceptronTrainer(new AveragedPerceptronTrainer.OldArguments(), host), CheckOutput);
        }

        [Fact, TestCategory("Trivial")]
        public void TrivialSVM()
        {
            TrivialHelper<Float>(host => new LinearSVM(new LinearSVM.OldArguments(), host), CheckOutput);
        }

        [Fact, TestCategory("Trivial")]
        public void TrivialLogisticRegression()
        {
            TrivialHelper<Float>(host => new LogisticRegression(new LogisticRegression.OldArguments(), host), CheckOutput);
        }

        [Fact, TestCategory("Trivial")]
        public void TrivialMulticlassLR()
        {
            TrivialHelper<Float[]>(host => new MulticlassLogisticRegression(new MulticlassLogisticRegression.OldArguments(), host), CheckOutput);
        }
    }
#endif
}
