using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class TrainerExtensionsTests
    {
        [TestMethod]
        public void TrainerExtensionInstanceTests()
        {
            var context = new MLContext();
            var trainerNames = Enum.GetValues(typeof(TrainerName)).Cast<TrainerName>();
            foreach(var trainerName in trainerNames)
            {
                var extension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                var instance = extension.CreateInstance(context, null);
                Assert.IsNotNull(instance);
                var sweepParams = extension.GetHyperparamSweepRanges();
                Assert.IsNotNull(sweepParams);
            }
        }

        [TestMethod]
        public void GetTrainersByMaxIterations()
        {
            var tasks = new TaskKind[] { TaskKind.BinaryClassification,
                TaskKind.MulticlassClassification, TaskKind.Regression  };

            foreach(var task in tasks)
            {
                var trainerSet10 = TrainerExtensionCatalog.GetTrainers(task, 10);
                var trainerSet50 = TrainerExtensionCatalog.GetTrainers(task, 50);
                var trainerSet100 = TrainerExtensionCatalog.GetTrainers(task, 100);

                Assert.IsNotNull(trainerSet10);
                Assert.IsNotNull(trainerSet50);
                Assert.IsNotNull(trainerSet100);

                Assert.IsTrue(trainerSet10.Count() < trainerSet50.Count());
                Assert.IsTrue(trainerSet50.Count() < trainerSet100.Count());
            }
        }
    }
}
