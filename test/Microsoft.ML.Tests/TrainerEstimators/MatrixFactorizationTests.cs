using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Recommend;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [Fact]
        public void MatrixFactorization_Estimator()
        {
            string labelColumnName = "Label";
            string xColumnName = "X";
            string yColumnName = "Y";
            var data = new TextLoader(Env, GetMfTrivialLoaderArgs(labelColumnName, xColumnName, yColumnName))
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            var est = new MatrixFactorizationTrainer(Env, labelColumnName, xColumnName, yColumnName, 
                advancedSettings:s=>
                {
                    s.NumIterations = 3;
                    s.NumThreads = 1;
                    s.K = 4;
                });

            TestEstimatorCore(est, data);

            Done();
        }

        private TextLoader.Arguments GetMfTrivialLoaderArgs(string labelColumnName, string xColumnName, string yColumnName)
        {
            return new TextLoader.Arguments()
            {
                Separator = "\t",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.R4, new [] { new TextLoader.Range(0) }),
                    new TextLoader.Column(xColumnName, DataKind.U4, new [] { new TextLoader.Range(1) }, new KeyRange(0, 19)),
                    new TextLoader.Column(yColumnName, DataKind.U4, new [] { new TextLoader.Range(2) }, new KeyRange(0, 39)),
                }
            };
        }
    }
}
