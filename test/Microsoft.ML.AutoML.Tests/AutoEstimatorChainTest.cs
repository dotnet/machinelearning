using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TestFramework;
using Xunit;
using Microsoft.ML.AutoML.AutoPipeline;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.AutoPipeline.Test
{
    public class AutoEstimatorChainTest : BaseTestClass
    {
        public AutoEstimatorChainTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void AnyTest()
        {
            var context = new MLContext();
            //context.Transforms.ReplaceMissingValues("Features").Append(context.Recommendation().Trainers.MatrixFactorization, Data.TransformerScope.Training, )
        }
    }
}
