using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Sweeper;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Sweeper.Tests
{
    public class SweeperUniqueValuesTest
    {
        [Fact]
        public void SweeperReturnsDistinctValuesForUniformRandomSweeper()
        {
            DiscreteValueGenerator valueGenerator = SetSingleParameter();
            using (var writer = new StreamWriter(new MemoryStream()))
            using (var env = new TlcEnvironment(42, outWriter: writer, errWriter: writer))
            {
                var sweeper = new UniformRandomSweeper(env, new SweeperBase.ArgumentsBase(), new[] { valueGenerator });
                var results = sweeper.ProposeSweeps(5000);
                Assert.NotNull(results);
                int length = results.Length;
                Assert.Equal(2, length);
            }
        }
        [Fact]
        public void SweeperReturnsDistinctValuesForRandomGridSweeper()
        {
            DiscreteValueGenerator valueGenerator = SetSingleParameter();
            using (var writer = new StreamWriter(new MemoryStream()))
            using (var env = new TlcEnvironment(42, outWriter: writer, errWriter: writer))
            {
                var sweeper = new RandomGridSweeper(env, new RandomGridSweeper.Arguments(), new[] { valueGenerator });
                var results = sweeper.ProposeSweeps(5000);
                Assert.NotNull(results);
                int length = results.Length;
                Assert.Equal(2, length);
            }
        }

        private static DiscreteValueGenerator SetSingleParameter()
        {
            var args = new DiscreteParamArguments();
            args.Name = "TestParam";
            args.Values = new string[] { "one", "two" };
            var valueGenerator = new DiscreteValueGenerator(args);
            return valueGenerator;
        }
    }
}
