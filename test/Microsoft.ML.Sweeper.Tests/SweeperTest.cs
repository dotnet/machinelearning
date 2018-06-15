using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Sweeper;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Sweeper.Tests
{
    public class SweeperTest
    {
        [Fact]
        public void SweeperReturnsDistinctValues()
        {
            var args = new DiscreteParamArguments();
            args.Name = "Amazing";
            args.Values = new string[] { "one" };
            var valueGenerator = new DiscreteValueGenerator(args);
            using (var writer = new StreamWriter(new MemoryStream()))
            using (var env = new TlcEnvironment(42, outWriter: writer, errWriter: writer))
            {
                var sweeper = new UniformRandomSweeper(env, new SweeperBase.ArgumentsBase(), new[] { valueGenerator });
                var results = sweeper.ProposeSweeps(2);
                Assert.NotNull(results);
                int length = results.Length;
                Assert.Equal(1, length);
            }
        }
    }
}
