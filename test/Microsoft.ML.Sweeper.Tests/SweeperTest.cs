// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Sweeper.Tests
{
    public class SweeperTest : BaseTestClass
    {
        public SweeperTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void UniformRandomSweeperReturnsDistinctValuesWhenProposeSweep()
        {
            DiscreteValueGenerator valueGenerator = CreateDiscreteValueGenerator();

            var env = new MLContext(42);
            var sweeper = new UniformRandomSweeper(env,
                    new SweeperBase.OptionsBase(),
                    new[] { valueGenerator });

            var results = sweeper.ProposeSweeps(3);
            Assert.NotNull(results);

            int length = results.Length;
            Assert.Equal(2, length);
        }

        [Fact]
        public void RandomGridSweeperReturnsDistinctValuesWhenProposeSweep()
        {
            DiscreteValueGenerator valueGenerator = CreateDiscreteValueGenerator();

            var env = new MLContext(42);
            var sweeper = new RandomGridSweeper(env,
                new RandomGridSweeper.Options(),
                new[] { valueGenerator });

            var results = sweeper.ProposeSweeps(3);
            Assert.NotNull(results);

            int length = results.Length;
            Assert.Equal(2, length);
        }

        private static DiscreteValueGenerator CreateDiscreteValueGenerator()
        {
            var args = new DiscreteParamOptions()
            {
                Name = "TestParam",
                Values = new string[] { "one", "two" }
            };

            return new DiscreteValueGenerator(args);
        }
    }
}
