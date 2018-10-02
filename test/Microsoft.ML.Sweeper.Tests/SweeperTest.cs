// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
        public void UniformRandomSweeperReturnsDistinctValuesWhenProposeSweep()
        {
            DiscreteValueGenerator valueGenerator = CreateDiscreteValueGenerator();

            using (var writer = new StreamWriter(new MemoryStream()))
            using (var env = new ConsoleEnvironment(42, outWriter: writer, errWriter: writer))
            {
                var sweeper = new UniformRandomSweeper(env,
                    new SweeperBase.ArgumentsBase(),
                    new[] { valueGenerator });

                var results = sweeper.ProposeSweeps(3);
                Assert.NotNull(results);

                int length = results.Length;
                Assert.Equal(2, length);
            }
        }

        [Fact]
        public void RandomGridSweeperReturnsDistinctValuesWhenProposeSweep()
        {
            DiscreteValueGenerator valueGenerator = CreateDiscreteValueGenerator();

            using (var writer = new StreamWriter(new MemoryStream()))
            using (var env = new ConsoleEnvironment(42, outWriter: writer, errWriter: writer))
            {
                var sweeper = new RandomGridSweeper(env,
                    new RandomGridSweeper.Arguments(),
                    new[] { valueGenerator });

                var results = sweeper.ProposeSweeps(3);
                Assert.NotNull(results);

                int length = results.Length;
                Assert.Equal(2, length);
            }
        }

        private static DiscreteValueGenerator CreateDiscreteValueGenerator()
        {
            var args = new DiscreteParamArguments()
            {
                Name = "TestParam",
                Values = new string[] { "one", "two" }
            };

            return new DiscreteValueGenerator(args);
        }
    }
}
