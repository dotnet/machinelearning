// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    
    public class SweeperTests
    {
        [Fact]
        public void SmacQuickRunTest()
        {
            var numInitialPopulation = 10;

            var floatValueGenerator = new FloatValueGenerator(new FloatParamArguments() { Name = "float", Min = 1, Max = 1000 });
            var floatLogValueGenerator = new FloatValueGenerator(new FloatParamArguments() { Name = "floatLog", Min = 1, Max = 1000, LogBase = true });
            var longValueGenerator = new LongValueGenerator(new LongParamArguments() { Name = "long", Min = 1, Max = 1000 });
            var longLogValueGenerator = new LongValueGenerator(new LongParamArguments() { Name = "longLog", Min = 1, Max = 1000, LogBase = true });
            var discreteValueGeneator = new DiscreteValueGenerator(new DiscreteParamArguments() { Name = "discrete", Values = new[] { "200", "400", "600", "800" } });

            var sweeper = new SmacSweeper(new MLContext(), new SmacSweeper.Arguments()
            {
                SweptParameters = new IValueGenerator[] {
                    floatValueGenerator,
                    floatLogValueGenerator,
                    longValueGenerator,
                    longLogValueGenerator,
                    discreteValueGeneator
                },
                NumberInitialPopulation = numInitialPopulation
            });

            // sanity check grid
            Assert.NotNull(floatValueGenerator[0].ValueText);
            Assert.NotNull(floatLogValueGenerator[0].ValueText);
            Assert.NotNull(longValueGenerator[0].ValueText);
            Assert.NotNull(longLogValueGenerator[0].ValueText);
            Assert.NotNull(discreteValueGeneator[0].ValueText);

            List<RunResult> results = new List<RunResult>();

            RunResult bestResult = null;
            for (var i = 0; i < numInitialPopulation + 1; i++)
            {
                ParameterSet[] pars = sweeper.ProposeSweeps(1, results);

                foreach (ParameterSet p in pars)
                {
                    float x1 = float.Parse(p["float"].ValueText);
                    float x2 = float.Parse(p["floatLog"].ValueText);
                    long x3 = long.Parse(p["long"].ValueText);
                    long x4 = long.Parse(p["longLog"].ValueText);
                    int x5 = int.Parse(p["discrete"].ValueText);

                    double metric = x1 + x2 + x3 + x4 + x5;

                    RunResult result = new RunResult(p, metric, true);
                    if (bestResult == null || bestResult.MetricValue < metric)
                    {
                        bestResult = result;
                    }
                    results.Add(result);

                    Console.WriteLine($"{metric}\t{x1},{x2}");
                }

            }

            Console.WriteLine($"Best: {bestResult.MetricValue}");

            Assert.NotNull(bestResult);
            Assert.True(bestResult.MetricValue > 0);
        }

        [Fact(Skip = "This test is too slow to run as part of automation.")]
        public void Smac4ParamsConvergenceTest()
        {
            var sweeper = new SmacSweeper(new MLContext(), new SmacSweeper.Arguments()
            {
                SweptParameters = new INumericValueGenerator[] {
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x1", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x2", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x3", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x4", Min = 1, Max = 1000}),
                },
            });
            
            List<RunResult> results = new List<RunResult>();

            RunResult bestResult = null;
            for (var i = 0; i < 300; i++)
            {
                ParameterSet[] pars = sweeper.ProposeSweeps(1, results);

                // if run converged, break
                if (pars == null)
                {
                    break;
                }

                foreach (ParameterSet p in pars)
                {
                    float x1 = (p["x1"] as FloatParameterValue).Value;
                    float x2 = (p["x2"] as FloatParameterValue).Value;
                    float x3 = (p["x3"] as FloatParameterValue).Value;
                    float x4 = (p["x4"] as FloatParameterValue).Value;

                    double metric = -200 * (Math.Abs(100 - x1) +
                        Math.Abs(300 - x2) +
                        Math.Abs(500 - x3) +
                        Math.Abs(700 - x4));

                    RunResult result = new RunResult(p, metric, true);
                    if (bestResult == null || bestResult.MetricValue < metric)
                    {
                        bestResult = result;
                    }
                    results.Add(result);

                    Console.WriteLine($"{metric}\t{x1},{x2},{x3},{x4}");
                }

            }

            Console.WriteLine($"Best: {bestResult.MetricValue}");
        }

        [Fact(Skip = "This test is too slow to run as part of automation.")]
        public void Smac2ParamsConvergenceTest()
        {
            var sweeper = new SmacSweeper(new MLContext(), new SmacSweeper.Arguments()
            {
                SweptParameters = new INumericValueGenerator[] {
                    new FloatValueGenerator(new FloatParamArguments() { Name = "foo", Min = 1, Max = 5}),
                    new LongValueGenerator(new LongParamArguments() { Name = "bar", Min = 1, Max = 1000, LogBase = true })
                },
            });

            Random rand = new Random(0);
            List<RunResult> results = new List<RunResult>();

            int count = 0;
            while (true)
            {
                ParameterSet[] pars = sweeper.ProposeSweeps(1, results);
                if(pars == null)
                {
                    break;
                }
                foreach (ParameterSet p in pars)
                {
                    float foo = 0;
                    long bar = 0;

                    foo = (p["foo"] as FloatParameterValue).Value;
                    bar = (p["bar"] as LongParameterValue).Value;

                    double metric = ((5 - Math.Abs(4 - foo)) * 200) + (1001 - Math.Abs(33 - bar)) + rand.Next(1, 20);
                    results.Add(new RunResult(p, metric, true));
                    count++;
                    Console.WriteLine("{0}--{1}--{2}--{3}", count, foo, bar, metric);
                }
            }
        }
    }
}
