// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class SweeperTests : BaseTestClass
    {
        public SweeperTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void SmacQuickRunTest()
        {
            var numInitialPopulation = 10;

            var floatValueGenerator = new FloatValueGenerator(new FloatParamArguments() { Name = "float", Min = 1, Max = 1000 });
            var floatLogValueGenerator = new FloatValueGenerator(new FloatParamArguments() { Name = "floatLog", Min = 1, Max = 1000, LogBase = true });
            var longValueGenerator = new LongValueGenerator(new LongParamArguments() { Name = "long", Min = 1, Max = 1000 });
            var longLogValueGenerator = new LongValueGenerator(new LongParamArguments() { Name = "longLog", Min = 1, Max = 1000, LogBase = true });
            var discreteValueGenerator = new DiscreteValueGenerator(new DiscreteParamArguments() { Name = "discrete", Values = new[] { "200", "400", "600", "800" } });

            var sweeper = new SmacSweeper(new MLContext(1), new SmacSweeper.Arguments()
            {
                SweptParameters = new IValueGenerator[] {
                    floatValueGenerator,
                    floatLogValueGenerator,
                    longValueGenerator,
                    longLogValueGenerator,
                    discreteValueGenerator
                },
                NumberInitialPopulation = numInitialPopulation
            });

            // sanity check grid
            Assert.NotNull(floatValueGenerator[0].ValueText);
            Assert.NotNull(floatLogValueGenerator[0].ValueText);
            Assert.NotNull(longValueGenerator[0].ValueText);
            Assert.NotNull(longLogValueGenerator[0].ValueText);
            Assert.NotNull(discreteValueGenerator[0].ValueText);

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
            var sweeper = new SmacSweeper(new MLContext(1), new SmacSweeper.Arguments()
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
            var sweeper = new SmacSweeper(new MLContext(1), new SmacSweeper.Arguments()
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
                if (pars == null)
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

        [Fact]
        public void TestLongParameterValue()
        {
            LongParameterValue value1 = new LongParameterValue(nameof(value1), 1);
            LongParameterValue value2 = new LongParameterValue(nameof(value2), 2);
            LongParameterValue duplicateValue1 = new LongParameterValue(nameof(value1), 1);

            Assert.False(value1.Equals(value2));
            Assert.True(value1.Equals(value1));
            Assert.True(value1.Equals(duplicateValue1));
            Assert.False(value1.Equals((object)value2));
            Assert.True(value1.Equals((object)value1));
            Assert.True(value1.Equals((object)duplicateValue1));

            Assert.False(value1.Equals(new FloatParameterValue(nameof(value1), 1.0f)));
            Assert.False(value1.Equals((IParameterValue)null));
            Assert.False(value1.Equals((object)null));
            Assert.False(value1.Equals(new object()));

            Assert.Equal(value1.GetHashCode(), value1.GetHashCode());
            Assert.Equal(value1.GetHashCode(), duplicateValue1.GetHashCode());
        }

        [Fact]
        public void TestFloatParameterValue()
        {
            FloatParameterValue value1 = new FloatParameterValue(nameof(value1), 1.0f);
            FloatParameterValue value2 = new FloatParameterValue(nameof(value2), 2.0f);
            FloatParameterValue duplicateValue1 = new FloatParameterValue(nameof(value1), 1.0f);

            Assert.False(value1.Equals(value2));
            Assert.True(value1.Equals(value1));
            Assert.True(value1.Equals(duplicateValue1));
            Assert.False(value1.Equals((object)value2));
            Assert.True(value1.Equals((object)value1));
            Assert.True(value1.Equals((object)duplicateValue1));

            Assert.False(value1.Equals(new LongParameterValue(nameof(value1), 1)));
            Assert.False(value1.Equals((IParameterValue)null));
            Assert.False(value1.Equals((object)null));
            Assert.False(value1.Equals(new object()));

            Assert.Equal(value1.GetHashCode(), value1.GetHashCode());
            Assert.Equal(value1.GetHashCode(), duplicateValue1.GetHashCode());
        }

        [Fact]
        public void TestStringParameterValue()
        {
            StringParameterValue value1 = new StringParameterValue(nameof(value1), "1");
            StringParameterValue value2 = new StringParameterValue(nameof(value2), "2");
            StringParameterValue duplicateValue1 = new StringParameterValue(nameof(value1), "1");

            Assert.False(value1.Equals(value2));
            Assert.True(value1.Equals(value1));
            Assert.True(value1.Equals(duplicateValue1));
            Assert.False(value1.Equals((object)value2));
            Assert.True(value1.Equals((object)value1));
            Assert.True(value1.Equals((object)duplicateValue1));

            Assert.False(value1.Equals(new LongParameterValue(nameof(value1), 1)));
            Assert.False(value1.Equals((IParameterValue)null));
            Assert.False(value1.Equals((object)null));
            Assert.False(value1.Equals(new object()));

            Assert.Equal(value1.GetHashCode(), value1.GetHashCode());
            Assert.Equal(value1.GetHashCode(), duplicateValue1.GetHashCode());
        }

        [Fact]
        public void TestParameterSetEquality()
        {
            LongParameterValue value1 = new LongParameterValue(nameof(value1), 1);
            LongParameterValue value2 = new LongParameterValue(nameof(value2), 2);
            StringParameterValue stringValue1 = new StringParameterValue(nameof(value1), "1");

            var parameterSet = new ParameterSet(new[] { value1 });
            Assert.False(parameterSet.Equals(null));

            // Verify Equals for sets with different hash codes
            var parameterSetNewHash = new ParameterSet(new IParameterValue[] { value1 }.ToDictionary(x => x.Name), hash: parameterSet.GetHashCode() + 1);
            Assert.NotEqual(parameterSet.GetHashCode(), parameterSetNewHash.GetHashCode());
            Assert.False(parameterSet.Equals(parameterSetNewHash));

            // Verify Equals for sets with the same hash code, but different number of values
            var parameterSetMoreValues = new ParameterSet(new IParameterValue[] { value1, value2 }.ToDictionary(x => x.Name), hash: parameterSet.GetHashCode());
            Assert.Equal(parameterSet.GetHashCode(), parameterSetMoreValues.GetHashCode());
            Assert.False(parameterSet.Equals(parameterSetMoreValues));

            // Verify Equals for sets with the same hash and item counts, but one of the items has a different name
            var parameterSetDifferentName = new ParameterSet(new IParameterValue[] { value2 }.ToDictionary(x => x.Name), hash: parameterSet.GetHashCode());
            Assert.Equal(parameterSet.GetHashCode(), parameterSetDifferentName.GetHashCode());
            Assert.False(parameterSet.Equals(parameterSetDifferentName));

            // Verify Equals for sets with the same hash and item names, but one of the items has a different value
            var parameterSetDifferentValue = new ParameterSet(new IParameterValue[] { stringValue1 }.ToDictionary(x => x.Name), hash: parameterSet.GetHashCode());
            Assert.Equal(parameterSet.GetHashCode(), parameterSetDifferentValue.GetHashCode());
            Assert.False(parameterSet.Equals(parameterSetDifferentValue));

            // Verify Equals for sets with the same hash and items
            var parameterSetSameHash = new ParameterSet(new IParameterValue[] { value1 }.ToDictionary(x => x.Name), hash: parameterSet.GetHashCode());
            Assert.Equal(parameterSet.GetHashCode(), parameterSetSameHash.GetHashCode());
            Assert.True(parameterSet.Equals(parameterSetSameHash));
        }
    }
}
