// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Sweeper;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Sweeper.RunTests
{
    using Float = System.Single;

    public sealed partial class TestSweeper : BaseTestBaseline
    {
        public TestSweeper(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [InlineData("bla", 10, 20, false, 10, 1000, "15")]
        [InlineData("bla", 10, 1000, true, 10, 1000, "99")]
        [InlineData("bla", 10, 1000, true, 10, 3, "99")]
        public void TestLongValueSweep(string name, int min, int max, bool logBase, int stepSize, int numSteps, string valueText)
        {
            var paramSweep = new LongValueGenerator(new LongParamArguments() { Name = name, Min = min, Max = max, LogBase = logBase, StepSize = stepSize, NumSteps = numSteps });
            IParameterValue value = paramSweep.CreateFromNormalized(0.5);
            Assert.Equal(name, value.Name);
            Assert.Equal(valueText, value.ValueText);
        }

        [Fact]
        public void TestLongValueGeneratorRoundTrip()
        {
            var paramSweep = new LongValueGenerator(new LongParamArguments() { Name = "bla", Min = 0, Max = 17 });
            var value = new LongParameterValue("bla", 5);
            float normalizedValue = paramSweep.NormalizeValue(value);
            IParameterValue unNormalizedValue = paramSweep.CreateFromNormalized(normalizedValue);
            Assert.Equal("5", unNormalizedValue.ValueText);

            IParameterValue param = paramSweep.CreateFromNormalized(0.345);
            float unNormalizedParam = paramSweep.NormalizeValue(param);
            Assert.Equal("5", param.ValueText);
            Assert.Equal((Float)5 / 17, unNormalizedParam);
        }

        [Theory]
        [InlineData("bla", 10, 20, false, 10, 1000, "15")]
        [InlineData("bla", 10, 1000, true, 10, 1000, "100")]
        [InlineData("bla", 10, 1000, true, 10, 3, "100")]
        public void TestFloatValueSweep(string name, int min, int max, bool logBase, int stepSize, int numSteps, string valueText)
        {
            var paramSweep = new FloatValueGenerator(new FloatParamArguments() { Name = name, Min = min, Max = max, LogBase = logBase, StepSize = stepSize, NumSteps = numSteps });
            IParameterValue value = paramSweep.CreateFromNormalized(0.5);
            Assert.Equal(name, value.Name);
            Assert.Equal(valueText, value.ValueText);
        }

        [Fact]
        public void TestFloatValueGeneratorRoundTrip()
        {
            var paramSweep = new FloatValueGenerator(new FloatParamArguments() { Name = "bla", Min = 1, Max = 5 });
            var random = new Random(123);
            var normalizedValue = (Float)random.NextDouble();
            var value = (FloatParameterValue)paramSweep.CreateFromNormalized(normalizedValue);
            var originalNormalizedValue = paramSweep.NormalizeValue(value);
            Assert.Equal(normalizedValue, originalNormalizedValue);

            var originalValue = (FloatParameterValue)paramSweep.CreateFromNormalized(normalizedValue);
            Assert.Equal(originalValue.Value, value.Value);
        }

        [Theory]
        [InlineData(0.5, "bar")]
        [InlineData(0.75, "baz")]
        public void TestDiscreteValueSweep(double normalizedValue, string expected)
        {
            var paramSweep = new DiscreteValueGenerator(new DiscreteParamArguments() { Name = "bla", Values = new[] { "foo", "bar", "baz" } });
            var value = paramSweep.CreateFromNormalized(normalizedValue);
            Assert.Equal("bla", value.Name);
            Assert.Equal(expected, value.ValueText);
        }

        [Fact]
        public void TestRandomSweeper()
        {
            var args = new SweeperBase.ArgumentsBase();
            using (var env = new TlcEnvironment(42))
            {
                args.SweptParameters = new[]
                {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=foo", "min=10", "max=20"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=100", "max=200"),
                };

                var sweeper = new UniformRandomSweeper(env, args);
                var initialList = sweeper.ProposeSweeps(5, new List<RunResult>());
                Assert.Equal(5, initialList.Length);
                foreach (var parameterSet in initialList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 10);
                            Assert.True(val <= 20);
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 100);
                            Assert.True(val <= 200);
                        }
                        else
                        {
                            Assert.True(false, "Wrong parameter");
                        }
                    }
                }
            }
        }

        [Fact]
        public void TestSimpleSweeperAsync()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                int sweeps = 100;

                var subcomp = new SubComponent<IAsyncSweeper, SignatureAsyncSweeper>("UniformRandomSweeper",
                    "p=fp{name=foo min=1 max=5}", "p=lp{name=bar min=1 max=1000 logbase+}");
                var sweeper = subcomp.CreateInstance(env);
                var paramSets = new List<ParameterSet>();
                for (int i = 0; i < sweeps; i++)
                {
                    var task = sweeper.Propose();
                    Assert.True(task.IsCompleted);
                    paramSets.Add(task.Result.ParameterSet);
                    var result = new RunResult(task.Result.ParameterSet, random.NextDouble(), true);
                    sweeper.Update(task.Result.Id, result);
                }
                Assert.Equal(sweeps, paramSets.Count);
                CheckAsyncSweeperResult(paramSets);

                // Test consumption without ever calling Update.
                var gridArgs = new RandomGridSweeper.Arguments();
                gridArgs.SweptParameters = new[] {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("fp", "name=foo", "min=1", "max=5"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=1", "max=1000", "logbase+"),
                };
                var gridSweeper = new SimpleAsyncSweeper(env, gridArgs);
                paramSets.Clear();
                for (int i = 0; i < sweeps; i++)
                {
                    var task = gridSweeper.Propose();
                    Assert.True(task.IsCompleted);
                    paramSets.Add(task.Result.ParameterSet);
                }
                Assert.Equal(sweeps, paramSets.Count);
                CheckAsyncSweeperResult(paramSets);
            }
        }
        
        [Fact]
        public void TestDeterministicSweeperAsyncCancellation()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                var args = new DeterministicSweeperAsync.Arguments();
                args.BatchSize = 5;
                args.Relaxation = 1;
                args.Sweeper = new SubComponent<ISweeper, SignatureSweeper>("KDO", "p=fp{name=foo min=1 max=5}",
                    "p=lp{name=bar min=1 max=1000 logbase+}");
                var sweeper = new DeterministicSweeperAsync(env, args);

                int sweeps = 20;
                var tasks = new List<Task<ParameterSetWithId>>();
                int numCompleted = 0;
                for (int i = 0; i < sweeps; i++)
                {
                    var task = sweeper.Propose();
                    if (i < args.BatchSize - args.Relaxation)
                    {
                        Assert.True(task.IsCompleted);
                        sweeper.Update(task.Result.Id, new RunResult(task.Result.ParameterSet, random.NextDouble(), true));
                        numCompleted++;
                    }
                    else
                        tasks.Add(task);
                }
                // Cancel after the first barrier and check if the number of registered actions
                // is indeed 2 * batchSize.
                sweeper.Cancel();
                Task.WaitAll(tasks.ToArray());
                foreach (var task in tasks)
                {
                    if (task.Result != null)
                        numCompleted++;
                }
                Assert.Equal(args.BatchSize + args.BatchSize, numCompleted);
            }
        }

        [Fact]
        public void TestDeterministicSweeperAsync()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                var args = new DeterministicSweeperAsync.Arguments();
                args.BatchSize = 5;
                args.Relaxation = args.BatchSize - 1;
                args.Sweeper = new SubComponent<ISweeper, SignatureSweeper>("SMAC", "p=fp{name=foo min=1 max=5}",
                    "p=lp{name=bar min=1 max=1000 logbase+}");
                var sweeper = new DeterministicSweeperAsync(env, args);

                // Test single-threaded consumption.
                int sweeps = 10;
                var paramSets = new List<ParameterSet>();
                for (int i = 0; i < sweeps; i++)
                {
                    var task = sweeper.Propose();
                    Assert.True(task.IsCompleted);
                    paramSets.Add(task.Result.ParameterSet);
                    var result = new RunResult(task.Result.ParameterSet, random.NextDouble(), true);
                    sweeper.Update(task.Result.Id, result);
                }
                Assert.Equal(sweeps, paramSets.Count);
                CheckAsyncSweeperResult(paramSets);

                // Create two batches and test if the 2nd batch is executed after the synchronization barrier is reached.
                object mlock = new object();
                var tasks = new Task<ParameterSetWithId>[sweeps];
                args.Relaxation = args.Relaxation - 1;
                sweeper = new DeterministicSweeperAsync(env, args);
                paramSets.Clear();
                var results = new List<KeyValuePair<int, IRunResult>>();
                for (int i = 0; i < args.BatchSize; i++)
                {
                    var task = sweeper.Propose();
                    Assert.True(task.IsCompleted);
                    tasks[i] = task;
                    if (task.Result == null)
                        continue;
                    results.Add(new KeyValuePair<int, IRunResult>(task.Result.Id, new RunResult(task.Result.ParameterSet, 0.42, true)));
                }
                // Register comsumers for the 2nd batch. Those consumers will await until at least one run
                // in the previous batch has been posted to the sweeper.
                for (int i = args.BatchSize; i < 2 * args.BatchSize; i++)
                {
                    var task = sweeper.Propose();
                    Assert.False(task.IsCompleted);
                    tasks[i] = task;
                }
                // Call update to unblock the 2nd batch.
                foreach (var run in results)
                    sweeper.Update(run.Key, run.Value);

                Task.WaitAll(tasks);
                tasks.All(t => t.IsCompleted);
            }
        }

        [Fact]
        public void TestDeterministicSweeperAsyncParallel()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                int batchSize = 5;
                int sweeps = 20;
                var paramSets = new List<ParameterSet>();
                var args = new DeterministicSweeperAsync.Arguments();
                args.BatchSize = batchSize;
                args.Relaxation = batchSize - 2;
                args.Sweeper = new SubComponent<ISweeper, SignatureSweeper>("SMAC", "p=fp{name=foo min=1 max=5}",
                    "p=lp{name=bar min=1 max=1000 logbase+}");
                var sweeper = new DeterministicSweeperAsync(env, args);

                var mlock = new object();
                var options = new ParallelOptions();
                options.MaxDegreeOfParallelism = 4;

                // Sleep randomly to simulate doing work.
                int[] sleeps = new int[sweeps];
                for (int i = 0; i < sleeps.Length; i++)
                    sleeps[i] = random.Next(10, 100);
                var r = Parallel.For(0, sweeps, options, (int i) =>
                {
                    var task = sweeper.Propose();
                    task.Wait();
                    Assert.True(task.Status == TaskStatus.RanToCompletion);
                    var paramWithId = task.Result;
                    if (paramWithId == null)
                        return;
                    Thread.Sleep(sleeps[i]);
                    var result = new RunResult(paramWithId.ParameterSet, 0.42, true);
                    sweeper.Update(paramWithId.Id, result);
                    lock (mlock)
                        paramSets.Add(paramWithId.ParameterSet);
                });
                Assert.True(paramSets.Count <= sweeps);
                CheckAsyncSweeperResult(paramSets);
            }
        }

        [Fact(Skip = "The test is not able to load ldrandpl")]
        public void TestNelderMeadSweeperAsync()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                int batchSize = 5;
                int sweeps = 40;
                var paramSets = new List<ParameterSet>();
                var args = new DeterministicSweeperAsync.Arguments();
                args.BatchSize = batchSize;
                args.Relaxation = 0;
                args.Sweeper = new SubComponent<ISweeper, SignatureSweeper>("NM", "p=fp{name=foo min=1 max=5}",
                    "p=lp{name=bar min=1 max=1000 logbase+}");
                var sweeper = new DeterministicSweeperAsync(env, args);
                var mlock = new object();
                double[] metrics = new double[sweeps];
                for (int i = 0; i < metrics.Length; i++)
                    metrics[i] = random.NextDouble();

                for (int i = 0; i < sweeps; i++)
                {
                    var task = sweeper.Propose();
                    task.Wait();
                    var paramWithId = task.Result;
                    if (paramWithId == null)
                        return;
                    var result = new RunResult(paramWithId.ParameterSet, metrics[i], true);
                    sweeper.Update(paramWithId.Id, result);
                    lock (mlock)
                        paramSets.Add(paramWithId.ParameterSet);
                }
                Assert.True(paramSets.Count <= sweeps);
                CheckAsyncSweeperResult(paramSets);
            }
        }

        private void CheckAsyncSweeperResult(List<ParameterSet> paramSets)
        {
            Assert.NotNull(paramSets);
            foreach (var paramSet in paramSets)
            {
                foreach (var parameterValue in paramSet)
                {
                    if (parameterValue.Name == "foo")
                    {
                        var val = Float.Parse(parameterValue.ValueText, CultureInfo.InvariantCulture);
                        Assert.True(val >= 1);
                        Assert.True(val <= 5);
                    }
                    else if (parameterValue.Name == "bar")
                    {
                        var val = long.Parse(parameterValue.ValueText);
                        Assert.True(val >= 1);
                        Assert.True(val <= 1000);
                    }
                    else
                    {
                        Assert.True(false, "Wrong parameter");
                    }
                }
            }
        }

        [Fact]
        public void TestRandomGridSweeper()
        {
            var args = new RandomGridSweeper.Arguments();
            using (var env = new TlcEnvironment(42))
            {
                args.SweptParameters = new[] {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=foo", "min=10", "max=20", "numsteps=3"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=100", "max=10000", "logbase+", "stepsize=10"),
                };
                var sweeper = new RandomGridSweeper(env, args);
                var initialList = sweeper.ProposeSweeps(5, new List<RunResult>());
                Assert.Equal(5, initialList.Length);
                var gridPoint = new bool[3][] {
                    new bool[3],
                    new bool[3],
                    new bool[3]
                };
                int i = 0;
                int j = 0;
                foreach (var parameterSet in initialList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 10 || val == 15 || val == 20);
                            i = (val == 10) ? 0 : (val == 15) ? 1 : 2;
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 100 || val == 1000 || val == 10000);
                            j = (val == 100) ? 0 : (val == 1000) ? 1 : 2;
                        }
                        else
                        {
                            Assert.True(false, "Wrong parameter");
                        }
                    }
                    Assert.False(gridPoint[i][j]);
                    gridPoint[i][j] = true;
                }

                var nextList = sweeper.ProposeSweeps(5, initialList.Select(p => new RunResult(p)));
                Assert.Equal(4, nextList.Length);
                foreach (var parameterSet in nextList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 10 || val == 15 || val == 20);
                            i = (val == 10) ? 0 : (val == 15) ? 1 : 2;
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 100 || val == 1000 || val == 10000);
                            j = (val == 100) ? 0 : (val == 1000) ? 1 : 2;
                        }
                        else
                        {
                            Assert.True(false, "Wrong parameter");
                        }
                    }
                    Assert.False(gridPoint[i][j]);
                    gridPoint[i][j] = true;
                }

                gridPoint = new bool[3][] {
                    new bool[3],
                    new bool[3],
                    new bool[3]
                };
                var lastList = sweeper.ProposeSweeps(10, null);
                Assert.Equal(9, lastList.Length);
                foreach (var parameterSet in lastList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 10 || val == 15 || val == 20);
                            i = (val == 10) ? 0 : (val == 15) ? 1 : 2;
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val == 100 || val == 1000 || val == 10000);
                            j = (val == 100) ? 0 : (val == 1000) ? 1 : 2;
                        }
                        else
                        {
                            Assert.True(false, "Wrong parameter");
                        }
                    }
                    Assert.False(gridPoint[i][j]);
                    gridPoint[i][j] = true;
                }
                Assert.True(gridPoint.All(bArray => bArray.All(b => b)));
            }
        }

        [Fact(Skip = "The test is not able to load ldrandpl")]
        public void TestNelderMeadSweeper()
        {
            var random = new Random(42);
            var args = new NelderMeadSweeper.Arguments();
            using (var env = new TlcEnvironment(42))
            {
                args.SweptParameters = new[] {
                    new SubComponent("fp", "name=foo", "min=1", "max=5"),
                    new SubComponent("lp", "name=bar", "min=1", "max=1000", "logbase+"),
                };
                var sweeper = new NelderMeadSweeper(env, args);
                var sweeps = sweeper.ProposeSweeps(5, new List<RunResult>());
                Assert.Equal(3, sweeps.Length);

                var results = new List<IRunResult>();
                for (int i = 1; i < 10; i++)
                {
                    foreach (var parameterSet in sweeps)
                    {
                        foreach (var parameterValue in parameterSet)
                        {
                            if (parameterValue.Name == "foo")
                            {
                                var val = Float.Parse(parameterValue.ValueText, CultureInfo.InvariantCulture);
                                Assert.True(val >= 1);
                                Assert.True(val <= 5);
                            }
                            else if (parameterValue.Name == "bar")
                            {
                                var val = long.Parse(parameterValue.ValueText);
                                Assert.True(val >= 1);
                                Assert.True(val <= 1000);
                            }
                            else
                            {
                                Assert.True(false, "Wrong parameter");
                            }
                        }
                        results.Add(new RunResult(parameterSet, random.NextDouble(), true));
                    }

                    sweeps = sweeper.ProposeSweeps(5, results);
                }
                Assert.True(sweeps.Length <= 5);
            }
        }

#if !PORTED
        [Fact]
        public void TestSmacSweeper()
        {
            RunMTAThread(() =>
            {
                var random = new Random(42);
                var args = new SmacSweeper.Arguments();
                using (var env = new TlcEnvironment(42))
                {
                    int maxInitSweeps = 5;
                    args.NumberInitialPopulation = 20;
                    args.SweptParameters = new[] {
                        new SubComponent<IValueGenerator, SignatureSweeperParameter>("fp", "name=foo", "min=1", "max=5"),
                        new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=1", "max=1000", "logbase+"),
                    };
                    var sweeper = new SmacSweeper(env, args);
                    var results = new List<IRunResult>();
                    var sweeps = sweeper.ProposeSweeps(maxInitSweeps, results);
                    Assert.Equal(Math.Min(args.NumberInitialPopulation, maxInitSweeps), sweeps.Length);

                    for (int i = 1; i < 10; i++)
                    {
                        foreach (var parameterSet in sweeps)
                        {
                            foreach (var parameterValue in parameterSet)
                            {
                                if (parameterValue.Name == "foo")
                                {
                                    var val = Float.Parse(parameterValue.ValueText, CultureInfo.InvariantCulture);
                                    Assert.True(val >= 1);
                                    Assert.True(val <= 5);
                                }
                                else if (parameterValue.Name == "bar")
                                {
                                    var val = long.Parse(parameterValue.ValueText);
                                    Assert.True(val >= 1);
                                    Assert.True(val <= 1000);
                                }
                                else
                                {
                                    Assert.True(false, "Wrong parameter");
                                }
                            }
                            results.Add(new RunResult(parameterSet, random.NextDouble(), true));
                        }

                        sweeps = sweeper.ProposeSweeps(5, results);
                    }
                    Assert.True(sweeps.Length == 5);
                }
            });
        }

        [Fact]
        public void TestSimpleSweeperAsyncParallel()
        {
            var random = new Random(42);
            using (var env = new TlcEnvironment(42))
            {
                var args = new LowDiscrepancyRandomSweeper.Arguments();
                args.SweptParameters = new[] {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("fp", "name=foo", "min=1", "max=5"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=1", "max=1000", "logbase+"),
                };
                var sweeper = new SimpleAsyncSweeper(env, args);
                int sweeps = 100;

                var paramSets = new List<ParameterSet>();
                var options = new ParallelOptions();
                options.MaxDegreeOfParallelism = 4;
                paramSets.Clear();
                object mlock = new object();
                var r = Parallel.For(0, sweeps, options, (int i) =>
                {
                    var task = sweeper.Propose();
                    Assert.True(task.IsCompleted);
                    lock (mlock)
                        paramSets.Add(task.Result.ParameterSet);
                    Thread.Sleep(100);
                    var result = new RunResult(task.Result.ParameterSet, 0.42, true);
                    sweeper.Update(task.Result.Id, result);
                });
                Contracts.Assert(r.IsCompleted);
                Assert.Equal(sweeps, paramSets.Count);
                CheckAsyncSweeperResult(paramSets);
            }
        }

        [Fact]
        public void TestSobolRandomSweeper()
        {
            var args = new LowDiscrepancyRandomSweeper.Arguments();
            using (var env = new TlcEnvironment(42))
            {
                args.SweptParameters = new[] {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=foo", "min=10", "max=20"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=100", "max=200"),
                };
                args.UseSobol = true;
                var sweeper = new LowDiscrepancyRandomSweeper(env, args);
                var initialList = sweeper.ProposeSweeps(5, new List<RunResult>());
                Assert.Equal(5, initialList.Length);
                foreach (var parameterSet in initialList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 10);
                            Assert.True(val <= 20);
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 100);
                            Assert.True(val <= 200);
                        }
                        else
                            Assert.Fail("Wrong parameter");
                    }
                }
            }
        }

        [Fact]
        public void TestNiederreiterRandomSweeper()
        {
            var args = new LowDiscrepancyRandomSweeper.Arguments();
            using (var env = new TlcEnvironment(42))
            {
                args.SweptParameters = new[] {
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=foo", "min=10", "max=20"),
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>("lp", "name=bar", "min=100", "max=200"),
                };
                var sweeper = new LowDiscrepancyRandomSweeper(env, args);
                var initialList = sweeper.ProposeSweeps(5, new List<RunResult>());
                Assert.Equal(5, initialList.Length);
                foreach (var parameterSet in initialList)
                {
                    foreach (var parameterValue in parameterSet)
                    {
                        if (parameterValue.Name == "foo")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 10);
                            Assert.True(val <= 20);
                        }
                        else if (parameterValue.Name == "bar")
                        {
                            var val = long.Parse(parameterValue.ValueText);
                            Assert.True(val >= 100);
                            Assert.True(val <= 200);
                        }
                        else
                            Assert.Fail("Wrong parameter");
                    }
                }
            }
        }
#endif
    }
}
