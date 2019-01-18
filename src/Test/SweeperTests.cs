using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class SweeperTests
    {
        [Ignore]
        [TestMethod]
        public void Smac2ParamsTest()
        {
            var sweeper = new SmacSweeper(new SmacSweeper.Arguments()
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

        [Ignore]
        [TestMethod]
        public void Smac4ParamsTest()
        {
            var sweeper = new SmacSweeper(new SmacSweeper.Arguments()
            {
                SweptParameters = new INumericValueGenerator[] {
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x1", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x2", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x3", Min = 1, Max = 1000}),
                    new FloatValueGenerator(new FloatParamArguments() { Name = "x4", Min = 1, Max = 1000}),
                },
            });

            Random rand = new Random(0);
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
                        Math.Abs(700 - x4) );

                    RunResult result = new RunResult(p, metric, true);
                    if(bestResult == null || bestResult.MetricValue < metric)
                    {
                        bestResult = result;
                    }
                    results.Add(result);

                    Console.WriteLine($"{metric}\t{x1},{x2},{x3},{x4}");
                }

            }

            Console.WriteLine($"Best: {bestResult.MetricValue}");
        }
    }
}
