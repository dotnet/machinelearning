// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Xunit;

namespace Microsoft.ML.PerformanceTests
{
    [Config(typeof(TrainConfig))]
    public class TextLoaderBench : BenchmarkBase
    {
        private MLContext _mlContext;
        private IDataView _dataView;
        private static int _numColumns = 100;
        private static int _numRows = 3000;
        private static int _maxWordLength = 15;
        private static int _numColumnsToGet = 20;
        private List<TextLoader.Column> _columns;


        [GlobalSetup]
        public void SetupData()
        {
            _mlContext = new MLContext(seed: 1);
            var path = Path.GetTempFileName();
            Console.WriteLine($"Created dataset in temporary file:\n{path}\n");
            path = RandomFile.CreateRandomFile(path, _numRows, _numColumns, _maxWordLength);

            _columns = new List<TextLoader.Column>();
            for (int i = 0; i < _numColumns; i++)
            {
                _columns.Add(new TextLoader.Column($"Column{i}", DataKind.String, i));
            }

            var textLoader = _mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = _columns.ToArray(),
                HasHeader = false,
                Separators = new char[] { ',' },
                AllowQuoting = true,
                ReadMultilines = true,
                EscapeChar = '\\',
            });

            _dataView = textLoader.Load(path);
        }

        [Benchmark]
        public void TestTextLoaderGetters()
        {
            using (var rowCursor = _dataView.GetRowCursorForAllColumns())
            {
                var getters = new List<ValueGetter<ReadOnlyMemory<char>>>();
                for (int i = 0; i < _numColumnsToGet; i++)
                {
                    getters.Add(rowCursor.GetGetter<ReadOnlyMemory<char>>(_dataView.Schema[i]));
                }

                ReadOnlyMemory<char> buff = default;
                while (rowCursor.MoveNext())
                {
                    for (int i = 0; i < _numColumnsToGet; i++)
                        getters[i](ref buff);
                }
            }

            //* Summary *

            //BenchmarkDotNet = v0.12.0, OS = Windows 10.0.18363
            //Intel Core i7 - 8650U CPU 1.90GHz(Kaby Lake R), 1 CPU, 8 logical and 4 physical cores
            //.NET Core SDK = 3.1.100 - preview3 - 014645
            //    [Host]     : .NET Core 2.1.13(CoreCLR 4.6.28008.01, CoreFX 4.6.28008.01), X64 RyuJIT
            //  Job - XQBLAM : .NET Core 2.1.13(CoreCLR 4.6.28008.01, CoreFX 4.6.28008.01), X64 RyuJIT

            //Arguments =/ p:Configuration = Release  Toolchain = netcoreapp2.1  IterationCount = 1
            //LaunchCount = 3  MaxIterationCount = 20  RunStrategy = ColdStart
            //UnrollFactor = 1  WarmupCount = 1

            //| Method                  | Mean      | Error     | StdDev    | Extra Metric  |
            //| ----------------------  | --------: | ---------:| ---------:| -------------:|
            //| TestTextLoaderGetters   | 1.012 s   | 0.6649 s  | 0.0364 s  | -             |

            //// * Legends *
            //Mean         : Arithmetic mean of all measurements
            //Error        : Half of 99.9 % confidence interval
            // StdDev       : Standard deviation of all measurements
            // Extra Metric: Value of the provided extra metric
            //  1 s: 1 Second(1 sec)

            //// ***** BenchmarkRunner: End *****
            //// ** Remained 0 benchmark(s) to run **
            //            Run time: 00:00:16(16.05 sec), executed benchmarks: 1

            //Global total time: 00:00:33(33.18 sec), executed benchmarks: 1

            return;
        }
    }
}
