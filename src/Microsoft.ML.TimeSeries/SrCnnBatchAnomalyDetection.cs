// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataView;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// The detect modes of SrCnn models.
    /// </summary>
    public enum SrCnnDetectMode
    {
        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag).
        /// </summary>
        AnomalyOnly = 0,

        /// <summary>
        /// In this mode, output (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
        /// </summary>
        AnomalyAndMargin = 1,

        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag, ExpectedValue).
        /// </summary>
        AnomalyAndExpectedValue = 2
    }

    // TODO: SrCnn
    internal sealed class SrCnnBatchAnomalyDetector : BatchDataViewMapperBase<double, SrCnnBatchAnomalyDetector.Batch>
    {
        private const int MinBatchSize = 12;
        private readonly int _batchSize;
        private readonly string _inputColumnName;
        private readonly SrCnnDetectMode _detectMode;

        private class Bindings : ColumnBindingsBase
        {
            private readonly VectorDataViewType _outputColumnType;
            private readonly int _inputColumnIndex;

            public Bindings(DataViewSchema input, string inputColumnName, string outputColumnName, VectorDataViewType outputColumnType)
                : base(input, true, outputColumnName)
            {
                _outputColumnType = outputColumnType;
                _inputColumnIndex = Input[inputColumnName].Index;
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Check(iinfo == 0);
                return _outputColumnType;
            }

            // Get a predicate for the input columns.
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.Count];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                        active[_inputColumnIndex] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public SrCnnBatchAnomalyDetector(IHostEnvironment env, IDataView input, string inputColumnName, string outputColumnName, double threshold, int batchSize, double sensitivity, SrCnnDetectMode detectMode)
            : base(env, "SrCnnBatchAnomalyDetector", input)
        {

            Contracts.CheckParam(batchSize >= MinBatchSize, nameof(batchSize), "batch size is too small");
            _detectMode = detectMode;
            int outputSize = 6; // TODO: determine based on detectMode
            SchemaBindings = new Bindings(input.Schema, inputColumnName, outputColumnName, new VectorDataViewType(NumberDataViewType.Double, outputSize));
            _batchSize = batchSize;
            _inputColumnName = inputColumnName;
        }

        protected override ColumnBindingsBase SchemaBindings { get; }

        protected override Delegate[] CreateGetters(DataViewRowCursor input, Batch currentBatch, bool[] active)
        {
            if (!SchemaBindings.AnyNewColumnsActive(x => active[x]))
                return new Delegate[1];
            return new[] { currentBatch.CreateGetter(input, _inputColumnName) };
        }

        protected override Batch InitializeBatch(DataViewRowCursor input) => new Batch(_batchSize, _detectMode);

        protected override Func<bool> GetIsNewBatchDelegate(DataViewRowCursor input)
        {
            return () => input.Position % _batchSize == 0;
        }

        protected override Func<bool> GetLastInBatchDelegate(DataViewRowCursor input)
        {
            return () => (input.Position + 1) % _batchSize == 0;
        }

        protected override ValueGetter<double> GetLookAheadGetter(DataViewRowCursor input)
        {
            return input.GetGetter<double>(input.Schema[_inputColumnName]);
        }

        protected override Func<int, bool> GetSchemaBindingDependencies(Func<int, bool> predicate)
        {
            return (SchemaBindings as Bindings).GetDependencies(predicate);
        }

        protected override void ProcessExample(Batch currentBatch, double currentInput)
        {
            currentBatch.AddValue(currentInput);
        }

        protected override void ProcessBatch(Batch currentBatch)
        {
            currentBatch.Process();
            currentBatch.Reset();
        }

        public sealed class Batch
        {
            private List<double> _previousBatch;
            private List<double> _batch;
            private readonly SrCnnDetectMode _detectMode;
            private double _cursor;

            public Batch(int batchSize, SrCnnDetectMode detectMode)
            {
                _detectMode = detectMode;
                _previousBatch = new List<double>(batchSize);
                _batch = new List<double>(batchSize);
            }

            public void AddValue(double value)
            {
                _batch.Add(value);
            }

            public int Count => _batch.Count;

            public void Process()
            {
                // TODO: replace with run of SrCnn
                _cursor = _batch.Sum();
            }

            public void Reset()
            {
                var tempBatch = _previousBatch;
                _previousBatch = _batch;
                _batch = tempBatch;
                _batch.Clear();
            }

            public ValueGetter<VBuffer<double>> CreateGetter(DataViewRowCursor input, string inputCol)
            {
                ValueGetter<double> srcGetter = input.GetGetter<double>(input.Schema[inputCol]);
                ValueGetter<VBuffer<double>> getter =
                    (ref VBuffer<double> dst) =>
                    {
                        double src = default;
                        srcGetter(ref src);
                        // TODO: replace with SrCnn result
                        dst = new VBuffer<double>(6, new[] {
                            src * _cursor,
                            (src + 1) * _cursor,
                            (src + 2) * _cursor,
                            (src + 3) * _cursor,
                            (src + 4) * _cursor,
                            (src + 5) * _cursor,
                        });
                    };
                return getter;
            }
        }
    }
}
