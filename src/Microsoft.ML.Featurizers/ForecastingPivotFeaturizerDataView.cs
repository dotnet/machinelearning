// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using static Microsoft.ML.Featurizers.CommonExtensions;

namespace Microsoft.ML.Transforms
{

    internal sealed class ForecastingPivotFeaturizerDataView : IDataTransform
    {
        #region Typed Columns
        private ForecastingPivotTransformer _parent;

        #endregion

        private readonly IHostEnvironment _host;
        private readonly IDataView _source;
        private readonly ForecastingPivotFeaturizerEstimator.Options _options;
        private readonly DataViewSchema _schema;

        internal ForecastingPivotFeaturizerDataView(IHostEnvironment env, IDataView input, ForecastingPivotFeaturizerEstimator.Options options, ForecastingPivotTransformer parent)
        {
            _host = env;
            _source = input;

            _options = options;
            _parent = parent;

            _schema = _parent.GetOutputSchema(input.Schema);
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            var input = _source.GetRowCursorForAllColumns();
            return new Cursor(_host, input, _options, _schema);
        }

        // Can't use parallel cursors so this defaults to calling non-parallel version
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
             new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

        // Since we may delete rows we don't know the row count
        public long? GetRowCount() => null;

        public void Save(ModelSaveContext ctx)
        {
            _parent.Save(ctx);
        }

        private sealed class Cursor : DataViewRowCursor
        {
            private class PivotColumn
            {
                private ValueGetter<VBuffer<double>> _getter;
                private VBuffer<double> _curValues;

                private readonly int _row;
                private int _curCol;

                public PivotColumn(DataViewRowCursor input, string sourceColumnName, int row)
                {
                    _getter = input.GetGetter<VBuffer<double>>(input.Schema[sourceColumnName]);
                    _curValues = default;

                    var dimensions = ((VectorDataViewType)(input.Schema[sourceColumnName].Type)).Dimensions;
                    ColumnCount = dimensions[1];

                    _row = row;
                    _curCol = 0;
                }

                public int ColumnCount { get; private set; }

                public double GetValueAndSetColumn(int col)
                {
                    _curCol = col;
                    return GetValue(_curCol);
                }

                public void MoveNext()
                {
                    _getter(ref _curValues);
                }

                private double GetValue(int col)
                {
                    return _curValues.GetItemOrDefault((_row * ColumnCount) + col);
                }

                public double GetStoredValue()
                {
                    return GetValue(_curCol);
                }
            }

            private readonly IChannelProvider _ch;
            private DataViewRowCursor _input;
            private long _position;
            private bool _isGood;
            private readonly DataViewSchema _schema;
            private readonly ForecastingPivotFeaturizerEstimator.Options _options;

            private Dictionary<string, PivotColumn> _pivotColumns;
            private readonly UInt32 _maxCols;
            private int _currentCol;
            private UInt32 _currentHorizon;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, ForecastingPivotFeaturizerEstimator.Options options, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                // Start isGood at true. This is not exposed outside of the class.
                _isGood = true;
                _input = input;
                _position = -1;
                _schema = schema;
                _options = options;

                _pivotColumns = new Dictionary<string, PivotColumn>();
                _currentCol = -1;

                foreach (var col in _options.ColumnsToPivot)
                {
                    var sourceCol = input.Schema[col];
                    var annotations = sourceCol.Annotations;

                    // Getting the annotation this way is a temporary fix since even though the values are known at this time,
                    // SchemaShape does not expose them. To work around this the annotations are stored in the format
                    // Annotation=Value. We will just parse this and get the value.
                    var columnNamesAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

                    var columnNames = columnNamesAnnotationName.Split('=')[1].Split(',');

                    for (int i = 0; i < columnNames.Length; i++)
                    {
                        _pivotColumns[columnNames[i]] = new PivotColumn(input, col, i);
                    }
                }

                // All columns should have the same amount of slots in the vector
                _maxCols = (UInt32)_pivotColumns.First().Value.ColumnCount;

                // The horizon starts with the _maxCols + 1 amount and decreases until 1, when it is reset back to this value.
                // The reason it is + 1 is because maxCols is 0 to N, where horizon is 1 to N + 1.
                _currentHorizon = _maxCols + 1;
            }

            public sealed override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                       (ref DataViewRowId val) =>
                       {
                           _ch.Check(_isGood, RowCursorUtils.FetchValueStateError);
                           val = new DataViewRowId((ulong)Position, 0);
                       };
            }

            public sealed override DataViewSchema Schema => _schema;

            /// <summary>
            /// Since rows will be dropped
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => true;

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                _ch.Check(IsColumnActive(column));

                var thisCol = _schema[column.Name];

                if ((_pivotColumns.Keys.Contains(column.Name) && thisCol.Name == column.Name && thisCol.Type == column.Type) || column.Name == _options.HorizonColumnName)
                {
                    if (column.Name == _options.HorizonColumnName)
                        return MakeHorizonGetter() as ValueGetter<TValue>;
                    else
                        return MakePivotGetter(column) as ValueGetter<TValue>;
                }
                else
                {
                    return _input.GetGetter<TValue>(column);
                }
            }

            public override bool MoveNext()
            {
                bool exitLoop = false;
                while (_isGood && !exitLoop)
                {
                    // If we haven't done anything yet or we are at the end of a row, advance our source pointer.
                    // We also need to reset _curHorizon to its max value here.
                    if (_currentCol == _maxCols || _currentCol == -1)
                    {
                        // Advance source coursor and break if no more data.
                        _isGood = _input.MoveNext();
                        if (!_isGood)
                            break;

                        _currentHorizon = _maxCols + 1;
                        _currentCol = 0;
                        foreach (var column in _pivotColumns.Values)
                        {
                            column.MoveNext();
                        }
                    }

                    for (int col = _currentCol; col < _maxCols; col++)
                    {
                        var nanFound = false;

                        foreach (var column in _pivotColumns.Values)
                        {
                            if (double.IsNaN(column.GetValueAndSetColumn(col)))
                            {
                                nanFound = true;
                                break;
                            }
                        }

                        // Break from loop because we now have valid values
                        // Update the _currentCol so we start from the correct place next time.
                        // Decrease currentHorizon by 1.
                        // Update our current position.
                        _currentHorizon -= 1;
                        _currentCol = col + 1;
                        if (!nanFound)
                        {
                            exitLoop = true;
                            _position++;
                            break;
                        }
                    }
                }

                return _isGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _input.Batch;

            private Delegate MakePivotGetter(DataViewSchema.Column column)
            {
                var pivotColumn = _pivotColumns[column.Name];
                ValueGetter<double> result = (ref double dst) =>
                {
                    dst = pivotColumn.GetStoredValue();
                };

                return result;
            }

            private Delegate MakeHorizonGetter()
            {
                ValueGetter<UInt32> result = (ref UInt32 dst) =>
                {
                    dst = _currentHorizon;
                };

                return result;
            }
        }
    }
}
