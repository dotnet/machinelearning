// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

namespace Microsoft.ML.Transforms
{

    internal sealed class ShortGrainDropperDataView : IDataTransform
    {
        private ShortDropTransformer _parent;

        #region Native Exports

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_Transform"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, out bool skipRow, out IntPtr errorHandle);

        #endregion

        private readonly IHostEnvironment _host;
        private readonly IDataView _source;
        private readonly string[] _grainColumns;
        private readonly DataViewSchema _schema;

        internal ShortGrainDropperDataView(IHostEnvironment env, IDataView input, string[] grainColumns, ShortDropTransformer parent)
        {
            _host = env;
            _source = input;

            _grainColumns = grainColumns;
            _parent = parent;

            // Use existing schema since it doesn't change.
            _schema = _source.Schema;
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            var input = _source.GetRowCursorForAllColumns();
            return new Cursor(_host, input, _parent.CloneTransformer(), _grainColumns, _schema);
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
            private readonly IChannelProvider _ch;
            private DataViewRowCursor _input;
            private long _position;
            private bool _isGood;
            private readonly DataViewSchema _schema;
            private readonly TransformerEstimatorSafeHandle _transformer;
            private ValueGetter<ReadOnlyMemory<char>>[] _grainGetters;

            // These are class variables so they are only allocated once.
            private GCHandle[] _grainHandles;
            private IntPtr[] _grainArray;
            private GCHandle _arrayHandle;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, TransformerEstimatorSafeHandle transformer, string[] grainColumns, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                _input = input;
                _position = -1;
                _schema = schema;
                _transformer = transformer;
                _grainGetters = new ValueGetter<ReadOnlyMemory<char>>[grainColumns.Length];

                _grainHandles = new GCHandle[grainColumns.Length];
                _grainArray = new IntPtr[grainColumns.Length];

                InitializeGrainGetters(grainColumns, ref _grainGetters);
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

            protected override void Dispose(bool disposing)
            {
                if (!_transformer.IsClosed)
                    _transformer.Close();
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type.
            /// Since all we are doing is dropping rows, we can just use the source getter.
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                _ch.Check(IsColumnActive(column));

                return _input.GetGetter<TValue>(column);
            }

            public override bool MoveNext()
            {
                _position++;
                while (true)
                {
                    // If there are no more source rows exit loop and return false.
                    _isGood = _input.MoveNext();
                    if (!_isGood)
                        break;

                    try
                    {
                        CreateGrainStringArrays(_grainGetters, ref _grainHandles, ref _arrayHandle, ref _grainArray);
                        var success = TransformDataNative(_transformer, _arrayHandle.AddrOfPinnedObject(), new IntPtr(_grainArray.Length), out bool skipRow, out IntPtr errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If native featurizer returns true it means to skip this row, so stay in loop.
                        // If it returns false then use it, so exit loop.
                        if (!skipRow)
                            break;
                    }
                    finally
                    {
                        FreeGrainStringArrays(ref _grainHandles, ref _arrayHandle);
                    }
                }

                return _isGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _input.Batch;

            private void InitializeGrainGetters(string[] grainColumns, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
            {
                // Create getters for the source grain columns.

                for (int i = 0; i < _grainGetters.Length; i++)
                {
                    // Inititialize the getter and move it to a valid position.
                    grainGetters[i] = _input.GetGetter<ReadOnlyMemory<char>>(_input.Schema[grainColumns[i]]);
                }
            }
        }
    }
}
