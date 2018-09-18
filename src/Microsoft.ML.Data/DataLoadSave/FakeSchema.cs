// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Data.DataLoadSave
{

    /// <summary>
    /// A fake schema that is manufactured out of a SchemaShape.
    /// It will pretend that all vector sizes are equal to 10, all key value counts are equal to 10,
    /// and all values are defaults (for metadata).
    /// </summary>
    internal sealed class FakeSchema : ISchema
    {
        private const int AllVectorSizes = 10;
        private const int AllKeySizes = 10;

        private readonly IHostEnvironment _env;
        private readonly SchemaShape _shape;
        private readonly Dictionary<string, int> _colMap;

        public FakeSchema(IHostEnvironment env, SchemaShape inputShape)
        {
            _env = env;
            _shape = inputShape;
            _colMap = Enumerable.Range(0, _shape.Columns.Length)
                .ToDictionary(idx => _shape.Columns[idx].Name, idx => idx);
        }

        public int ColumnCount => _shape.Columns.Length;

        public string GetColumnName(int col)
        {
            _env.Check(0 <= col && col < ColumnCount);
            return _shape.Columns[col].Name;
        }

        public ColumnType GetColumnType(int col)
        {
            _env.Check(0 <= col && col < ColumnCount);
            var inputCol = _shape.Columns[col];
            return MakeColumnType(inputCol);
        }

        public bool TryGetColumnIndex(string name, out int col) => _colMap.TryGetValue(name, out col);

        private static ColumnType MakeColumnType(SchemaShape.Column inputCol)
        {
            ColumnType curType = inputCol.ItemType;
            if (inputCol.IsKey)
                curType = new KeyType(curType.AsPrimitive.RawKind, 0, AllKeySizes);
            if (inputCol.Kind == SchemaShape.Column.VectorKind.VariableVector)
                curType = new VectorType(curType.AsPrimitive, 0);
            else if (inputCol.Kind == SchemaShape.Column.VectorKind.Vector)
                curType = new VectorType(curType.AsPrimitive, AllVectorSizes);
            return curType;
        }

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            _env.Check(0 <= col && col < ColumnCount);
            var inputCol = _shape.Columns[col];
            var metaShape = inputCol.Metadata;
            if (metaShape == null || !metaShape.TryFindColumn(kind, out var metaColumn))
                throw _env.ExceptGetMetadata();

            var colType = MakeColumnType(metaColumn);
            _env.Check(colType.RawType.Equals(typeof(TValue)));

            if (colType.IsVector)
            {
                // This as an atypical use of VBuffer: we create it in GetMetadataVec, and then pass through
                // via boxing to be returned out of this method. This is intentional.
                value = (TValue)Utils.MarshalInvoke(GetMetadataVec<int>, colType.ItemType.RawType);
            }
            else
                value = default;
        }

        private object GetMetadataVec<TItem>() => new VBuffer<TItem>(AllVectorSizes, 0, null, null);

        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            _env.Check(0 <= col && col < ColumnCount);
            var inputCol = _shape.Columns[col];
            var metaShape = inputCol.Metadata;
            if (metaShape == null || !metaShape.TryFindColumn(kind, out var metaColumn))
                return null;
            return MakeColumnType(metaColumn);
        }

        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            _env.Check(0 <= col && col < ColumnCount);
            var inputCol = _shape.Columns[col];
            var metaShape = inputCol.Metadata;
            if (metaShape == null)
                return Enumerable.Empty<KeyValuePair<string, ColumnType>>();

            return metaShape.Columns.Select(c => new KeyValuePair<string, ColumnType>(c.Name, MakeColumnType(c)));
        }
    }
}
