// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data.DataLoadSave
{
    /// <summary>
    /// A fake schema that is manufactured out of a SchemaShape.
    /// It will pretend that all vector sizes are equal to 10, all key value counts are equal to 10,
    /// and all values are defaults (for metadata).
    /// </summary>
    internal static class FakeSchemaFactory
    {
        private const int AllVectorSizes = 10;
        private const int AllKeySizes = 10;

        public static Schema Create(SchemaShape shape)
        {
            var builder = new SchemaBuilder();

            for (int i = 0; i < shape.Count; ++i)
            {
                var metaBuilder = new MetadataBuilder();
                var partialMetadata = shape[i].Metadata;
                for (int j = 0; j < partialMetadata.Count; ++j)
                {
                    var metaColumnType = MakeColumnType(partialMetadata[i]);
                    Delegate del;
                    if (metaColumnType is VectorType vectorType)
                        del = Utils.MarshalInvoke(GetDefaultVectorGetter<int>, vectorType.ItemType.RawType);
                    else
                        del = Utils.MarshalInvoke(GetDefaultGetter<int>, metaColumnType.RawType);
                    metaBuilder.Add(partialMetadata[j].Name, metaColumnType, del);
                }
                builder.AddColumn(shape[i].Name, MakeColumnType(shape[i]));
            }
            return builder.GetSchema();
        }

        private static ColumnType MakeColumnType(SchemaShape.Column column)
        {
            ColumnType curType = column.ItemType;
            if (column.IsKey)
                curType = new KeyType(((PrimitiveType)curType).RawType, AllKeySizes);
            if (column.Kind == SchemaShape.Column.VectorKind.VariableVector)
                curType = new VectorType((PrimitiveType)curType, 0);
            else if (column.Kind == SchemaShape.Column.VectorKind.Vector)
                curType = new VectorType((PrimitiveType)curType, AllVectorSizes);
            return curType;
        }

        private static Delegate GetDefaultVectorGetter<TValue>()
        {
            ValueGetter<VBuffer<TValue>> getter = (ref VBuffer<TValue> value) => value = new VBuffer<TValue>(AllVectorSizes, 0, null, null);
            return getter;
        }

        private static Delegate GetDefaultGetter<TValue>()
        {
            ValueGetter<TValue> getter = (ref TValue value) => value = default;
            return getter;
        }

    }
}
