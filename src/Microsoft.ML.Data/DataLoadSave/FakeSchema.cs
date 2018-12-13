// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Data.DataLoadSave
{
    /// <summary>
    /// A fake schema that is manufactured out of a SchemaShape.
    /// It will pretend that all vector sizes are equal to 10, all key value counts are equal to 10,
    /// and all values are defaults (for metadata).
    /// </summary>
    internal sealed class FakeSchemaFactory
    {
        private const int AllVectorSizes = 10;
        private const int AllKeySizes = 10;
        private readonly SchemaShape _shape;

        public FakeSchemaFactory(SchemaShape inputShape)
        {
            _shape = inputShape;
        }

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
                    var del = GetMetadataGetter(metaColumnType);
                    metaBuilder.Add(partialMetadata[j].Name, metaColumnType, del);
                }
                builder.AddColumn(shape[i].Name, MakeColumnType(shape[i]));
            }
            return builder.GetSchema();
        }

        private static ColumnType MakeColumnType(SchemaShape.Column inputCol)
        {
            ColumnType curType = inputCol.ItemType;
            if (inputCol.IsKey)
                curType = new KeyType(((PrimitiveType)curType).RawKind, 0, AllKeySizes);
            if (inputCol.Kind == SchemaShape.Column.VectorKind.VariableVector)
                curType = new VectorType((PrimitiveType)curType, 0);
            else if (inputCol.Kind == SchemaShape.Column.VectorKind.Vector)
                curType = new VectorType((PrimitiveType)curType, AllVectorSizes);
            return curType;
        }

        private static Delegate GetMetadataGetter(ColumnType colType)
        {
            if (colType.IsVector)
                return Utils.MarshalInvoke(GetDefaultVectorGetter<int>, colType.ItemType.RawType);
            else
                return Utils.MarshalInvoke(GetDefaultGetter<int>, colType.RawType);
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
