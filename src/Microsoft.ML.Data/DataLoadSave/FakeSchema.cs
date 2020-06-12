// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data.DataLoadSave
{
    /// <summary>
    /// A fake schema that is manufactured out of a SchemaShape.
    /// It will pretend that all vector sizes are equal to 10, all key value counts are equal to 10,
    /// and all values are defaults (for annotations).
    /// </summary>
    [BestFriend]
    internal static class FakeSchemaFactory
    {
        private static readonly FuncStaticMethodInfo1<Delegate> _getDefaultVectorGetterMethodInfo = new FuncStaticMethodInfo1<Delegate>(GetDefaultVectorGetter<int>);
        private static readonly FuncStaticMethodInfo1<Delegate> _getDefaultGetterMethodInfo = new FuncStaticMethodInfo1<Delegate>(GetDefaultGetter<int>);

        private const int AllVectorSizes = 10;
        private const int AllKeySizes = 10;

        public static DataViewSchema Create(SchemaShape shape)
        {
            var builder = new DataViewSchema.Builder();

            for (int i = 0; i < shape.Count; ++i)
            {
                var metaBuilder = new DataViewSchema.Annotations.Builder();
                var partialAnnotations = shape[i].Annotations;
                for (int j = 0; j < partialAnnotations.Count; ++j)
                {
                    var metaColumnType = MakeColumnType(partialAnnotations[j]);
                    Delegate del;
                    if (metaColumnType is VectorDataViewType vectorType)
                        del = Utils.MarshalInvoke(_getDefaultVectorGetterMethodInfo, vectorType.ItemType.RawType);
                    else
                        del = Utils.MarshalInvoke(_getDefaultGetterMethodInfo, metaColumnType.RawType);
                    metaBuilder.Add(partialAnnotations[j].Name, metaColumnType, del);
                }
                builder.AddColumn(shape[i].Name, MakeColumnType(shape[i]), metaBuilder.ToAnnotations());
            }
            return builder.ToSchema();
        }

        private static DataViewType MakeColumnType(SchemaShape.Column column)
        {
            DataViewType curType = column.ItemType;
            if (column.IsKey)
                curType = new KeyDataViewType(((PrimitiveDataViewType)curType).RawType, AllKeySizes);
            if (column.Kind == SchemaShape.Column.VectorKind.VariableVector)
                curType = new VectorDataViewType((PrimitiveDataViewType)curType, 0);
            else if (column.Kind == SchemaShape.Column.VectorKind.Vector)
                curType = new VectorDataViewType((PrimitiveDataViewType)curType, AllVectorSizes);
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
