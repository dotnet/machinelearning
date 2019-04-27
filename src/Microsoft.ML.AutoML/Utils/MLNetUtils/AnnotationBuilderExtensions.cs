﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class AnnotationBuilderExtensions
    {
        /// <summary>
        /// Add slot names annotation.
        /// </summary>
        /// <param name="builder">The <see cref="T:Microsoft.Data.DataView.DataViewSchema.Annotations.Builder" /> to which to add the slot names.</param>
        /// <param name="size">The size of the slot names vector.</param>
        /// <param name="getter">The getter delegate for the slot names.</param>
        public static void AddSlotNames(this DataViewSchema.Annotations.Builder builder, int size, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter)
        {
            builder.Add("SlotNames", new VectorDataViewType(TextDataViewType.Instance, size), getter, null);
        }

        /// <summary>
        /// Add key values annotation.
        /// </summary>
        /// <typeparam name="TValue">The value type of key values.</typeparam>
        /// <param name="builder">The <see cref="T:Microsoft.Data.DataView.DataViewSchema.Annotations.Builder" /> to which to add the key values.</param>
        /// <param name="size">The size of key values vector.</param>
        /// <param name="valueType">The value type of key values. Its raw type must match <typeparamref name="TValue" />.</param>
        /// <param name="getter">The getter delegate for the key values.</param>
        public static void AddKeyValues<TValue>(this DataViewSchema.Annotations.Builder builder, int size, PrimitiveDataViewType valueType, ValueGetter<VBuffer<TValue>> getter)
        {
            builder.Add("KeyValues", new VectorDataViewType(valueType, size), getter, null);
        }
    }
}
