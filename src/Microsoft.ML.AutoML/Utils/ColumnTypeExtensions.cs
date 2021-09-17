// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class DataViewTypeExtensions
    {
        public static bool IsNumber(this DataViewType columnType)
        {
            return columnType is NumberDataViewType;
        }

        public static bool IsText(this DataViewType columnType)
        {
            return columnType is TextDataViewType;
        }

        public static bool IsBool(this DataViewType columnType)
        {
            return columnType is BooleanDataViewType;
        }

        public static bool IsVector(this DataViewType columnType)
        {
            return columnType is VectorDataViewType;
        }

        public static bool IsKey(this DataViewType columnType)
        {
            return columnType is KeyDataViewType;
        }
    }
}
