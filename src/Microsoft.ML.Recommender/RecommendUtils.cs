// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Recommender
{
   internal static class RecommendUtils
    {
        /// <summary>
        /// Check if the considered data, <see cref="RoleMappedData"/>, contains column roles specified by <see cref="XKind"/> and <see cref="YKind"/>.
        /// If the column roles, X and Y, uniquely exists in data, their <see cref="ColumnInfo"/> would be assigned to xColumn and yColumn.
        /// </summary>
        /// <param name="data">The considered data being checked</param>
        /// <param name="xColumn">The column as role X in the input data</param>
        /// <param name="yColumn">The column as role Y in the input data</param>
        /// <param name="isDecode">Whether a non-user error should be thrown as a decode</param>
        public static void CheckAndGetXYColumns(RoleMappedData data, out ColumnInfo xColumn, out ColumnInfo yColumn, bool isDecode)
        {
            Contracts.AssertValue(data);
            CheckRowColumnType(data, XKind, out xColumn, isDecode);
            CheckRowColumnType(data, YKind, out yColumn, isDecode);
        }

        /// <summary>
        /// Returns whether a type is a U4 key of known cardinality, and if so, sets
        /// <paramref name="keyType"/> to a non-null value.
        /// </summary>
        private static bool TryMarshalGoodRowColumnType(ColumnType type, out KeyType keyType)
        {
            keyType = type as KeyType;
            return type.KeyCount > 0 && type.RawKind == DataKind.U4 &&
                   keyType != null;
        }

        /// <summary>
        /// Checks whether a column kind in a RoleMappedData is unique, and its type
        /// is a U4 key of known cardinality.
        /// </summary>
        /// <param name="data">The training examples</param>
        /// <param name="role">The column role to try to extract</param>
        /// <param name="info">The extracted column info</param>
        /// <param name="isDecode">Whether a non-user error should be thrown as a decode</param>
        /// <returns>The type cast to a key-type</returns>
        private static KeyType CheckRowColumnType(RoleMappedData data, RoleMappedSchema.ColumnRole role, out ColumnInfo info, bool isDecode)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValue(role.Value);

            const string format2 = "There should be exactly one column with role {0}, but {1} were found instead";
            if (!data.Schema.HasUnique(role))
            {
                int kindCount = Utils.Size(data.Schema.GetColumns(role));
                if (isDecode)
                    throw Contracts.ExceptDecode(format2, role.Value, kindCount);
                throw Contracts.Except(format2, role.Value, kindCount);
            }
            info = data.Schema.GetColumns(role)[0];

            // REVIEW tfinley: Should we be a bit less restrictive? This doesn't seem like
            // too terrible of a restriction.
            const string format = "Column '{0}' with role {1} should be a known cardinality U4 key, but is instead '{2}'";
            KeyType keyType;
            if (!TryMarshalGoodRowColumnType(info.Type, out keyType))
            {
                if (isDecode)
                    throw Contracts.ExceptDecode(format, info.Name, role.Value, info.Type);
                throw Contracts.Except(format, info.Name, role.Value, info.Type);
            }
            return keyType;
        }

        /// <summary>
        /// The column role that is treated as column index in matrix factorization problem
        /// </summary>
        public static RoleMappedSchema.ColumnRole XKind => "X";

        /// <summary>
        /// The column role that is treated as row index in matrix factorization problem
        /// </summary>
        public static RoleMappedSchema.ColumnRole YKind => "Y";
    }
}
