// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Recommender
{
    internal static class RecommenderUtils
    {
        /// <summary>
        /// Check if the considered data, <see cref="RoleMappedData"/>, contains column roles specified by <see cref="MatrixColumnIndexKind"/> and <see cref="MatrixRowIndexKind"/>.
        /// If the column roles, <see cref="MatrixColumnIndexKind"/> and <see cref="MatrixRowIndexKind"/>, uniquely exist in data, their <see cref="DataViewSchema.Column"/> would be assigned
        /// to the two out parameters below.
        /// </summary>
        /// <param name="data">The considered data being checked</param>
        /// <param name="matrixColumnIndexColumn">The schema column as the row in the input data</param>
        /// <param name="matrixRowIndexColumn">The schema column as the column in the input data</param>
        /// <param name="isDecode">Whether a non-user error should be thrown as a decode</param>
        public static void CheckAndGetMatrixIndexColumns(RoleMappedData data, out DataViewSchema.Column matrixColumnIndexColumn, out DataViewSchema.Column matrixRowIndexColumn, bool isDecode)
        {
            Contracts.AssertValue(data);
            CheckRowColumnType(data, MatrixColumnIndexKind, out matrixColumnIndexColumn, isDecode);
            CheckRowColumnType(data, MatrixRowIndexKind, out matrixRowIndexColumn, isDecode);
        }

        /// <summary>
        /// Returns whether a type is a U4 key of known cardinality, and if so, sets
        /// <paramref name="keyType"/> to a non-null value.
        /// </summary>
        private static bool TryMarshalGoodRowColumnType(DataViewType type, out KeyType keyType)
        {
            keyType = type as KeyType;
            return keyType?.Count > 0 && type.RawType == typeof(uint);
        }

        /// <summary>
        /// Checks whether a column kind in a <see cref="RoleMappedData"/> is unique, and its type
        /// is a <see cref="InternalDataKind.U4"/> key of known cardinality.
        /// </summary>
        /// <param name="data">The training examples</param>
        /// <param name="role">The column role to try to extract</param>
        /// <param name="col">The extracted schema column</param>
        /// <param name="isDecode">Whether a non-user error should be thrown as a decode</param>
        /// <returns>The type cast to a key-type</returns>
        private static KeyType CheckRowColumnType(RoleMappedData data, RoleMappedSchema.ColumnRole role, out DataViewSchema.Column col, bool isDecode)
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
            col = data.Schema.GetColumns(role)[0];

            // REVIEW tfinley: Should we be a bit less restrictive? This doesn't seem like
            // too terrible of a restriction.
            const string format = "Column '{0}' with role {1} should be a known cardinality U4 key, but is instead '{2}'";
            KeyType keyType;
            if (!TryMarshalGoodRowColumnType(col.Type, out keyType))
            {
                if (isDecode)
                    throw Contracts.ExceptDecode(format, col.Name, role.Value, col.Type);
                throw Contracts.Except(format, col.Name, role.Value, col.Type);
            }
            return keyType;
        }

        /// <summary>
        /// The column role that is treated as column index in matrix factorization problem
        /// </summary>
        public static RoleMappedSchema.ColumnRole MatrixColumnIndexKind => "MatrixColumnIndex";

        /// <summary>
        /// The column role that is treated as row index in matrix factorization problem
        /// </summary>
        public static RoleMappedSchema.ColumnRole MatrixRowIndexKind => "MatrixRowIndex";
    }
}
