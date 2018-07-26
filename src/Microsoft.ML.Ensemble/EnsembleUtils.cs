// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Ensemble
{
    internal static class EnsembleUtils
    {
        /// <summary>
        /// Return a dataset with non-selected features zeroed out.
        /// </summary>
        public static RoleMappedData SelectFeatures(IHost host, RoleMappedData data, BitArray features)
        {
            Contracts.AssertValue(host);
            Contracts.AssertValue(data);
            Contracts.Assert(data.Schema.Feature != null);
            Contracts.AssertValue(features);

            var type = data.Schema.Feature.Type;
            Contracts.Assert(features.Length == type.VectorSize);
            int card = Utils.GetCardinality(features);
            if (card == type.VectorSize)
                return data;

            // REVIEW: This doesn't preserve metadata on the features column. Should it?
            var name = data.Schema.Feature.Name;
            var view = LambdaColumnMapper.Create(
                host, "FeatureSelector", data.Data, name, name, type, type,
                (ref VBuffer<Single> src, ref VBuffer<Single> dst) => SelectFeatures(ref src, features, card, ref dst));

            var res = new RoleMappedData(view, data.Schema.GetColumnRoleNames());
            return res;
        }

        /// <summary>
        /// Fill dst with values selected from src if the indices of the src values are set in includedIndices,
        /// otherwise assign default(T). The length of dst will be equal to src.Length.
        /// </summary>
        public static void SelectFeatures<T>(ref VBuffer<T> src, BitArray includedIndices, int cardinality, ref VBuffer<T> dst)
        {
            Contracts.Assert(Utils.Size(includedIndices) == src.Length);
            Contracts.Assert(cardinality == Utils.GetCardinality(includedIndices));
            Contracts.Assert(cardinality < src.Length);

            var values = dst.Values;
            var indices = dst.Indices;

            if (src.IsDense)
            {
                if (cardinality >= src.Length / 2)
                {
                    T defaultValue = default;
                    if (Utils.Size(values) < src.Length)
                        values = new T[src.Length];
                    for (int i = 0; i < src.Length; i++)
                        values[i] = !includedIndices[i] ? defaultValue : src.Values[i];
                    dst = new VBuffer<T>(src.Length, values, indices);
                }
                else
                {
                    if (Utils.Size(values) < cardinality)
                        values = new T[cardinality];
                    if (Utils.Size(indices) < cardinality)
                        indices = new int[cardinality];

                    int count = 0;
                    for (int i = 0; i < src.Length; i++)
                    {
                        if (includedIndices[i])
                        {
                            Contracts.Assert(count < cardinality);
                            values[count] = src.Values[i];
                            indices[count] = i;
                            count++;
                        }
                    }

                    Contracts.Assert(count == cardinality);
                    dst = new VBuffer<T>(src.Length, count, values, indices);
                }
            }
            else
            {
                int valuesSize = Utils.Size(values);
                int indicesSize = Utils.Size(indices);
                if (valuesSize < src.Count || indicesSize < src.Count)
                {
                    if (valuesSize < cardinality)
                        values = new T[cardinality];
                    if (indicesSize < cardinality)
                        indices = new int[cardinality];
                }

                int count = 0;
                for (int i = 0; i < src.Count; i++)
                {
                    if (includedIndices[src.Indices[i]])
                    {
                        values[count] = src.Values[i];
                        indices[count] = src.Indices[i];
                        count++;
                    }
                }

                dst = new VBuffer<T>(src.Length, count, values, indices);
            }
        }
    }
}
