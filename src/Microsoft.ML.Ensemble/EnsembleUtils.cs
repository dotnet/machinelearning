// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
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
            Contracts.Assert(data.Schema.Feature.HasValue);
            Contracts.AssertValue(features);
            var featCol = data.Schema.Feature.Value;

            var type = featCol.Type;
            var typeVectorSize = type.GetVectorSize();
            Contracts.Assert(features.Length == typeVectorSize);
            int card = Utils.GetCardinality(features);
            if (card == typeVectorSize)
                return data;

            // REVIEW: This doesn't preserve metadata on the features column. Should it?
            var name = featCol.Name;
            var view = LambdaColumnMapper.Create(
                host, "FeatureSelector", data.Data, name, name, type, type,
                (in VBuffer<Single> src, ref VBuffer<Single> dst) => SelectFeatures(in src, features, card, ref dst));

            var res = new RoleMappedData(view, data.Schema.GetColumnRoleNames());
            return res;
        }

        /// <summary>
        /// Fill dst with values selected from src if the indices of the src values are set in includedIndices,
        /// otherwise assign default(T). The length of dst will be equal to src.Length.
        /// </summary>
        public static void SelectFeatures<T>(in VBuffer<T> src, BitArray includedIndices, int cardinality, ref VBuffer<T> dst)
        {
            Contracts.Assert(Utils.Size(includedIndices) == src.Length);
            Contracts.Assert(cardinality == Utils.GetCardinality(includedIndices));
            Contracts.Assert(cardinality < src.Length);

            var srcValues = src.GetValues();
            if (src.IsDense)
            {
                if (cardinality >= src.Length / 2)
                {
                    T defaultValue = default;
                    var editor = VBufferEditor.Create(ref dst, src.Length);
                    for (int i = 0; i < srcValues.Length; i++)
                        editor.Values[i] = !includedIndices[i] ? defaultValue : srcValues[i];
                    dst = editor.Commit();
                }
                else
                {
                    var editor = VBufferEditor.Create(ref dst, src.Length, cardinality);

                    int count = 0;
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        if (includedIndices[i])
                        {
                            Contracts.Assert(count < cardinality);
                            editor.Values[count] = srcValues[i];
                            editor.Indices[count] = i;
                            count++;
                        }
                    }

                    Contracts.Assert(count == cardinality);
                    dst = editor.Commit();
                }
            }
            else
            {
                var editor = VBufferEditor.Create(ref dst, src.Length, cardinality);

                int count = 0;
                var srcIndices = src.GetIndices();
                for (int i = 0; i < srcValues.Length; i++)
                {
                    if (includedIndices[srcIndices[i]])
                    {
                        editor.Values[count] = srcValues[i];
                        editor.Indices[count] = srcIndices[i];
                        count++;
                    }
                }

                dst = editor.CommitTruncated(count);
            }
        }
    }
}
