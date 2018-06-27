// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Ensemble
{
    public sealed class Batch
    {
        public readonly RoleMappedData TrainInstances;
        public readonly RoleMappedData TestInstances;

        public Batch(RoleMappedData trainData, RoleMappedData testData)
        {
            Contracts.CheckValue(trainData, nameof(trainData));
            Contracts.CheckValue(testData, nameof(testData));
            TrainInstances = trainData;
            TestInstances = testData;
        }
    }
}
