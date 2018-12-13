﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.Ensemble.Selector
{
    internal interface ISubsetSelector
    {
        void Initialize(RoleMappedData data, int size, int batchSize, Single validationDatasetProportion);
        IEnumerable<Batch> GetBatches(Random rand);
        IEnumerable<Subset> GetSubsets(Batch batch, Random rand);
        RoleMappedData GetTestData(Subset subset, Batch batch);
    }

    public delegate void SignatureEnsembleDataSelector();

    [TlcModule.ComponentKind("EnsembleSubsetSelector")]
    internal interface ISupportSubsetSelectorFactory : IComponentFactory<ISubsetSelector>
    {
    }
}
