// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal interface IDatasetManager
    {
    }

    internal class TrainTestDatasetManager : IDatasetManager
    {
        public IDataView TrainDataset { get; set; }

        public IDataView TestDataset { get; set; }
    }

    internal class CrossValidateDatasetManager : IDatasetManager
    {
        public IDataView Dataset { get; set; }

        public int? Fold { get; set; }
    }
}
