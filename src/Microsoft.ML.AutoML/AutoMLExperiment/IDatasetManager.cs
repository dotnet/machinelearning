// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Interface for dataset manager. This interface doesn't include any method or property definition and is used by <see cref="AutoMLExperiment"/> and other components to retrieve the instance of the actual
    /// dataset manager from containers.
    /// </summary>
    public interface IDatasetManager
    {
    }

    public class TrainTestDatasetManager : IDatasetManager
    {
        public IDataView TrainDataset { get; set; }

        public IDataView TestDataset { get; set; }
    }

    public class CrossValidateDatasetManager : IDatasetManager
    {
        public IDataView Dataset { get; set; }

        public int? Fold { get; set; }
    }
}
