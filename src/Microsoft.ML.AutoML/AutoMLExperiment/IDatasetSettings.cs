// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    internal interface IDatasetSettings
    {
    }

    internal class TrainTestDatasetSettings : IDatasetSettings
    {
        public IDataView TrainDataset { get; set; }

        public IDataView TestDataset { get; set; }
    }

    internal class CrossValidateDatasetSettings : IDatasetSettings
    {
        public IDataView Dataset { get; set; }

        public int? Fold { get; set; }
    }
}
