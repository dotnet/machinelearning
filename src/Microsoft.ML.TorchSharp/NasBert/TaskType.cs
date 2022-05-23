// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TorchSharp.NasBert
{
    public enum TaskType
    {
        None = 0,
        MaskedLM = 1,
        SentenceClassification = 2,
        SentenceRegression = 3
    }
}
