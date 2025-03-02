// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.GenAI.Core;

public interface IQuantizeModule
{
    public void Int8();

    /// <summary>
    /// Quantize using BitsAndBytes.FP4
    /// </summary>
    public void FP4();
}
