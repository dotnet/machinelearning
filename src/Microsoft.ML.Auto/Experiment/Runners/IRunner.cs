// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.ML.Auto
{
    internal interface IRunner<TRunDetails> where TRunDetails : RunDetails
    {
        (SuggestedPipelineRunDetails suggestedPipelineRunDetails, TRunDetails runDetails) 
            Run (SuggestedPipeline pipeline, DirectoryInfo modelDirectory, int iterationNum);
    }
}