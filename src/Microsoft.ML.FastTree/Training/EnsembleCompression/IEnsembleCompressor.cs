// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    internal interface IEnsembleCompressor<TLabel>
    {
        void Initialize(int numTrees, Dataset trainSet, TLabel[] labels, int randomSeed);

        void SetTreeScores(int idx, double[] scores);

        bool Compress(IChannel ch, InternalTreeEnsemble ensemble, double[] trainScores, int bestIteration, int maxTreesAfterCompression);

        InternalTreeEnsemble GetCompressedEnsemble();
    }
}
