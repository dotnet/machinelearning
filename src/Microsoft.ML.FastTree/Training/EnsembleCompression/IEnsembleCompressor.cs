// -----------------------------------------------------------------------
// <copyright file="FileObjectStore.cs" company="Microsoft">
//     Copyright (C) All Rights Reserved
// </copyright>
// -----------------------------------------------------------------------

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public interface IEnsembleCompressor<TLabel>
    {
        void Initialize(int numTrees, Dataset trainSet, TLabel[] labels, int randomSeed);

        void SetTreeScores(int idx, double[] scores);

        bool Compress(IChannel ch, Ensemble ensemble, double[] trainScores, int bestIteration, int maxTreesAfterCompression);

        Ensemble GetCompressedEnsemble();
    }
}
