// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    public sealed class DocumentPartitioning
    {
        private readonly int[] _leafBegin;
        private readonly int[] _leafCount;
        private readonly int[] _documents;
        private int[] _tempDocuments;
        private int[] _initialDocuments;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numDocuments">number of document</param>
        /// <param name="maxLeaves">number of leaves</param>
        public DocumentPartitioning(int numDocuments, int maxLeaves)
        {
            Contracts.Assert(numDocuments >= 0);
            Contracts.Assert(maxLeaves > 0);

            _leafBegin = new int[maxLeaves];
            _leafCount = new int[maxLeaves];
            _documents = new int[numDocuments];
        }

        public DocumentPartitioning(int[] documents, int numDocuments, int maxLeaves)
            : this(numDocuments, maxLeaves)
        {
            _initialDocuments = new int[numDocuments];
            for (int d = 0; d < numDocuments; ++d)
                _initialDocuments[d] = documents[d];
        }

        /// <summary>
        /// Constructs partitioning object based on the documents and RegressionTree splits
        /// NOTE: It has been optimized for speed and multiprocs with 10x gain on naive LINQ implementation
        /// </summary>
        public DocumentPartitioning(RegressionTree tree, Dataset dataset)
            : this(dataset.NumDocs, tree.NumLeaves)
        {
            using (Timer.Time(TimerEvent.DocumentPartitioningConstruction))
            {
                // figure out which leaf each document belongs to
                // NOTE: break it up into NumThreads chunks. This minimizes the number of re-computations necessary in
                // the row-wise indexer.
                int innerLoopSize = 1 + dataset.NumDocs / BlockingThreadPool.NumThreads; // +1 is to make sure we don't have a few left over at the end

                // figure out the exact number of chunks, needed in pathological cases when NumDocs < NumThreads
                int numChunks = dataset.NumDocs / innerLoopSize;
                if (dataset.NumDocs % innerLoopSize != 0)
                    ++numChunks;
                var perChunkDocumentLists = new List<int>[numChunks][];
                // REVIEW: This partitioning doesn't look optimal.
                // Probably make sence to investigate better ways of splitting data?
                var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumDocs / innerLoopSize)];
                var actionIndex = 0;
                for (int docStart = 0; docStart < dataset.NumDocs; docStart += innerLoopSize)
                {
                    var fromDoc = docStart;
                    var toDoc = Math.Min(docStart + innerLoopSize, dataset.NumDocs);
                    var chunkIndex = docStart / innerLoopSize;
                    actions[actionIndex++] = () =>
                    {
                        Contracts.Assert(perChunkDocumentLists[chunkIndex] == null);

                        var featureBins = dataset.GetFeatureBinRowwiseIndexer();

                        List<int>[] perLeafDocumentLists = Enumerable.Range(0, tree.NumLeaves)
                            .Select(x => new List<int>(innerLoopSize / tree.NumLeaves))
                            .ToArray();

                        for (int d = fromDoc; d < toDoc; d++)
                        {
                            int leaf = tree.GetLeaf(featureBins[d]);
                            perLeafDocumentLists[leaf].Add(d);
                        }

                        perChunkDocumentLists[chunkIndex] = perLeafDocumentLists;
                    };
                }
                Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);

                // establish leaf starts and document counts
                _leafCount = Enumerable.Range(0, tree.NumLeaves)
                    .Select(leaf => Enumerable.Range(0, perChunkDocumentLists.Length)
                        .Select(thread => perChunkDocumentLists[thread][leaf].Count)
                        .Sum())
                    .ToArray();

                var cumulativeLength = _leafCount.CumulativeSum<int>().Take(tree.NumLeaves - 1);
                _leafBegin = Enumerable.Range(0, 1).Concat(cumulativeLength).ToArray();

                // move all documents that belong to the same leaf together
                Contracts.Assert(_documents.Length == _leafBegin[tree.NumLeaves - 1] + _leafCount[tree.NumLeaves - 1]);
                actions = new Action[tree.NumLeaves];
                actionIndex = 0;
                for (int leaf = 0; leaf < tree.NumLeaves; leaf++)
                {
                    var l = leaf;
                    actions[actionIndex++] = () =>
                    {
                        int documentPos = _leafBegin[l];
                        for (int chunkIndex = 0; chunkIndex < perChunkDocumentLists.Length; chunkIndex++)
                        {
                            foreach (int d in perChunkDocumentLists[chunkIndex][l])
                            {
                                _documents[documentPos++] = d;
                            }
                            perChunkDocumentLists[chunkIndex][l] = null;
                        }
                    };
                }
                Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            }
        }

        /// <summary>
        /// Returns the total number of documents handled by the partitioning
        /// </summary>
        public int NumDocs
        {
            get { return _documents.Length; }
        }

        public int[] Documents
        {
            get { return _documents; }
        }

        /// <summary>
        /// Resets the data structure, as if it was newly created
        /// </summary>
        public void Initialize()
        {
            Array.Clear(_leafCount, 0, _leafCount.Length);
            _leafBegin[0] = 0;
            _leafCount[0] = _documents.Length;
            if (_initialDocuments == null)
            {
                for (int d = 0; d < _documents.Length; ++d)
                    _documents[d] = d;
            }
            else
            {
                for (int d = 0; d < _documents.Length; ++d)
                    _documents[d] = _initialDocuments[d];
            }
        }

        /// <summary>
        /// Does sampling with replacement on each leaf node and returns leaf count times of sample labels.
        /// </summary>
        public double[] GetDistribution(double[] targets, double[] weights, int quantileSampleCount, Random rand, int leafCount, out double[] distributionWeights)
        {
            double[] dist = new double[leafCount * quantileSampleCount];
            if (weights == null)
                distributionWeights = null;
            else
                distributionWeights = new double[leafCount * quantileSampleCount];

            int count = 0;
            for (int i = 0; i < leafCount; i++)
            {
                for (int j = 0; j < quantileSampleCount; j++)
                {
                    var randInst = _leafBegin[i] + rand.Next(_leafCount[i]);
                    if (distributionWeights != null)
                        distributionWeights[count] = weights[_documents[randInst]];

                    dist[count++] = targets[_documents[randInst]];
                }
            }

            return dist;
        }

        /// <summary>
        /// Splits the documents of a specified leaf to its two children based on a feature and a threshold value
        /// </summary>
        /// <param name="leaf">the leaf being split</param>
        /// <param name="indexer"></param>
        /// <param name="threshold">the threshold</param>
        /// <param name="gtChildIndex">Index of child node that contains documents whose split 
        /// feature value is greater than the split threshold</param>
        public unsafe void Split(int leaf, IIntArrayForwardIndexer indexer, UInt32 threshold, int gtChildIndex)
        {
            using (Timer.Time(TimerEvent.DocumentPartitioningSplit))
            {
                if (_tempDocuments == null)
                    _tempDocuments = new int[_documents.Length];

                // Note: lteChildIndex = leaf
                int begin = _leafBegin[leaf];
                int end = begin + _leafCount[leaf];
                int newEnd = begin;
                int tempEnd = begin;

                fixed (int* pDocuments = _documents)
                fixed (int* pTempDocuments = _tempDocuments)
                {
                    for (int curr = begin; curr < end; ++curr)
                    {
                        int doc = pDocuments[curr];
                        if (indexer[doc] > threshold)
                            pTempDocuments[tempEnd++] = doc;
                        else
                            pDocuments[newEnd++] = doc;
                    }
                }

                int newCount = newEnd - begin;
                int gtCount = tempEnd - begin;
                Array.Copy(_tempDocuments, begin, _documents, newEnd, gtCount);

                _leafCount[leaf] = newCount;
                _leafBegin[gtChildIndex] = newEnd;
                _leafCount[gtChildIndex] = gtCount;
            }
        }

        /// <summary>
        /// Splits the documents of a specified leaf to its two children based on a feature and a threshold value
        /// </summary>
        /// <param name="leaf">the leaf being split</param>
        /// <param name="bins">Split feature flock's bin</param>
        /// <param name="categoricalIndices">Catgeorical feature indices</param>
        /// <param name="gtChildIndex">Index of child node that contains documents whose split 
        /// feature value is greater than the split threshold</param>
        public unsafe void Split(int leaf, IntArray bins, HashSet<int> categoricalIndices, int gtChildIndex)
        {
            Contracts.Assert(bins != null);

            using (Timer.Time(TimerEvent.DocumentPartitioningSplit))
            {
                if (_tempDocuments == null)
                    _tempDocuments = new int[_documents.Length];

                // Note: lteChildIndex = leaf
                int begin = _leafBegin[leaf];
                int end = begin + _leafCount[leaf];
                int newEnd = begin;
                int tempEnd = begin;
                var flockBinIndex = bins.GetIndexer();
                fixed (int* pDocuments = _documents)
                fixed (int* pTempDocuments = _tempDocuments)
                {
                    for (int curr = begin; curr < end; ++curr)
                    {
                        int doc = pDocuments[curr];
                        int hotFeature = flockBinIndex[doc];

                        if (categoricalIndices.Contains(hotFeature - 1))
                            pTempDocuments[tempEnd++] = doc;
                        else
                            pDocuments[newEnd++] = doc;
                    }
                }

                int newCount = newEnd - begin;
                int gtCount = tempEnd - begin;
                Array.Copy(_tempDocuments, begin, _documents, newEnd, gtCount);

                _leafCount[leaf] = newCount;
                _leafBegin[gtChildIndex] = newEnd;
                _leafCount[gtChildIndex] = gtCount;
            }
        }

        /// <summary>
        /// Get the document partitions of a specified leaf if it is split based on a feature and a threshold value.
        /// </summary>
        /// <param name="leaf">the leaf being split</param>
        /// <param name="indexer">the indexer to access the feature value</param>
        /// <param name="threshold">the threshold</param>
        /// <param name="leftDocuments">[out] the left documents split from the leaf</param>
        /// <param name="leftDocumentSize">[out] the size of left documents</param>
        /// <param name="rightDocuments">[out] the right documents split from the leaf</param>
        /// <param name="rightDocumentSize">[out] the size of right documents</param>
        public unsafe void GetLeafDocumentPartitions(
            int leaf,
            IIntArrayForwardIndexer indexer,
            UInt32 threshold,
            out int[] leftDocuments,
            out int leftDocumentSize,
            out int[] rightDocuments,
            out int rightDocumentSize)
        {
            using (Timer.Time(TimerEvent.DocumentPartitioningSplit))
            {
                leftDocuments = new int[_leafCount[leaf]];
                leftDocumentSize = 0;

                rightDocuments = new int[_leafCount[leaf]];
                rightDocumentSize = 0;

                int begin = _leafBegin[leaf];
                int end = begin + _leafCount[leaf];

                fixed (int* pDocuments = _documents)
                fixed (int* pTempLeftDocIndices = leftDocuments)
                fixed (int* pTempRightDocIndices = rightDocuments)
                {
                    for (int curr = begin; curr < end; ++curr)
                    {
                        int doc = pDocuments[curr];
                        if (indexer[doc] > threshold)
                        {
                            pTempRightDocIndices[rightDocumentSize++] = doc;
                        }
                        else
                        {
                            pTempLeftDocIndices[leftDocumentSize++] = doc;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Returns an enumeration of the document indices associated with a specified leaf, in ascending order
        /// </summary>
        /// <param name="leaf">the leaf index</param>
        /// <returns>the enumeration</returns>
        public IEnumerable<int> DocumentsInLeaf(int leaf)
        {
            int index = _leafBegin[leaf];
            int end = _leafBegin[leaf] + _leafCount[leaf];
            while (index < end)
            {
                yield return _documents[index];
                ++index;
            }
        }

        public int GetLeafDocuments(int leaf, int[] documents)
        {
            Array.Copy(_documents, _leafBegin[leaf], documents, 0, _leafCount[leaf]);
            return _leafCount[leaf];
        }

        public void ReferenceLeafDocuments(int leaf, out int[] documents, out int begin, out int count)
        {
            documents = _documents;
            begin = _leafBegin[leaf];
            count = _leafCount[leaf];
        }

        /// <summary>
        /// How many documents are associated with a specified leaf
        /// </summary>
        /// <param name="leaf">the leaf</param>
        /// <returns>the number of documents</returns>
        public int NumDocsInLeaf(int leaf) { return _leafCount[leaf]; }

        /// <summary>
        /// Calculates the mean of a double array only on the elements that correspond to a specified leaf in the tree
        /// </summary>
        /// <param name="array">the double array</param>
        /// <param name="leaf">the leaf index</param>
        /// <param name="filterZeros"></param>
        /// <returns>the mean</returns>
        public double Mean(double[] array, int leaf, bool filterZeros)
        {
            double mean = 0.0;
            int end = _leafBegin[leaf] + _leafCount[leaf];
            int count = (filterZeros ? 0 : _leafCount[leaf]);
            if (filterZeros)
            {
                double value;
                for (int i = _leafBegin[leaf]; i < end; ++i)
                {
                    value = array[_documents[i]];
                    if (value != 0)
                    {
                        mean += value;
                        count++;
                    }
                }
            }
            else
            {
                for (int i = _leafBegin[leaf]; i < end; ++i)
                    mean += array[_documents[i]];
            }
            return mean / count;
        }

        /// <summary>
        /// Calculates the weighted mean of a double array only on the elements that correspond to a specified leaf in the tree
        /// </summary>
        /// <param name="array">the double array</param>
        /// <param name="sampleWeights">Weights of array elements</param>
        /// <param name="leaf">the leaf index</param>
        /// <param name="filterZeros"></param>
        /// <returns>the mean</returns>
        public double Mean(double[] array, double[] sampleWeights, int leaf, bool filterZeros)
        {
            if (sampleWeights == null)
            {
                return Mean(array, leaf, filterZeros);
            }
            double mean = 0.0;
            int end = _leafBegin[leaf] + _leafCount[leaf];
            double sumWeight = 0;
            if (filterZeros)
            {
                double value;
                for (int i = _leafBegin[leaf]; i < end; ++i)
                {
                    value = array[_documents[i]];
                    if (value != 0)
                    {
                        FloatType weight = (FloatType)sampleWeights[_documents[i]];
                        mean += value * weight;
                        sumWeight += weight;
                    }
                }
            }
            else
            {
                for (int i = _leafBegin[leaf]; i < end; ++i)
                {
                    FloatType weight = (FloatType)sampleWeights[_documents[i]];
                    mean += array[_documents[i]] * weight;
                    sumWeight += weight;
                }
            }
            return mean / sumWeight;
        }

    } //of class DocumentPartitioning
}
