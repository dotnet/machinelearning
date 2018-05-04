// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// A dataset of features.
    /// </summary>
    public sealed class Dataset
    {
        private readonly DatasetSkeleton _datasetSkeleton;
        private readonly FeatureFlockBase[] _flocks;
        // Maps index of the flock, to index of the first feature of that flock.
        private readonly int[] _flockToFirstFeature;
        // Maps index of a feature, to the flock containing that feature. In combination with
        // _flockToFirstFeature can easily recover the feature sub-index within the flock itself.
        private readonly int[] _featureToFlock;

        public UInt32[] DupeIds { get; private set; }

        public enum DupeIdInfo
        {
            NoInformation = 0,
            Unique = 1,
            FormatNotSupported = 1000000,
            Code404 = 1000001
        };

        public const int Version = 3;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dataset"/> class.
        /// </summary>
        /// <param name="datasetSkeleton">The dataset skeleton corresponding to the features</param>
        /// <param name="flocks">An array of feature flocks</param>
        public Dataset(DatasetSkeleton datasetSkeleton, FeatureFlockBase[] flocks)
        {
            Contracts.AssertValue(datasetSkeleton);
            Contracts.AssertValue(flocks);
            Contracts.Assert(flocks.All(f => f.Examples == datasetSkeleton.NumDocs));

            _datasetSkeleton = datasetSkeleton;
            _maxDocsPerQuery = -1;
            _flocks = flocks;

            _flockToFirstFeature = new int[_flocks.Length];
            if (_flocks.Length > 0)
            {
                for (int i = 1; i < _flocks.Length; ++i)
                {
                    Contracts.AssertValue(_flocks[i - 1]);
                    _flockToFirstFeature[i] = _flockToFirstFeature[i - 1] + _flocks[i - 1].Count;
                }
                var lastFlock = _flocks[_flocks.Length - 1];
                Contracts.AssertValue(lastFlock);
                int numFeatures = _flockToFirstFeature[_flockToFirstFeature.Length - 1] + lastFlock.Count;
                Contracts.Assert(numFeatures == _flocks.Sum(f => f.Count));
                _featureToFlock = new int[numFeatures];
                for (int flock = 0; flock < _flockToFirstFeature.Length; ++flock)
                {
                    int min = _flockToFirstFeature[flock];
                    int lim = min + _flocks[flock].Count;
                    for (int feat = min; feat < lim; ++feat)
                        _featureToFlock[feat] = flock;
                }
            }
            else
                _featureToFlock = new int[0];
        }

        /// <summary>
        /// Maps a global feature index, to the index of the particular flock, as well as the
        /// index of the subfeature within that flock.
        /// </summary>
        /// <param name="feature">The index of the feature at the dataset level</param>
        /// <param name="flock">The index of the flock containing this feature</param>
        /// <param name="subfeature">The index of the feature within the flock</param>
        public void MapFeatureToFlockAndSubFeature(int feature, out int flock, out int subfeature)
        {
            Contracts.Assert(0 <= feature && feature < NumFeatures);
            flock = _featureToFlock[feature];
            subfeature = feature - _flockToFirstFeature[flock];
            Contracts.Assert(0 <= flock && flock < NumFlocks);
            Contracts.Assert(0 <= subfeature && subfeature < _flocks[flock].Count);
        }

        /// <summary>
        /// Given a flock index, returns the index of the first feature in this flock.
        /// </summary>
        /// <param name="flock">Index of the flock</param>
        /// <returns>The index of the first feature that belongs to this flock</returns>
        public int FlockToFirstFeature(int flock)
        {
            Contracts.Assert(0 <= flock && flock < NumFlocks);
            return _flockToFirstFeature[flock];
        }

        #region Skeleton, skeleton passthroughs, and skeleton derived quantities
        /// <summary>
        /// Gets the dataset skeleton.
        /// </summary>
        /// <value>The skeleton.</value>
        public DatasetSkeleton Skeleton => _datasetSkeleton;

        /// <summary>
        /// Gets the labels.
        /// </summary>
        /// <value>The labels.</value>
        public short[] Ratings => _datasetSkeleton.Ratings;

        public double[] Targets => _datasetSkeleton.ActualTargets;

        /// <summary>
        /// Gets the boundaries.
        /// </summary>
        /// <value>The boundaries.</value>
        public int[] Boundaries => _datasetSkeleton.Boundaries;

        /// <summary>
        /// Gets the query ids.
        /// </summary>
        /// <value>The query ids.</value>
        public ulong[] QueryIds => _datasetSkeleton.QueryIds;

        /// <summary>
        /// Gets the doc ids.
        /// </summary>
        /// <value>The doc ids.</value>
        public ulong[] DocIds => _datasetSkeleton.DocIds;

        /// <summary>
        /// Gets the max DCG.
        /// </summary>
        /// <value>The max DCG.</value>
        public double[][] MaxDcg => _datasetSkeleton.MaxDcg;

        private int _maxDocsPerQuery;

        /// <summary>
        /// Gets the max number of docs per any query.
        /// </summary>
        /// <value>The max number of docs per any query.</value>
        public int MaxDocsPerQuery
        {
            get
            {
                if (_maxDocsPerQuery < 0)
                {
                    if (NumQueries == 0)
                        _maxDocsPerQuery = 0;
                    else
                        _maxDocsPerQuery = Enumerable.Range(0, NumQueries).Select(NumDocsInQuery).Max();
                }
                return _maxDocsPerQuery;
            }
        }

        /// <summary>
        /// Gets the number of docs in the entire dataset.
        /// </summary>
        /// <value>The number of docs in the entire dataset.</value>
        public int NumDocs
        {
            get { return _datasetSkeleton.NumDocs; }
        }

        /// <summary>
        /// Nums the docs in a given query.
        /// </summary>
        /// <param name="queryIndex">Index of the query.</param>
        /// <returns>the number of docs in the query</returns>
        public int NumDocsInQuery(int queryIndex)
        {
            return _datasetSkeleton.Boundaries[queryIndex + 1] - _datasetSkeleton.Boundaries[queryIndex];
        }

        /// <summary>
        /// Gets the number of queries in the dataset.
        /// </summary>
        /// <value>The number of queries in the dataset.</value>
        public int NumQueries
        {
            get { return _datasetSkeleton.NumQueries; }
        }

        /// <summary>
        /// Returns the document to query
        /// </summary>
        /// <returns>The associated document</returns>
        public int[] DocToQuery
        {
            get { return _datasetSkeleton.DocToQuery; }
        }

        /// <summary>
        /// Returns the query weights object in underlying dataset skeleton
        /// </summary>
        public double[] SampleWeights
        {
            get { return _datasetSkeleton.SampleWeights; }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public long SizeInBytes()
        {
            return _datasetSkeleton.SizeInBytes() + _flocks.Sum(x => (long)x.SizeInBytes());
        }
        #endregion

        /// <summary>
        /// Gets the array of features.
        /// </summary>
        /// <value>The array of features.</value>
        public FeatureFlockBase[] Flocks
        {
            get { return _flocks; }
        }

        /// <summary>
        /// The number of feature flocks.
        /// </summary>
        public int NumFlocks
        {
            get { return _flocks.Length; }
        }

        /// <summary>
        /// The number of features.
        /// </summary>
        public int NumFeatures
        {
            get { return _featureToFlock.Length; }
        }

        public IIntArrayForwardIndexer GetIndexer(int feature)
        {
            Contracts.Assert(0 <= feature && feature < _featureToFlock.Length);
            int flock;
            int subfeature;
            MapFeatureToFlockAndSubFeature(feature, out flock, out subfeature);
            return _flocks[flock].GetIndexer(subfeature);
        }

        /// <summary>
        /// Split a dataset by queries into disjoint parts
        /// </summary>
        /// <param name="fraction">an array of the fractional size of each part, must sum to 1.0</param>
        /// <param name="randomSeed">a seed that deterministically defines the split</param>
        /// <param name="destroyThisDataset">do you want the features of this dataset to be destroyed on-the-fly as the new datasets are created</param>
        /// <returns></returns>
        public Dataset[] Split(double[] fraction, int randomSeed, bool destroyThisDataset)
        {
            int numParts = fraction.Length;
            int[][] assignment;
            DatasetSkeleton[] datasetSkeletonPart = _datasetSkeleton.Split(fraction, randomSeed, out assignment);
            FeatureFlockBase[][] featureParts = Utils.BuildArray(numParts, i => new FeatureFlockBase[NumFlocks]);
            Parallel.For(0, NumFlocks, new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads },
                (int flockIndex) =>
                {
                    SplitThreadWorker(featureParts, flockIndex, assignment, destroyThisDataset);
                });
            // create datasets
            Dataset[] datasets = Enumerable.Range(0, numParts).Select(p => datasetSkeletonPart[p] == null ?
                null : new Dataset(datasetSkeletonPart[p], featureParts[p])).ToArray(numParts);
            // create and return the datasets
            return datasets;
        }

        /// <summary>
        /// Creates a new Dataset, which includes a subset of the docs in this Dataset.
        /// </summary>
        /// <param name="docIndices">A sorted array of doc indices</param>
        /// <param name="destroyThisDataset">Determines if this Dataset is deleted on the fly as the
        /// new one is created (this reduces peak memory)</param>
        public Dataset GetSubDataset(int[] docIndices, bool destroyThisDataset)
        {
#if !NO_STORE
            return GetSubDataset(docIndices, destroyThisDataset, null);
        }

        public Dataset GetSubDataset(int[] docIndices, bool destroyThisDataset, FileObjectStore<IntArrayFormatter> newBinsCache)
        {
#endif
            int[] queryIndices = docIndices.Select(d => DocToQuery[d]).ToArray();
            ulong[] uniqueQueryIds = queryIndices.Distinct().Select(q => QueryIds[q]).ToArray();

            // calculate boundaries
            int[] boundaries = new int[uniqueQueryIds.Length + 1];
            boundaries[0] = 0;
            int queryIndex = 1;
            for (int q = 1; q < queryIndices.Length; ++q)
            {
                if (queryIndices[q] != queryIndices[q - 1])
                    boundaries[queryIndex++] = q;
            }
            boundaries[uniqueQueryIds.Length] = queryIndices.Length;

            // construct skeleton
            DatasetSkeleton datasetSkeleton = new DatasetSkeleton(docIndices.Select(d => Ratings[d]).ToArray(),
                boundaries,
                uniqueQueryIds,
                docIndices.Select(d => DocIds[d]).ToArray());

            // create features
            FeatureFlockBase[] features = new FeatureFlockBase[NumFlocks];
            int[][] assignment = new int[][] { docIndices };
            Parallel.For(0, NumFlocks, new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads },
                (int flockIndex) =>
                {
#if !NO_STORE
                    GetSubDataset_ThreadWorker(features, flockIndex, assignment, destroyThisDataset,newBinsCache);
#else
                    GetSubDatasetThreadWorker(features, flockIndex, assignment, destroyThisDataset);
#endif
                });

            uint[] filteredDupeIds = null;

            // Filter the dupe ids, if any
            if (DupeIds != null)
            {
                uint[] dupeIds = DupeIds;
                filteredDupeIds = docIndices.Select(i => dupeIds[i]).ToArray();
            }

            // auxiliary data
            Dictionary<string, DatasetSkeletonQueryDocData> auxData = _datasetSkeleton.AuxiliaryData;
            Dictionary<string, DatasetSkeletonQueryDocData> newAuxData = new Dictionary<string, DatasetSkeletonQueryDocData>();
            foreach (KeyValuePair<string, DatasetSkeletonQueryDocData> pair in auxData)
            {
                newAuxData[pair.Key] = pair.Value.GetSubset(pair.Value.IsQueryLevel ? queryIndices.Distinct().ToArray() : docIndices);
            }
            datasetSkeleton.AuxiliaryData = newAuxData;

            // create new Dataset
            Dataset dataset = new Dataset(datasetSkeleton, features);
            dataset.DupeIds = filteredDupeIds;
            return dataset;
        }

#if !NO_STORE
        private void GetSubDataset_ThreadWorker(DerivedFeature[] features, int f, int[][] docAssignment, bool destroyThisDataset, FileObjectStore<IntArrayFormatter> newBinsCache)
        {
            features[f] = Features[f].Split(docAssignment)[0];
            features[f].BinsCache = newBinsCache;

            if (newBinsCache != null)
            {
                features[f].Bins = null;
            }

            if (destroyThisDataset)
                Features[f] = null;
        }
#else
        private void GetSubDatasetThreadWorker(FeatureFlockBase[] features, int f, int[][] docAssignment, bool destroyThisDataset)
        {
            features[f] = Flocks[f].Split(docAssignment)[0];
            if (destroyThisDataset)
                Flocks[f] = null;
        }
#endif

        private void SplitThreadWorker(FeatureFlockBase[][] features, int f, int[][] docAssignment, bool destroyThisDataset)
        {
            FeatureFlockBase[] featureParts = Flocks[f].Split(docAssignment);
            for (int i = 0; i < docAssignment.Length; ++i)
                features[i][f] = featureParts[i];
            if (destroyThisDataset)
                Flocks[f] = null;
        }

        /// <summary>
        /// Returns a row-wise forward indexer across multiple features in the dataset.
        /// </summary>
        /// <param name="activeFeatures">Boolean array indicating active features, or null to
        /// indicate all features should be used</param>
        /// <returns>Row forward indexer</returns>
        public RowForwardIndexer GetFeatureBinRowwiseIndexer(bool[] activeFeatures = null)
        {
            return new RowForwardIndexer(this, activeFeatures);
        }

        public struct DatasetSkeletonQueryDocData
        {
            public bool IsQueryLevel; // Either query or document level.
            public Array Data;

            public DatasetSkeletonQueryDocData GetSubset(int[] docArray)
            {
                DatasetSkeletonQueryDocData qdd = new DatasetSkeletonQueryDocData();

                qdd.IsQueryLevel = IsQueryLevel;

                Type arrayDataType = Data.GetType().GetElementType();
                qdd.Data = Array.CreateInstance(arrayDataType, docArray.Length);
                for (int i = 0; i < docArray.Length; ++i)
                    qdd.Data.SetValue(Data.GetValue(docArray[i]), i);

                return qdd;
            }
        }

        /// <summary>
        /// A class that contains all of the feature-independent data of the dataset
        /// </summary>
        public sealed class DatasetSkeleton
        {
            private short[] _ratings;
            public readonly int[] Boundaries;
            public readonly ulong[] QueryIds;
            public readonly ulong[] DocIds;
            public double[][] MaxDcg;
            private int[] _docToQuery;

            public Dictionary<string, DatasetSkeletonQueryDocData> AuxiliaryData { get; set; }

            /// <summary>
            /// Initializes a new instance of the <see cref="DatasetSkeleton"/> class.
            /// </summary>
            /// <param name="ratings"></param>
            /// <param name="boundaries">The boundaries.</param>
            /// <param name="queryIds">The query ids.</param>
            /// <param name="docIds">The doc ids.</param>
            /// <param name="actualTargets"></param>
            public DatasetSkeleton(short[] ratings, int[] boundaries, ulong[] queryIds, ulong[] docIds, double[] actualTargets = null) :
                this(ratings, boundaries, queryIds, docIds, MaxDcgRange(ratings, boundaries, 10), actualTargets)
            { }

            /// <summary>
            /// Initializes a new instance of the <see cref="DatasetSkeleton"/> class.
            /// </summary>
            /// <param name="ratings">The ratings.</param>
            /// <param name="boundaries">The boundaries.</param>
            /// <param name="queryIds">The query ids.</param>
            /// <param name="docIds">The doc ids.</param>
            /// <param name="maxDcg">The vector of maxDCG.</param>
            /// <param name="actualTargets"></param>
            public DatasetSkeleton(short[] ratings, int[] boundaries, ulong[] queryIds, ulong[] docIds, double[][] maxDcg, double[] actualTargets = null)
            {
                AuxiliaryData = new Dictionary<string, DatasetSkeletonQueryDocData>();
                _ratings = ratings;
                if (actualTargets != null)
                    ActualTargets = actualTargets;
                else
                {
                    ActualTargets = new double[_ratings.Length];
                    for (int i = 0; i < ActualTargets.Length; i++)
                        ActualTargets[i] = (double)_ratings[i];
                }

                Boundaries = boundaries;
                QueryIds = queryIds;
                DocIds = docIds;
                MaxDcg = maxDcg;

                // check that the arguments are consistent
                CheckConsistency();

                // create docToQuery
                _docToQuery = new int[docIds.Length];
                for (int q = 0; q < queryIds.Length; ++q)
                {
                    for (int d = boundaries[q]; d < boundaries[q + 1]; ++d)
                    {
                        _docToQuery[d] = q;
                    }
                }
            }

            public DatasetSkeleton(byte[] buffer, ref int position)
            {
                AuxiliaryData = new Dictionary<string, DatasetSkeletonQueryDocData>();
                using (Timer.Time(TimerEvent.ConstructFromByteArray))
                {
                    _ratings = buffer.ToShortArray(ref position);
                    Boundaries = buffer.ToIntArray(ref position);
                    QueryIds = buffer.ToULongArray(ref position);
                    DocIds = buffer.ToULongArray(ref position);
                    MaxDcg = buffer.ToDoubleJaggedArray(ref position);
                    _docToQuery = buffer.ToIntArray(ref position);
                }
            }

            /// <summary>
            /// Checks the consistency of the DatasetSkeleton
            /// </summary>
            private void CheckConsistency()
            {
                Contracts.Check(Ratings != null && Boundaries != null && QueryIds != null && DocIds != null && MaxDcg != null,
                    "DatasetSkeleton is missing a critical field");

                Contracts.Check(Ratings.Length == DocIds.Length, "Length of label array does not match length of docID array");
                Contracts.Check(Boundaries.Length == QueryIds.Length + 1, "Length of boundaries array does not match length of queryID array");
                Contracts.Check(Utils.Size(MaxDcg) == 0 || Utils.Size(MaxDcg[0]) == QueryIds.Length, "Length of MaxDCG does not match number of queries");
            }

            public double[] ActualTargets
            {
                get;
                private set;
            }

            public short[] Ratings
            {
                get { return _ratings; }
            }

            public int[] DocToQuery
            {
                get { return _docToQuery; }
            }

            public int NumDocs
            {
                get { return DocIds.Length; }
            }

            public int NumQueries
            {
                get { return QueryIds.Length; }
            }

            /// <summary>
            /// Returns the number of bytes written by the member ToByteArray()
            /// </summary>
            public int SizeInBytes()
            {
                return Ratings.SizeInBytes()
                    + Boundaries.SizeInBytes()
                    + QueryIds.SizeInBytes()
                    + DocIds.SizeInBytes()
                    + MaxDcg.SizeInBytes()
                    + DocToQuery.SizeInBytes();
            }

            /// <summary>
            /// Writes a binary representation of this class to a byte buffer, at a given position.
            /// The position is incremented to the end of the representation
            /// </summary>
            /// <param name="buffer">a byte array where the binary represenaion is written</param>
            /// <param name="position">the position in the byte array</param>
            public void ToByteArray(byte[] buffer, ref int position)
            {
                Ratings.ToByteArray(buffer, ref position);
                Boundaries.ToByteArray(buffer, ref position);
                QueryIds.ToByteArray(buffer, ref position);
                DocIds.ToByteArray(buffer, ref position);
                MaxDcg.ToByteArray(buffer, ref position);
                DocToQuery.ToByteArray(buffer, ref position);
            }

            public byte[] ToByteArray()
            {
                int position = 0;
                byte[] buffer = new byte[SizeInBytes()];
                ToByteArray(buffer, ref position);
                return buffer;
            }

            public int[][] GetAssignments(double[] fraction, int randomSeed, out int[][] assignment)
            {
                // make sure fractions sum to 1.0
                if (Math.Abs(fraction.Sum() - 1.0) > 1e-6)
                    throw Contracts.Except("In Dataset.Split(), fractions must sum to 1.0");

                // create a deterministic random number generator
                Random rnd = new Random(randomSeed);

                // get the number of parts and the number of queries in each part
                int numParts = fraction.Length;
                int[][] queries = null;
                if (randomSeed >= 0)
                {
                    int[] numQueries = fraction.Select(x => (int)(x * NumQueries)).ToArray(numParts);
                    numQueries[0] += NumQueries - numQueries.Sum();

                    // get a set of queries in each part
                    int[] perm = Utils.GetRandomPermutation(rnd, NumQueries);
                    queries = numQueries.Select(q => new int[q]).ToArray(numParts);
                    int posInPerm = 0;
                    for (int p = 0; p < numParts; ++p)
                    {
                        // skip empty parts
                        if (numQueries[p] == 0)
                            continue;
                        Array.Copy(perm, posInPerm, queries[p], 0, numQueries[p]);
                        Array.Sort(queries[p]);
                        posInPerm += numQueries[p];
                    }
                }
                else
                {
                    // With negative random seeds, we do query-id dependent sampling.
                    PseudorandomFunction func = new PseudorandomFunction(rnd);
                    int[] thresh = new int[numParts];
                    int val;
                    int p;
                    double cumulative = 0.0;
                    for (int i = 0; i < numParts; ++i)
                    {
                        cumulative += fraction[i];
                        thresh[i] = (int)(cumulative * Int32.MaxValue);
                        if (fraction[i] == 0.0)
                            thresh[i]--;
                    }
                    List<int>[] listQueries = Enumerable.Range(0, numParts).Select(x => new List<int>()).ToArray(numParts);

                    for (int q = 0; q < NumQueries; ++q)
                    {
                        val = func.Apply(QueryIds[q]);
                        for (p = 0; p < numParts && val > thresh[p]; ++p)
                            ;
                        listQueries[p].Add(q);
                    }
                    queries = listQueries.Select(x => x.ToArray()).ToArray(numParts);
                }

                // get the set of docs in each part
                assignment = Enumerable.Range(0, numParts).Select(
                    p => queries[p].SelectMany(q => Enumerable.Range(Boundaries[q], Boundaries[q + 1] - Boundaries[q])).ToArray()
                    ).ToArray(numParts);

                return queries;
            }

            public DatasetSkeleton[] Split(double[] fraction, int randomSeed, out int[][] assignment)
            {
                int[][] queries = GetAssignments(fraction, randomSeed, out assignment);
                int numParts = queries.Length;

                // get boundaries
                int[][] boundaries = queries.Select(q => new int[q.Length + 1]).ToArray(numParts);
                for (int p = 0; p < numParts; ++p)
                {
                    boundaries[p][0] = 0;
                    for (int q = 0; q < queries[p].Length; ++q)
                    {
                        boundaries[p][q + 1] = boundaries[p][q] + Boundaries[queries[p][q] + 1] - Boundaries[queries[p][q]];
                    }
                }

                // get docIds, queryIds, and labels
                short[][] ratings = new short[numParts][];
                ulong[][] queryIds = new ulong[numParts][];
                ulong[][] docIds = new ulong[numParts][];
                for (int p = 0; p < numParts; ++p)
                {
                    ratings[p] = assignment[p].Select(d => Ratings[d]).ToArray();
                    queryIds[p] = queries[p].Select(q => QueryIds[q]).ToArray();
                    docIds[p] = assignment[p].Select(d => DocIds[d]).ToArray();
                }

                // package everything up in datasetSkeleton objects
                DatasetSkeleton[] datasetSkeleton = Enumerable.Range(0, numParts).Select(
                    p => new DatasetSkeleton(ratings[p],
                                             boundaries[p],
                                             queryIds[p],
                                             docIds[p])).ToArray(numParts);

                // Do the auxiliary data.
                foreach (KeyValuePair<string, DatasetSkeletonQueryDocData> pair in AuxiliaryData)
                {
                    DatasetSkeletonQueryDocData qddata = pair.Value;
                    Type arrayDataType = qddata.Data.GetType().GetElementType();
                    for (int p = 0; p < numParts; ++p)
                    {
                        int[] mapping = (qddata.IsQueryLevel ? queries : assignment)[p];
                        Array newData = Array.CreateInstance(arrayDataType, mapping.Length);
                        for (int i = 0; i < mapping.Length; ++i)
                            newData.SetValue(qddata.Data.GetValue(mapping[i]), i);
                        datasetSkeleton[p].SetData(pair.Key, newData, qddata.IsQueryLevel);
                    }
                }

                return datasetSkeleton;
            }

            /// <summary>
            /// Takes an array of DatasetSkeleton objects and concatenates them into one big DatasetSkeleton
            /// </summary>
            /// <param name="parts">An array of DatasetSkeletons</param>
            /// <returns>A concatenated DatasetSkeleton</returns>
            public static DatasetSkeleton Concat(DatasetSkeleton[] parts)
            {
                int concatNumDocs = parts.Sum(x => x.NumDocs);
                int concatNumQueries = parts.Sum(x => x.NumQueries);

                // allocate
                short[] concatRatings = new short[concatNumDocs];
                ulong[] concatDocIds = new ulong[concatNumDocs];
                ulong[] concatQueryIds = new ulong[concatNumQueries];
                int[] concatBoundaries = new int[concatNumQueries + 1];

                // copy components into new arrays
                int docBegin = 0;
                int queryBegin = 0;
                for (int p = 0; p < parts.Length; ++p)
                {
                    int numDocs = parts[p].NumDocs;
                    int numQueries = parts[p].NumQueries;
                    Array.Copy(parts[p].Ratings, 0, concatRatings, docBegin, numDocs);
                    Array.Copy(parts[p].DocIds, 0, concatDocIds, docBegin, numDocs);
                    Array.Copy(parts[p].QueryIds, 0, concatQueryIds, queryBegin, numQueries);
                    for (int q = 0; q < numQueries; ++q)
                        concatBoundaries[queryBegin + q] = parts[p].Boundaries[q] + docBegin;
                    docBegin += numDocs;
                    queryBegin += numQueries;
                }
                concatBoundaries[queryBegin] = docBegin;

                DatasetSkeleton skel = new DatasetSkeleton(concatRatings, concatBoundaries, concatQueryIds, concatDocIds);
                SetConcatenatedAuxiliaryData(parts, skel);
                return skel;
            }

            private static double[] _labelMap = new double[] { 0.0, 3.0, 7.0, 15.0, 31.0 };
            private static readonly double[] _discountMap = new double[] { 1.44269504, 0.91023922, 0.72134752, 0.62133493, 0.55811062, 0.51389834, 0.48089834, 0.45511961, 0.43429448, 0.41703239, 0.40242960 };

            public static double[] LabelGainMap
            {
                get { return _labelMap; }
                set { _labelMap = value; }
            }

            /// <summary>
            /// Calculates natural-based max DCG at all truncations from 1 to trunc
            /// </summary>
            /// <param name="labels">vector of labels</param>
            /// <param name="boundaries">vector of query boundaries</param>
            /// <param name="trunc">max truncation</param>
            private static double[][] MaxDcgRange(short[] labels, int[] boundaries, int trunc)
            {
                double[][] maxAtN = Enumerable.Range(0, trunc).Select(x => new double[boundaries.Length - 1]).ToArray(trunc);
                int relevancyLevel = _labelMap.Length;
                int[] labelCounts = new int[relevancyLevel];

                for (int q = 0; q < boundaries.Length - 1; ++q)
                {
                    int maxTrunc = Math.Min(trunc, boundaries[q + 1] - boundaries[q]);

                    if (maxTrunc == 0)
                    {
                        for (int t = 0; t < trunc; t++)
                            maxAtN[t][q] = double.NaN;
                        continue;
                    }

                    Array.Clear(labelCounts, 0, relevancyLevel);

                    for (int l = boundaries[q]; l < boundaries[q + 1]; l++)
                    {
                        short label = labels[l];
                        labelCounts[label]++;
                    }

                    int topLabel = relevancyLevel - 1;
                    while (labelCounts[topLabel] == 0)
                        topLabel--;
                    maxAtN[0][q] = _labelMap[topLabel] * _discountMap[0];
                    labelCounts[topLabel]--;
                    for (int t = 1; t < maxTrunc; t++)
                    {
                        while (labelCounts[topLabel] == 0)
                            topLabel--;
                        maxAtN[t][q] = maxAtN[t - 1][q] + _labelMap[topLabel] * _discountMap[t];
                        labelCounts[topLabel]--;
                    }
                    for (int t = maxTrunc; t < trunc; t++)
                    {
                        maxAtN[t][q] = maxAtN[t - 1][q];
                    }
                }

                return maxAtN;
            }

            public void RecomputeMaxDcg(int truncationLevel)
            {
                MaxDcg = null;
                MaxDcg = MaxDcgRange(Ratings, Boundaries, truncationLevel);
            }

            /// <summary>
            /// Given the auxiliary data in a bunch of parts, set the concatenated dataset appropriately.
            /// </summary>
            /// <param name="parts">The individual parts of the dataset</param>
            /// <param name="concat">The concatenated version of this dataset</param>
            private static void SetConcatenatedAuxiliaryData(DatasetSkeleton[] parts, DatasetSkeleton concat)
            {
                // Get the union of all the auxiliary data names.
                Dictionary<string, bool> auxNames = new Dictionary<string, bool>();
                foreach (DatasetSkeleton part in parts)
                {
                    foreach (string name in part.AuxiliaryData.Keys)
                    {
                        auxNames[name] = true;
                    }
                }
                DatasetSkeletonQueryDocData[] partsDatas = new DatasetSkeletonQueryDocData[parts.Length];
                int[] docLengths = parts.Select(x => x.NumDocs).ToArray();
                int[] queryLengths = parts.Select(x => x.NumQueries).ToArray();
                foreach (string name in auxNames.Keys)
                {
                    for (int p = 0; p < parts.Length; ++p)
                    {
                        partsDatas[p] = parts[p].AuxiliaryData.ContainsKey(name) ? parts[p].AuxiliaryData[name] : default(DatasetSkeletonQueryDocData);
                    }
                    bool isQuery = partsDatas.First(pd => pd.Data != null).IsQueryLevel;
                    if (partsDatas.Any(pd => pd.Data != null && pd.IsQueryLevel != isQuery))
                    {
                        throw Contracts.Except("On auxiliary data {0}, disagreement on whether this is query/doc", name);
                    }
                    Array concatArray = ConcatArrays(partsDatas.Select(pd => pd.Data).ToArray(), isQuery ? queryLengths : docLengths, name);
                    concat.SetData(name, concatArray, isQuery);
                }
            }

            private static Array ConcatArrays(Array[] arrays, int[] lengths, string name)
            {
                // If all arrays are null (or there are no arrays), then the concat vector is null.
                if (arrays.All(x => x == null))
                    return null;
                // What is the total length?
                int newLength = lengths.Sum();
                // What is the type of these?
                Type t = arrays.First(x => x != null).GetType().GetElementType();

                if (arrays.Any(x => x != null && t != x.GetType().GetElementType()))
                {
                    IEnumerable<string> typeNameEnumerable = arrays.Select(x => x.GetType().GetElementType()).Distinct().Select(x => x.Name).OrderBy(n => n);
                    throw Contracts.Except("When combining auxiliary data, the types of elements must match. Distinct types {0} detected for data named {1}",
                        String.Join(", ", typeNameEnumerable), name);
                }
                Array a = Array.CreateInstance(t, newLength);
                int start = 0;
                for (int i = 0; i < lengths.Length; ++i)
                {
                    if (arrays[i] != null)
                        Array.Copy(arrays[i], 0, a, start, lengths[i]);
                    start += lengths[i];
                }
                return a;
            }

            /// <summary>
            /// Sets some named query or document level auxiliary data.
            /// </summary>
            /// <param name="name">The name of the parameter</param>
            /// <param name="array"></param>
            /// <param name="queryLevel"></param>
            public void SetData(string name, Array array, bool queryLevel)
            {
                int shouldHaveLength = queryLevel ? NumQueries : NumDocs;
                if (array.Length != shouldHaveLength)
                {
                    throw Contracts.Except(
                        "Input array for {0} had {1} elements, ought to have {2}",
                        name, array.Length, shouldHaveLength);
                }
                DatasetSkeletonQueryDocData dd;
                dd.Data = array;
                dd.IsQueryLevel = queryLevel;
                AuxiliaryData[name] = dd;
            }

            /// <summary>
            /// Retrieves some auxiliary data previously set to this skeleton.
            /// </summary>
            /// <typeparam name="T">The type of the array, which should match the type passed in</typeparam>
            public T[] GetData<T>(string name)
            {
                if (!AuxiliaryData.ContainsKey(name))
                    return null;
                return (T[])AuxiliaryData[name].Data;
            }

            private static string SampleWeightsSetName { get { return "SampleWeights"; } }
            public double[] SampleWeights
            {
                get { return GetData<double>(SampleWeightsSetName); }
                set
                {
                    if (value == null)
                    {
                        if (AuxiliaryData.ContainsKey(SampleWeightsSetName))
                        {
                            AuxiliaryData.Remove(SampleWeightsSetName);
                        }
                        return;
                    }
                    SetData(SampleWeightsSetName, value, false);
                }
            }
        }

        /// <summary>
        /// Structure allowing forward indexing by row, across multiple features in the dataset.
        /// </summary>
        public sealed class RowForwardIndexer
        {
            private readonly Dataset _dataset;
            private readonly FeatureFlockBase.FlockForwardIndexerBase[] _flockIndexers;

            public struct Row
            {
                private readonly RowForwardIndexer _indexer;
                private readonly int _rowIndex;

                /// <summary>
                /// Indexes the value of a feature for this row.
                /// </summary>
                /// <param name="featureIndex">The feature index</param>
                /// <returns>The binned valued of a feature for this row</returns>
                public int this[int featureIndex]
                {
                    get
                    {
                        int flock;
                        int subfeature;
                        _indexer._dataset.MapFeatureToFlockAndSubFeature(featureIndex, out flock, out subfeature);
                        Contracts.AssertValue(_indexer._flockIndexers[flock]);
                        return _indexer._flockIndexers[flock][subfeature, _rowIndex];
                    }
                }

                public Row(RowForwardIndexer indexer, int rowIndex)
                {
                    Contracts.AssertValue(indexer);
                    Contracts.Assert(0 <= rowIndex && rowIndex < indexer._dataset.NumDocs);
                    _indexer = indexer;
                    _rowIndex = rowIndex;
                }
            }

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="dataset">The dataset to create the indexer over</param>
            /// <param name="active">Either null to indicate all columns should be active, or
            /// a boolean array of length equal to the number of features that should be active</param>
            public RowForwardIndexer(Dataset dataset, bool[] active = null)
            {
                Contracts.AssertValue(dataset);
                Contracts.Assert(active == null || active.Length == dataset.NumFeatures);

                _dataset = dataset;
                if (active == null)
                    _flockIndexers = _dataset._flocks.Select(d => d.GetFlockIndexer()).ToArray(_dataset.NumFlocks);
                else
                {
                    // We have an actives array.
                    _flockIndexers = new FeatureFlockBase.FlockForwardIndexerBase[_dataset.NumFlocks];
                    for (int iflock = 0; iflock < _dataset.NumFlocks; ++iflock)
                    {
                        var flock = _dataset._flocks[iflock];
                        int offset = _dataset._flockToFirstFeature[iflock];
                        for (int i = 0; i < flock.Count; ++i)
                        {
                            if (active[i + offset])
                            {
                                _flockIndexers[iflock] = flock.GetFlockIndexer();
                                break;
                            }
                        }
                    }
                    // This assert uses a slower but more intuitive test to verify the correctness of the above code.
                    Contracts.Assert(Enumerable.Range(0, _dataset.NumFlocks).All(f =>
                        Enumerable.Range(_dataset._flockToFirstFeature[f], _dataset._flocks[f].Count).Any(i => active[i]) ==
                        (_flockIndexers[f] != null)));
                }
            }

            public Row this[int row] { get { return new Row(this, row); } }
        }
    }
}
