// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(KMeansPlusPlusTrainer.Summary, typeof(KMeansPlusPlusTrainer), typeof(KMeansPlusPlusTrainer.Arguments),
    new[] { typeof(SignatureClusteringTrainer), typeof(SignatureTrainer) },
    KMeansPlusPlusTrainer.UserNameValue,
    KMeansPlusPlusTrainer.LoadNameValue,
    KMeansPlusPlusTrainer.ShortName, "KMeans")]

[assembly: LoadableClass(typeof(void), typeof(KMeansPlusPlusTrainer), null, typeof(SignatureEntryPointModule), "KMeans")]

namespace Microsoft.ML.Runtime.KMeans
{
    /// <include file='./doc.xml' path='doc/members/member[@name="KMeans++"]/*' />
    public class KMeansPlusPlusTrainer : TrainerBase<KMeansPredictor>
    {
        public const string LoadNameValue = "KMeansPlusPlus";
        internal const string UserNameValue = "KMeans++ Clustering";
        internal const string ShortName = "KM";
        internal const string Summary = "K-means is a popular clustering algorithm. With K-means, the data is clustered into a specified "
            + "number of clusters in order to minimize the within-cluster sum of squares. K-means++ improves upon K-means by using a better "
            + "method for choosing the initial cluster centers.";

        public enum InitAlgorithm
        {
            KMeansPlusPlus = 0,
            Random = 1,
            KMeansParallel = 2
        }

        public class Arguments : UnsupervisedLearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of clusters", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "5,10,20,40")]
            [TlcModule.SweepableDiscreteParam("K", new object[] { 5, 10, 20, 40 })]
            public int K = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cluster initialization algorithm", ShortName = "init")]
            public InitAlgorithm InitAlgorithm = InitAlgorithm.KMeansParallel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance parameter for trainer convergence. Lower = slower, more accurate",
                ShortName = "ot")]
            [TGUI(Label = "Optimization Tolerance", Description = "Threshold for trainer convergence")]
            public Float OptTol = (Float)1e-7;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of iterations.", ShortName = "maxiter")]
            [TGUI(Label = "Max Number of Iterations")]
            public int MaxIterations = 1000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Memory budget (in MBs) to use for KMeans acceleration", ShortName = "accelMemBudgetMb")]
            [TGUI(Label = "Memory Budget (in MBs) for KMeans Acceleration")]
            public int AccelMemBudgetMb = 4 * 1024; // by default, use at most 4 GB

            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Defaults to automatic. Determinism not guaranteed.", ShortName = "nt,t,threads", SortOrder = 50)]
            [TGUI(Label = "Number of threads")]
            public int? NumThreads;
        }

        private readonly int _k;

        private readonly int _maxIterations; // max number of iterations to train
        private readonly Float _convergenceThreshold; // convergence thresholds

        private readonly long _accelMemBudgetMb;
        private readonly InitAlgorithm _initAlgorithm;
        private readonly int _numThreads;

        public override TrainerInfo Info { get; }
        public override PredictionKind PredictionKind => PredictionKind.Clustering;

        public KMeansPlusPlusTrainer(IHostEnvironment env, Arguments args)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(args.K > 0, nameof(args.K), "Must be positive");

            _k = args.K;

            Host.CheckUserArg(args.MaxIterations > 0, nameof(args.MaxIterations), "Must be positive");
            _maxIterations = args.MaxIterations;

            Host.CheckUserArg(args.OptTol > 0, nameof(args.OptTol), "Tolerance must be positive");
            _convergenceThreshold = args.OptTol;

            Host.CheckUserArg(args.AccelMemBudgetMb > 0, nameof(args.AccelMemBudgetMb), "Must be positive");
            _accelMemBudgetMb = args.AccelMemBudgetMb;

            _initAlgorithm = args.InitAlgorithm;

            Host.CheckUserArg(!args.NumThreads.HasValue || args.NumThreads > 0, nameof(args.NumThreads),
                "Must be either null or a positive integer.");
            _numThreads = ComputeNumThreads(Host, args.NumThreads);
            Info = new TrainerInfo();
        }

        public override KMeansPredictor Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector(out int dimensionality);
            Contracts.Assert(dimensionality > 0);

            using (var ch = Host.Start("Training"))
            {
                var pred = TrainCore(ch, data, dimensionality);
                ch.Done();
                return pred;
            }
        }

        private KMeansPredictor TrainCore(IChannel ch, RoleMappedData data, int dimensionality)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            // REVIEW: In high-dimensionality cases this is less than ideal and we should consider
            // using sparse buffers for the centroids.

            // The coordinates of the final centroids at the end of the training. During training
            // it holds the centroids of the previous iteration.
            var centroids = new VBuffer<Float>[_k];
            for (int i = 0; i < _k; i++)
                centroids[i] = VBufferUtils.CreateDense<Float>(dimensionality);

            ch.Info("Initializing centroids");
            long missingFeatureCount;
            long totalTrainingInstances;

            var cursorFactory = new FeatureFloatVectorCursor.Factory(data, CursOpt.Features | CursOpt.Id | CursOpt.Weight);
            // REVIEW: It would be nice to extract these out into subcomponents in the future. We should
            // revisit and even consider breaking these all into individual KMeans-flavored trainers, they
            // all produce a valid set of output centroids with various trade-offs in runtime (with perhaps
            // random initialization creating a set that's not terribly useful.) They could also be extended to
            // pay attention to their incoming set of centroids and incrementally train.
            if (_initAlgorithm == InitAlgorithm.KMeansPlusPlus)
            {
                KMeansPlusPlusInit.Initialize(Host, _numThreads, ch, cursorFactory, _k, dimensionality,
                    centroids, out missingFeatureCount, out totalTrainingInstances);
            }
            else if (_initAlgorithm == InitAlgorithm.Random)
            {
                KMeansRandomInit.Initialize(Host, _numThreads, ch, cursorFactory, _k,
                    centroids, out missingFeatureCount, out totalTrainingInstances);
            }
            else
            {
                // Defaulting to KMeans|| initialization.
                KMeansBarBarInitialization.Initialize(Host, _numThreads, ch, cursorFactory, _k, dimensionality,
                    centroids, _accelMemBudgetMb, out missingFeatureCount, out totalTrainingInstances);
            }

            KMeansUtils.VerifyModelConsistency(centroids);
            ch.Info("Centroids initialized, starting main trainer");

            KMeansLloydsYinYangTrain.Train(
                Host, _numThreads, ch, cursorFactory, totalTrainingInstances, _k, dimensionality, _maxIterations,
                _accelMemBudgetMb, _convergenceThreshold, centroids);

            KMeansUtils.VerifyModelConsistency(centroids);
            ch.Info("Model trained successfully on {0} instances", totalTrainingInstances);
            if (missingFeatureCount > 0)
            {
                ch.Warning(
                    "{0} instances with missing features detected and ignored. Consider using MissingHandler.",
                    missingFeatureCount);
            }
            return new KMeansPredictor(Host, _k, centroids, copyIn: true);
        }

        private static int ComputeNumThreads(IHost host, int? argNumThreads)
        {
            // REVIEW: For small data sets it would be nice to clamp down on concurrency, it
            // isn't going to get us a performance improvement.
            int maxThreads;
            if (host.ConcurrencyFactor < 1)
                maxThreads = Environment.ProcessorCount / 2;
            else
                maxThreads = host.ConcurrencyFactor;

            // If we specified a number of threads that's fine, but it must be below the
            // host-set concurrency factor.
            if (argNumThreads.HasValue)
                maxThreads = Math.Min(maxThreads, argNumThreads.Value);

            return Math.Max(1, maxThreads);
        }

        [TlcModule.EntryPoint(Name = "Trainers.KMeansPlusPlusClusterer",
            Desc = Summary,
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.KMeansClustering/doc.xml' path='doc/members/member[@name=""KMeans++""]/*' />",
                                 @"<include file='../Microsoft.ML.KMeansClustering/doc.xml' path='doc/members/example[@name=""KMeans++""]/*' />"})]
        public static CommonOutputs.ClusteringOutput TrainKMeans(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainKMeans");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.ClusteringOutput>(host, input,
                () => new KMeansPlusPlusTrainer(host, input),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }

    internal static class KMeansPlusPlusInit
    {
        private const Float Epsilon = (Float)1e-15;

        /// <summary>
        /// Initialize starting centroids via KMeans++ algorithm. This algorithm will always run single-threaded,
        /// regardless of the value of <paramref name="numThreads" />.
        /// </summary>
        public static void Initialize(
            IHost host, int numThreads, IChannel ch, FeatureFloatVectorCursor.Factory cursorFactory,
            int k, int dimensionality, VBuffer<Float>[] centroids,
            out long missingFeatureCount, out long totalTrainingInstances, bool showWarning = true)
        {
            missingFeatureCount = 0;
            totalTrainingInstances = 0;
            var stopWatch = new System.Diagnostics.Stopwatch();
            stopWatch.Start();
            const int checkIterations = 5;
            Float[] centroidL2s = new Float[k];

            // Need two vbuffers - one for the features of the current row, and one for the features of
            // the "candidate row".
            var candidate = default(VBuffer<Float>);
            using (var pCh = host.StartProgressChannel("KMeansPlusPlusInitialize"))
            {
                int i = 0;
                pCh.SetHeader(new ProgressHeader("centroids"), (e) => e.SetProgress(0, i, k));
                for (i = 0; i < k; i++)
                {
                    // Print a message to user if the anticipated time for initializing is more than an hour.
                    if (i == checkIterations)
                    {
                        stopWatch.Stop();
                        var elapsedHours = stopWatch.Elapsed.TotalHours;
                        var expectedHours = elapsedHours * k * (k - 1) / (checkIterations * (checkIterations - 1));
                        if (expectedHours > 1 && showWarning)
                        {
                            ch.Warning("Expected time to initialize all {0} clusters is {1:0} minutes. "
                                + "You can use init=KMeansParallel to switch to faster initialization.",
                                k, expectedHours * 60);
                        }
                    }

                    // initializing i-th centroid

                    Double cumulativeWeight = 0; // sum of weights accumulated so far
                    Float? cachedCandidateL2 = null;
                    bool haveCandidate = false;
                    // on all iterations except 0's, we calculate L2 norm of every instance and cache the current candidate's
                    using (var cursor = cursorFactory.Create())
                    {
                        while (cursor.MoveNext())
                        {
                            Float probabilityWeight;
                            Float l2 = 0;
                            if (i == 0)
                            {
                                // This check is only performed once, at the first pass of initialization
                                if (dimensionality != cursor.Features.Length)
                                {
                                    throw ch.Except(
                                        "Dimensionality doesn't match, expected {0}, got {1}",
                                        dimensionality,
                                        cursor.Features.Length);
                                }
                                probabilityWeight = 1;
                            }
                            else
                            {
                                l2 = VectorUtils.NormSquared(cursor.Features);
                                probabilityWeight = Float.PositiveInfinity;

                                for (int j = 0; j < i; j++)
                                {
                                    var distance = -2 * VectorUtils.DotProduct(ref cursor.Features, ref centroids[j])
                                        + l2 + centroidL2s[j];
                                    probabilityWeight = Math.Min(probabilityWeight, distance);
                                }

                                ch.Assert(FloatUtils.IsFinite(probabilityWeight));
                            }

                            if (probabilityWeight > 0)
                            {
                                probabilityWeight *= cursor.Weight;

                                // as a result of numerical error, we could get negative distance, do not decrease the sum
                                cumulativeWeight += probabilityWeight;

                                if (probabilityWeight > Epsilon &&
                                    host.Rand.NextSingle() < probabilityWeight / cumulativeWeight)
                                {
                                    // again, numerical error may cause selection of the same candidate twice, so ensure that the distance is non-trivially positive
                                    Utils.Swap(ref cursor.Features, ref candidate);
                                    haveCandidate = true;
                                    if (i > 0)
                                        cachedCandidateL2 = l2;
                                }
                            }
                        }

                        if (i == 0)
                        {
                            totalTrainingInstances = cursor.KeptRowCount;
                            missingFeatureCount = cursor.BadFeaturesRowCount;
                        }
                    }

                    // persist the candidate as a new centroid
                    if (!haveCandidate)
                    {
                        throw ch.Except(
                            "Not enough distinct instances to populate {0} clusters (only found {1} distinct instances)", k, i);
                    }

                    candidate.CopyTo(centroids[i].Values);
                    centroidL2s[i] = cachedCandidateL2 ?? VectorUtils.NormSquared(candidate);
                }
            }
        }
    }

    /// <summary>
    /// An instance of this class is used by SharedStates in YinYangTrainer
    /// and KMeansBarBarInitialization. It effectively bounds MaxInstancesToAccelerate and
    /// initializes RowIndexGetter.
    /// </summary>
    internal sealed class KMeansAcceleratedRowMap
    {
        // Retrieves the row's index for per-instance data. If the
        // row is not assigned an index (it occurred after 'maxInstancesToAccelerate')
        // or we are not accelerating then this returns -1.
        public readonly KMeansUtils.RowIndexGetter RowIndexGetter;

        public readonly bool IsAccelerated;
        public readonly int MaxInstancesToAccelerate;

        // When using parallel row cursors, there is no fixed, stable index for
        // each row. Instead the RowCursor provides a stable ID across multiple
        // cursorings. We map those IDs into an index to poke into the per instance
        // structures.
        private readonly HashArray<UInt128> _parallelIndexLookup;

        public KMeansAcceleratedRowMap(FeatureFloatVectorCursor.Factory factory, IChannel ch,
            long baseMaxInstancesToAccelerate, long totalTrainingInstances, bool isParallel)
        {
            Contracts.AssertValue(factory);
            Contracts.AssertValue(ch);
            Contracts.Assert(totalTrainingInstances > 0);
            Contracts.Assert(baseMaxInstancesToAccelerate >= 0);

            // MaxInstancesToAccelerate is bound by the .Net array size limitation for now.
            if (baseMaxInstancesToAccelerate > Utils.ArrayMaxSize)
                baseMaxInstancesToAccelerate = Utils.ArrayMaxSize;

            if (baseMaxInstancesToAccelerate > totalTrainingInstances)
                baseMaxInstancesToAccelerate = totalTrainingInstances;
            else if (baseMaxInstancesToAccelerate > 0)
                ch.Info("Accelerating the first {0} instances", baseMaxInstancesToAccelerate);

            if (baseMaxInstancesToAccelerate == 0)
                RowIndexGetter = (FeatureFloatVectorCursor cur) => -1;
            else
            {
                MaxInstancesToAccelerate = (int)baseMaxInstancesToAccelerate;
                IsAccelerated = true;
                if (!isParallel)
                    RowIndexGetter = (FeatureFloatVectorCursor cur) => cur.Row.Position < baseMaxInstancesToAccelerate ? (int)cur.Row.Position : -1;
                else
                {
                    _parallelIndexLookup = BuildParallelIndexLookup(factory);
                    RowIndexGetter = (FeatureFloatVectorCursor cur) =>
                    {
                        int idx;
                        // Sets idx to -1 on failure to find.
                        _parallelIndexLookup.TryGetIndex(cur.Id, out idx);
                        return idx;
                    };
                }
            }
        }

        /// <summary>
        /// Initializes the parallel index lookup HashArray using a sequential RowCursor. We
        /// preinitialize the HashArray so we can perform lock-free lookup operations during
        /// the primary KMeans pass.
        /// </summary>
        private HashArray<UInt128> BuildParallelIndexLookup(FeatureFloatVectorCursor.Factory factory)
        {
            Contracts.AssertValue(factory);

            HashArray<UInt128> lookup = new HashArray<UInt128>();
            int n = 0;
            using (var cursor = factory.Create())
            {
                while (cursor.MoveNext() && n < MaxInstancesToAccelerate)
                {
                    lookup.Add(cursor.Id);
                    n++;
                }
            }

            Contracts.Check(lookup.Count == n);
            return lookup;
        }
    }

    internal static class KMeansBarBarInitialization
    {
        /// <summary>
        /// Data for optimizing KMeans|| initialization. Very similar to SharedState class
        /// For every instance, there is a space for the best weight and best cluster computed.
        ///
        /// In this class, new clusters mean the clusters that were added to the cluster set
        /// in the previous round of KMeans|| and old clusters are the rest of them (the ones
        /// that were added in the rounds before the previous one).
        ///
        /// In every round of KMeans||, numSamplesPerRound new clusters are added to the set of clusters.
        /// There are 'numRounds' number of rounds. We compute and store the distance of each new
        /// cluster from every round to all of the previous clusters and use it
        /// to avoid unnecessary computation by applying the triangle inequality.
        /// </summary>
        private sealed class SharedState
        {
            private readonly KMeansAcceleratedRowMap _acceleratedRowMap;
            public KMeansUtils.RowIndexGetter RowIndexGetter { get { return _acceleratedRowMap.RowIndexGetter; } }

            // _bestCluster holds the index of the closest cluster for an instance.
            // Note that this array is only allocated for MaxInstancesToAccelerate elements.
            private readonly int[] _bestCluster;

            // _bestWeight holds the weight of instance x to _bestCluster[x] where weight(x) = dist(x, _bestCluster[x])^2 - norm(x)^2.
            // Note that this array is only allocated for MaxInstancesToAccelerate elements.
            private readonly Float[] _bestWeight;

            // The distance of each newly added cluster from the previous round to every old cluster
            // the first dimension of this array is the size of numSamplesPerRound
            // and the second dimension is the size of numRounds * numSamplesPerRound.
            // _clusterDistances[i][j] = dist(cluster[i+clusterPrevCount], cluster[j])
            // where clusterPrevCount-1 is the last index of the old clusters
            // and new clusters are stored in cluster[cPrevIdx..clusterCount-1] where
            // clusterCount-1 is the last index of the clusters.
            private readonly Float[,] _clusterDistances;

            public SharedState(FeatureFloatVectorCursor.Factory factory, IChannel ch, long baseMaxInstancesToAccelerate,
                long clusterBytes, bool isParallel, int numRounds, int numSamplesPerRound, long totalTrainingInstances)
            {
                Contracts.AssertValue(factory);
                Contracts.AssertValue(ch);
                Contracts.Assert(numRounds > 0);
                Contracts.Assert(numSamplesPerRound > 0);
                Contracts.Assert(totalTrainingInstances > 0);

                _acceleratedRowMap = new KMeansAcceleratedRowMap(factory, ch, baseMaxInstancesToAccelerate, totalTrainingInstances, isParallel);
                Contracts.Assert(_acceleratedRowMap.MaxInstancesToAccelerate >= 0,
                    "MaxInstancesToAccelerate cannot be negative as KMeansAcceleratedRowMap sets it to 0 when baseMaxInstancesToAccelerate is negative");

                // If maxInstanceToAccelerate is positive, it means that there was enough room to fully allocate clusterDistances
                // and allocate _bestCluster and _bestWeight as much as possible.
                if (_acceleratedRowMap.MaxInstancesToAccelerate > 0)
                {
                    _clusterDistances = new Float[numSamplesPerRound, numRounds * numSamplesPerRound];
                    _bestCluster = new int[_acceleratedRowMap.MaxInstancesToAccelerate];
                    _bestWeight = new Float[_acceleratedRowMap.MaxInstancesToAccelerate];

                    for (int i = 0; i < _acceleratedRowMap.MaxInstancesToAccelerate; i++)
                    {
                        _bestCluster[i] = -1;
                        _bestWeight[i] = Float.MaxValue;
                    }
                }
                else
                    ch.Info("There was not enough room to store distances of clusters from each other for acceleration of KMeans|| initialization. A memory efficient approach is used instead.");
            }

            public int GetBestCluster(int idx)
            {
                return _bestCluster[idx];
            }

            public Float GetBestWeight(int idx)
            {
                return _bestWeight[idx];
            }

            /// <summary>
            /// When assigning an accelerated row to a cluster, we store away the weight
            /// to its closest cluster, as well as the identity of the new
            /// closest cluster. Note that bestWeight can be negative since it is
            /// corresponding to the weight of a distance which does not have
            /// the L2 norm of the point itself.
            /// </summary>
            public void SetInstanceCluster(int n, Float bestWeight, int bestCluster)
            {
                Contracts.Assert(0 <= n && n < _acceleratedRowMap.MaxInstancesToAccelerate);
                Contracts.AssertValue(_clusterDistances);
                Contracts.Assert(0 <= bestCluster && bestCluster < _clusterDistances.GetLength(1), "bestCluster must be between 0..clusterCount-1");

                // Update best cluster and best weight accordingly.
                _bestCluster[n] = bestCluster;
                _bestWeight[n] = bestWeight;
            }

            /// <summary>
            /// Computes and stores the distance of a new cluster to an old cluster
            /// <paramref name="newClusterFeatures"/> must be between 0..numSamplesPerRound-1.
            /// </summary>
            public void SetClusterDistance(int newClusterIdxWithinSample, ref VBuffer<Float> newClusterFeatures, Float newClusterL2,
                int oldClusterIdx, ref VBuffer<Float> oldClusterFeatures, Float oldClusterL2)
            {
                if (_clusterDistances != null)
                {
                    Contracts.Assert(newClusterL2 >= 0);
                    Contracts.Assert(0 <= newClusterIdxWithinSample && newClusterIdxWithinSample < _clusterDistances.GetLength(0), "newClusterIdxWithinSample must be between 0..numSamplesPerRound-1");
                    Contracts.Assert(0 <= oldClusterIdx && oldClusterIdx < _clusterDistances.GetLength(1));

                    _clusterDistances[newClusterIdxWithinSample, oldClusterIdx] =
                        MathUtils.Sqrt(newClusterL2 - 2 * VectorUtils.DotProduct(ref newClusterFeatures, ref oldClusterFeatures) + oldClusterL2);
                }
            }

            /// <summary>
            /// This function is the key to use triangle inequality. Given an instance x distance to the best
            /// old cluster, cOld, and distance of a new cluster, cNew, to cOld, this function evaluates whether
            /// the distance computation of dist(x,cNew) can be avoided.
            /// </summary>
            public bool CanWeightComputationBeAvoided(Float instanceDistanceToBestOldCluster, int bestOldCluster, int newClusterIdxWithinSample)
            {
                Contracts.Assert(instanceDistanceToBestOldCluster >= 0);
                Contracts.Assert(0 <= newClusterIdxWithinSample && newClusterIdxWithinSample < _clusterDistances.GetLength(0),
                    "newClusterIdxWithinSample must be between 0..numSamplesPerRound-1");
                Contracts.Assert((_clusterDistances == null) || (bestOldCluster == -1 ||
                    (0 <= bestOldCluster && bestOldCluster < _clusterDistances.GetLength(1))),
                    "bestOldCluster must be -1 (not set/not enought room) or between 0..clusterCount-1");
                // Only use this if the memory was allocated for _clusterDistances and bestOldCluster index is valid.
                if (_clusterDistances != null && bestOldCluster != -1)
                {
                    // This is dist(cNew,cOld).
                    Float distanceBetweenOldAndNewClusters = _clusterDistances[newClusterIdxWithinSample, bestOldCluster];

                    // Use triangle inequality to evaluate whether weight computation can be avoided
                    // dist(x,cNew) + dist(x,cOld) > dist(cOld,cNew) =>
                    // dist(x,cNew) > dist(cOld,cNew) - dist(x,cOld) =>
                    // If dist(cOld,cNew) - dist(x,cOld) > dist(x,cOld), then dist(x,cNew) > dist(x,cOld). Therefore it is
                    // not necessary to compute dist(x,cNew).
                    if (distanceBetweenOldAndNewClusters - instanceDistanceToBestOldCluster > instanceDistanceToBestOldCluster)
                        return true;
                }
                return false;
            }
        }

        /// <summary>
        /// This function finds the best cluster and the best weight for an instance using
        /// smart triangle inequality to avoid unnecessary weight computations.
        ///
        /// Note that <paramref name="needToStoreWeight"/> is used to avoid the storing the new cluster in
        /// final round. After the final round, best cluster information will be ignored.
        /// </summary>
        private static void FindBestCluster(ref VBuffer<Float> point, int pointRowIndex, SharedState initializationState,
            int clusterCount, int clusterPrevCount, VBuffer<Float>[] clusters, Float[] clustersL2s, bool needRealDistanceSquared, bool needToStoreWeight,
            out Float minDistanceSquared, out int bestCluster)
        {
            Contracts.AssertValue(initializationState);
            Contracts.Assert(clusterCount > 0);
            Contracts.Assert(0 <= clusterPrevCount && clusterPrevCount < clusterCount);
            Contracts.AssertValue(clusters);
            Contracts.AssertValue(clustersL2s);

            bestCluster = -1;

            if (pointRowIndex != -1) // if the space was available for cur in initializationState.
            {
                // pointNorm is necessary for using triangle inequality.
                Float pointNorm = VectorUtils.NormSquared(point);
                // We have cached distance information for this point.
                bestCluster = initializationState.GetBestCluster(pointRowIndex);
                Float bestWeight = initializationState.GetBestWeight(pointRowIndex);
                // This is used by CanWeightComputationBeAvoided function in order to shortcut weight computation.
                int bestOldCluster = bestCluster;

                Float pointDistanceSquared = pointNorm + bestWeight;
                // Make pointDistanceSquared zero if it is negative, which it can be due to floating point instability.
                // Do this before taking the square root.
                pointDistanceSquared = (pointDistanceSquared >= 0.0f) ? pointDistanceSquared : 0.0f;
                Float pointDistance = MathUtils.Sqrt(pointDistanceSquared);

                // bestCluster is the best cluster from 0 to cPrevIdx-1 and bestWeight is the corresponding weight.
                // So, this loop only needs to process clusters from cPrevIdx.
                for (int j = clusterPrevCount; j < clusterCount; j++)
                {
                    if (initializationState.CanWeightComputationBeAvoided(pointDistance, bestOldCluster, j - clusterPrevCount))
                    {
#if DEBUG
                        // Lets check if our invariant actually holds
                        Contracts.Assert(-2 * VectorUtils.DotProduct(ref point, ref clusters[j]) + clustersL2s[j] > bestWeight);
#endif
                        continue;
                    }
                    Float weight = -2 * VectorUtils.DotProduct(ref point, ref clusters[j]) + clustersL2s[j];
                    if (bestWeight >= weight)
                    {
                        bestWeight = weight;
                        bestCluster = j;
                    }
                }
                if (needToStoreWeight && bestCluster != bestOldCluster)
                    initializationState.SetInstanceCluster(pointRowIndex, bestWeight, bestCluster);

                if (needRealDistanceSquared)
                    minDistanceSquared = bestWeight + pointNorm;
                else
                    minDistanceSquared = bestWeight;
            }
            else
            {
                // We did not cache any information about this point.
                // So, we need to go over all clusters to find the best cluster.
                int discardSecondBestCluster;
                float discardSecondBestWeight;
                KMeansUtils.FindBestCluster(ref point, clusters, clustersL2s, clusterCount, needRealDistanceSquared,
                    out minDistanceSquared, out bestCluster, out discardSecondBestWeight, out discardSecondBestCluster);
            }

        }

        /// <summary>
        /// This method computes the memory requirement for _clusterDistances in SharedState (clusterBytes) and
        /// the maximum number of instances whose weight to the closest cluster can be memorized in order to avoid
        /// recomputation later.
        /// </summary>
        private static void ComputeAccelerationMemoryRequirement(long accelMemBudgetMb, int numSamplesPerRound, int numRounds, bool isParallel,
            out long maxInstancesToAccelerate, out long clusterBytes)
        {
            // Compute the memory requirement for _clusterDistances.
            clusterBytes = sizeof(Float)
                * numSamplesPerRound            // for each newly added cluster
                * numRounds * numSamplesPerRound; // for older cluster

            // Second, figure out how many instances can be accelerated.
            int bytesPerInstance =
                sizeof(int)                            // for bestCluster
                + sizeof(Float)                        // for bestWeight
                + (isParallel ? sizeof(int) + 16 : 0); // for parallel rowCursor index lookup HashArray storage (16 bytes for RowId, 4 bytes for internal 'next' index)

            maxInstancesToAccelerate = Math.Max(0, (accelMemBudgetMb * 1024 * 1024 - clusterBytes) / bytesPerInstance);
        }

        /// <summary>
        /// KMeans|| Implementation, see http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
        /// This algorithm will require:
        /// - (k * overSampleFactor * rounds * diminsionality * 4) bytes for the final sampled clusters.
        /// - (k * overSampleFactor * numThreads * diminsionality * 4) bytes for the per-round sampling.
        ///
        /// Uses memory in initializationState to cache distances and avoids unnecessary distance computations
        /// akin to YinYang-KMeans paper.
        ///
        /// Everywhere in this function, weight of an instance x from a cluster c means weight(x,c) = dist(x,c)^2-norm(x)^2.
        /// We store weight in most cases to avoid unnecessary computation of norm(x).
        /// </summary>
        public static void Initialize(IHost host, int numThreads, IChannel ch, FeatureFloatVectorCursor.Factory cursorFactory,
            int k, int dimensionality, VBuffer<Float>[] centroids, long accelMemBudgetMb,
            out long missingFeatureCount, out long totalTrainingInstances)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckValue(ch, nameof(ch));
            ch.CheckValue(cursorFactory, nameof(cursorFactory));
            ch.CheckValue(centroids, nameof(centroids));
            ch.CheckUserArg(numThreads > 0, nameof(KMeansPlusPlusTrainer.Arguments.NumThreads), "Must be positive");
            ch.CheckUserArg(k > 0, nameof(KMeansPlusPlusTrainer.Arguments.K), "Must be positive");
            ch.CheckParam(dimensionality > 0, nameof(dimensionality), "Must be positive");
            ch.CheckUserArg(accelMemBudgetMb >= 0, nameof(KMeansPlusPlusTrainer.Arguments.AccelMemBudgetMb), "Must be non-negative");

            int numRounds;
            int numSamplesPerRound;
            // If k is less than 60, we haven't reached the threshold where the coefficients in
            // the time complexity between KM|| and KM++ balance out to favor KM||. In this case
            // we push the number of rounds to k - 1 (we choose a single point before beginning
            // any round), and only take a single point per round. This effectively reduces KM|| to
            // KM++. This implementation is sill however advantageous as it uses parallel reservoir sampling
            // to parallelize each step.
            if (k < 60)
            {
                numRounds = k - 1;
                numSamplesPerRound = 1;
            }
            // From the paper, 5 rounds and l=2k is shown to achieve good results for real-world datasets
            // with large k values.
            else
            {
                numRounds = 5;
                int overSampleFactor = 2;
                numSamplesPerRound = overSampleFactor * k;
            }
            int totalSamples = numSamplesPerRound * numRounds + 1;

            using (var pCh = host.StartProgressChannel(" KMeansBarBarInitialization.Initialize"))
            {
                // KMeansParallel performs 'rounds' iterations through the dataset, an initialization
                // round to choose the first centroid, and a final iteration to weight the chosen centroids.
                // From a time-to-completion POV all these rounds take about the same amount of time.
                int logicalExternalRounds = 0;
                pCh.SetHeader(new ProgressHeader("rounds"), (e) => e.SetProgress(0, logicalExternalRounds, numRounds + 2));

                // The final chosen points, to be approximately clustered to determine starting
                // centroids.
                VBuffer<Float>[] clusters = new VBuffer<Float>[totalSamples];
                // L2s, kept for distance trick.
                Float[] clustersL2s = new Float[totalSamples];

                int clusterCount = 0;
                int clusterPrevCount = -1;

                SharedState initializationState;
                {
                    // First choose a single point to form the first cluster block using a random
                    // sample.
                    Heap<KMeansUtils.WeightedPoint>[] buffer = null;
                    var rowStats = KMeansUtils.ParallelWeightedReservoirSample(host, numThreads, 1, cursorFactory,
                        (ref VBuffer<Float> point, int pointIndex) => (Float)1.0, (FeatureFloatVectorCursor cur) => -1,
                        ref clusters, ref buffer);
                    totalTrainingInstances = rowStats.TotalTrainingInstances;
                    missingFeatureCount = rowStats.MissingFeatureCount;

                    bool isParallel = numThreads > 1;

                    long maxInstancesToAccelerate;
                    long clusterBytes;
                    ComputeAccelerationMemoryRequirement(accelMemBudgetMb, numSamplesPerRound, numRounds, isParallel, out maxInstancesToAccelerate, out clusterBytes);

                    initializationState = new SharedState(cursorFactory, ch,
                        maxInstancesToAccelerate, clusterBytes, isParallel, numRounds, numSamplesPerRound, totalTrainingInstances);

                    VBufferUtils.Densify(ref clusters[clusterCount]);
                    clustersL2s[clusterCount] = VectorUtils.NormSquared(clusters[clusterCount]);
                    clusterPrevCount = clusterCount;
                    ch.Assert(clusterCount - clusterPrevCount <= numSamplesPerRound);
                    clusterCount++;
                    logicalExternalRounds++;
                    pCh.Checkpoint(logicalExternalRounds, numRounds + 2);

                    // Next we iterate through the dataset 'rounds' times, each time we choose
                    // instances probabilistically, weighting them with a likelihood proportional
                    // to their distance from their best cluster. For each round, this gives us
                    // a new set of 'numSamplesPerRound' instances which are very likely to be as
                    // far from our current total running set of instances as possible.
                    VBuffer<Float>[] roundSamples = new VBuffer<Float>[numSamplesPerRound];

                    KMeansUtils.WeightFunc weightFn = (ref VBuffer<Float> point, int pointRowIndex) =>
                    {
                        Float distanceSquared;
                        int discardBestCluster;
                        FindBestCluster(ref point, pointRowIndex, initializationState, clusterCount, clusterPrevCount, clusters,
                            clustersL2s, true, true, out distanceSquared, out discardBestCluster);

                        return (distanceSquared >= 0.0f) ? distanceSquared : 0.0f;
                    };

                    for (int r = 0; r < numRounds; r++)
                    {
                        // Iterate through the dataset, sampling 'numSamplesPerFound' data rows using
                        // the weighted probability distribution.
                        KMeansUtils.ParallelWeightedReservoirSample(host, numThreads, numSamplesPerRound, cursorFactory, weightFn,
                            (FeatureFloatVectorCursor cur) => initializationState.RowIndexGetter(cur), ref roundSamples, ref buffer);
                        clusterPrevCount = clusterCount;
                        for (int i = 0; i < numSamplesPerRound; i++)
                        {

                            Utils.Swap(ref roundSamples[i], ref clusters[clusterCount]);
                            VBufferUtils.Densify(ref clusters[clusterCount]);
                            clustersL2s[clusterCount] = VectorUtils.NormSquared(clusters[clusterCount]);

                            for (int j = 0; j < clusterPrevCount; j++)
                                initializationState.SetClusterDistance(i, ref clusters[clusterCount], clustersL2s[clusterCount], j, ref clusters[j], clustersL2s[j]);

                            clusterCount++;
                        }
                        ch.Assert(clusterCount - clusterPrevCount <= numSamplesPerRound);
                        logicalExternalRounds++;
                        pCh.Checkpoint(logicalExternalRounds, numRounds + 2);
                    }
                    ch.Assert(clusterCount == clusters.Length);
                }

                // Finally, we do one last pass through the dataset, finding for
                // each instance the closest chosen cluster instance and summing these into buckets to be used
                // to weight each one of our candidate chosen clusters.
                Float[][] weightBuffer = null;
                Float[] totalWeights = null;
                KMeansUtils.ParallelMapReduce<Float[], Float[]>(
                    numThreads, host, cursorFactory, initializationState.RowIndexGetter,
                    (ref Float[] weights) => weights = new Float[totalSamples],
                    (ref VBuffer<Float> point, int pointRowIndex, Float[] weights, IRandom rand) =>
                    {
                        int bestCluster;
                        Float discardBestWeight;
                        FindBestCluster(ref point, pointRowIndex, initializationState, clusterCount, clusterPrevCount, clusters,
                            clustersL2s, false, false, out discardBestWeight, out bestCluster);
#if DEBUG
                        int debugBestCluster = KMeansUtils.FindBestCluster(ref point, clusters, clustersL2s);
                        ch.Assert(bestCluster == debugBestCluster);
#endif
                        weights[bestCluster]++;
                    },
                    (Float[][] workStateWeights, IRandom rand, ref Float[] weights) =>
                    {
                        weights = new Float[totalSamples];
                        for (int i = 0; i < workStateWeights.Length; i++)
                            CpuMathUtils.Add(workStateWeights[i], weights, totalSamples);
                    },
                    ref weightBuffer, ref totalWeights);
#if DEBUG
                // This is running the original code to make sure that the new code matches the semantic of the original code.
                Float[] debugTotalWeights = null;

                Float[][] debugWeightBuffer = null;
                KMeansUtils.ParallelMapReduce<Float[], Float[]>(
                    numThreads, host, cursorFactory, (FeatureFloatVectorCursor cur) => -1,
                    (ref Float[] weights) => weights = new Float[totalSamples],
                    (ref VBuffer<Float> point, int discard, Float[] weights, IRandom rand) => weights[KMeansUtils.FindBestCluster(ref point, clusters, clustersL2s)]++,
                    (Float[][] workStateWeights, IRandom rand, ref Float[] weights) =>
                    {
                        weights = new Float[totalSamples];
                        for (int i = 0; i < workStateWeights.Length; i++)
                            for (int j = 0; j < workStateWeights[i].Length; j++)
                                weights[j] += workStateWeights[i][j];
                    },
                    ref debugWeightBuffer, ref debugTotalWeights);

                for (int i = 0; i < totalWeights.Length; i++)
                    ch.Assert(totalWeights[i] == debugTotalWeights[i]);
#endif
                ch.Assert(totalWeights.Length == clusters.Length);
                logicalExternalRounds++;

                // If we sampled exactly the right number of points then we can
                // copy them directly to the output centroids.
                if (clusters.Length == k)
                {
                    for (int i = 0; i < k; i++)
                        clusters[i].CopyTo(ref centroids[i]);
                }
                // Otherwise, using our much smaller set of possible cluster centroids, go ahead
                // and invoke the standard PlusPlus initialization routine to reduce this
                // set down into k clusters.
                else
                {
                    ArrayDataViewBuilder arrDv = new ArrayDataViewBuilder(host);
                    arrDv.AddColumn(DefaultColumnNames.Features, PrimitiveType.FromKind(DataKind.R4), clusters);
                    arrDv.AddColumn(DefaultColumnNames.Weight, PrimitiveType.FromKind(DataKind.R4), totalWeights);
                    var subDataViewCursorFactory = new FeatureFloatVectorCursor.Factory(
                        new RoleMappedData(arrDv.GetDataView(), null, DefaultColumnNames.Features, weight: DefaultColumnNames.Weight), CursOpt.Weight | CursOpt.Features);
                    long discard1;
                    long discard2;
                    KMeansPlusPlusInit.Initialize(host, numThreads, ch, subDataViewCursorFactory, k, dimensionality, centroids, out discard1, out discard2, false);
                }
            }
        }
    }

    internal static class KMeansRandomInit
    {
        /// <summary>
        /// Initialize starting centroids via reservoir sampling.
        /// </summary>
        public static void Initialize(
            IHost host, int numThreads, IChannel ch, FeatureFloatVectorCursor.Factory cursorFactory,
            int k, VBuffer<Float>[] centroids,
            out long missingFeatureCount, out long totalTrainingInstances)
        {
            using (var pCh = host.StartProgressChannel("KMeansRandomInitialize"))
            {
                Heap<KMeansUtils.WeightedPoint>[] buffer = null;
                VBuffer<Float>[] outCentroids = null;
                var rowStats = KMeansUtils.ParallelWeightedReservoirSample(host, numThreads, k, cursorFactory,
                    (ref VBuffer<Float> point, int pointRowIndex) => 1f, (FeatureFloatVectorCursor cur) => -1,
                    ref outCentroids, ref buffer);
                missingFeatureCount = rowStats.MissingFeatureCount;
                totalTrainingInstances = rowStats.TotalTrainingInstances;

                for (int i = 0; i < k; i++)
                    Utils.Swap(ref centroids[i], ref outCentroids[i]);
            }
        }
    }

    internal static class KMeansLloydsYinYangTrain
    {
        private abstract class WorkChunkStateBase
        {
            protected readonly int K;
            protected readonly long MaxInstancesToAccelerate;

            // Standard KMeans per-cluster data.
            protected readonly VBuffer<Float>[] Centroids;
            protected readonly long[] ClusterSizes;

            // YinYang per-cluster data.
            // cached sum of the first maxInstancesToAccelerate instances assigned to cluster i for 0 <= i < _k
            protected readonly VBuffer<Float>[] CachedSum;

#if DEBUG
            public VBuffer<Float>[] CachedSumDebug { get { return CachedSum; } }
#endif

            protected double PreviousAverageScore;
            protected double AverageScore;
            // Number of entries in AverageScore calculation.
            private long _n;

            // Debug/Progress stats.
            protected long GloballyFiltered;
            protected long NumChanged;
            protected long NumUnchangedMissed;

            public static void Initialize(
                long maxInstancesToAccelerate, int k, int dimensionality, int numThreads,
                out ReducedWorkChunkState reducedState, out WorkChunkState[] workChunkArr)
            {
                if (numThreads == 1)
                    workChunkArr = new WorkChunkState[0];
                else
                {
                    workChunkArr = new WorkChunkState[numThreads];
                    for (int i = 0; i < numThreads; i++)
                        workChunkArr[i] = new WorkChunkState(maxInstancesToAccelerate, k, dimensionality);
                }

                reducedState = new ReducedWorkChunkState(maxInstancesToAccelerate, k, dimensionality);
            }

            protected WorkChunkStateBase(long maxInstancesToAccelerate, int k, int dimensionality)
            {
                Centroids = new VBuffer<Float>[k];
                for (int j = 0; j < k; j++)
                    Centroids[j] = VBufferUtils.CreateDense<Float>(dimensionality);

                ClusterSizes = new long[k];

                if (maxInstancesToAccelerate > 0)
                {
                    CachedSum = new VBuffer<Float>[k];
                    for (int j = 0; j < k; j++)
                        CachedSum[j] = VBufferUtils.CreateDense<Float>(dimensionality);
                }

                K = k;
                MaxInstancesToAccelerate = maxInstancesToAccelerate;
            }

            public void Clear(bool keepCachedSums = false)
            {
                for (int i = 0; i < K; i++)
                    VBufferUtils.Clear(ref Centroids[i]);
                NumChanged = 0;
                NumUnchangedMissed = 0;
                GloballyFiltered = 0;
                _n = 0;
                PreviousAverageScore = AverageScore;
                AverageScore = 0;
                Array.Clear(ClusterSizes, 0, K);

                if (!keepCachedSums)
                {
                    for (int i = 0; i < K; i++)
                        VBufferUtils.Clear(ref CachedSum[i]);
                }
            }

            public void KeepYinYangAssignment(int bestCluster)
            {
                // this instance does not change clusters in this iteration
                // also, this instance does not account for average score calculation

                // REVIEW: This seems wrong. We're accumulating the globally-filtered rows into
                // the count that we use to find the average score for this iteration, yet we don't
                // accumulate their distance. Removing this will lead to the case where N will be
                // zero (all points filtered) and will cause a div-zero NaN score. It seems that we
                // should remove this line, and special case N = 0 when we average the distance score,
                // otherwise we'll continue to report a smaller epsilon and terminate earlier than we
                // otherwise would have.
                ClusterSizes[bestCluster]++;
                _n++;
                GloballyFiltered++;
            }

            public void UpdateClusterAssignment(bool firstIteration, ref VBuffer<Float> features, int cluster, int previousCluster, Float distance)
            {
                if (firstIteration)
                {
                    VectorUtils.Add(ref features, ref CachedSum[cluster]);
                    NumChanged++;
                }
                else if (previousCluster != cluster)
                {
                    // update the cachedSum as the instance moves from (previous) bestCluster[n] to cluster
                    VectorUtils.Add(ref features, ref CachedSum[cluster]);
                    // There doesnt seem to be a Subtract function that does a -= b, so doing a += (-1 * b)
                    VectorUtils.AddMult(ref features, -1, ref CachedSum[previousCluster]);
                    NumChanged++;
                }
                else
                    NumUnchangedMissed++;

                UpdateClusterAssignmentMetrics(cluster, distance);
            }

            public void UpdateClusterAssignment(ref VBuffer<Float> features, int cluster, Float distance)
            {
                VectorUtils.Add(ref features, ref Centroids[cluster]);
                UpdateClusterAssignmentMetrics(cluster, distance);
            }

            private void UpdateClusterAssignmentMetrics(int cluster, Float distance)
            {
                _n++;
                ClusterSizes[cluster]++;
                AverageScore += distance;
            }

            /// <summary>
            /// Reduces the array of work chunks into this chunk, coalescing the
            /// results from multiple worker threads partitioned over a parallel cursor set and
            /// clearing their values to prepare them for the next iteration.
            /// </summary>
            public static void Reduce(WorkChunkState[] workChunkArr, ReducedWorkChunkState reducedState)
            {
                for (int i = 0; i < workChunkArr.Length; i++)
                {
                    reducedState.AverageScore += workChunkArr[i].AverageScore;
                    reducedState._n += workChunkArr[i]._n;
                    reducedState.GloballyFiltered += workChunkArr[i].GloballyFiltered;
                    reducedState.NumChanged += workChunkArr[i].NumChanged;
                    reducedState.NumUnchangedMissed += workChunkArr[i].NumUnchangedMissed;

                    for (int j = 0; j < reducedState.ClusterSizes.Length; j++)
                    {
                        reducedState.ClusterSizes[j] += workChunkArr[i].ClusterSizes[j];
                        VectorUtils.Add(ref workChunkArr[i].CachedSum[j], ref reducedState.CachedSum[j]);
                        VectorUtils.Add(ref workChunkArr[i].Centroids[j], ref reducedState.Centroids[j]);
                    }

                    workChunkArr[i].Clear(keepCachedSums: false);
                }

                Contracts.Assert(FloatUtils.IsFinite(reducedState.AverageScore));
                reducedState.AverageScore /= reducedState._n;
            }

            protected virtual int Foo { get; }
        }

        private sealed class WorkChunkState
            : WorkChunkStateBase
        {
            public WorkChunkState(long maxInstancesToAccelerate, int k, int dimensionality)
                : base(maxInstancesToAccelerate, k, dimensionality)
            {
            }
        }

        private sealed class ReducedWorkChunkState
            : WorkChunkStateBase
        {
            public double AverageScoreDelta => Math.Abs(PreviousAverageScore - AverageScore);

            public ReducedWorkChunkState(long maxInstancesToAccelerate, int k, int dimensionality)
                : base(maxInstancesToAccelerate, k, dimensionality)
            {
                AverageScore = double.PositiveInfinity;
            }

            /// <summary>
            /// Updates all the passed in variables with the results of the most recent iteration
            /// of cluster assignment. It is assumed that centroids will contain the previous results
            /// of this call.
            /// </summary>
            public void UpdateClusters(VBuffer<Float>[] centroids, Float[] centroidL2s, Float[] deltas, ref Float deltaMax)
            {
                bool isAccelerated = MaxInstancesToAccelerate > 0;
                deltaMax = 0;
                // calculate new centroids
                for (int i = 0; i < K; i++)
                {
                    if (isAccelerated)
                        VectorUtils.Add(ref CachedSum[i], ref Centroids[i]);

                    if (ClusterSizes[i] > 1)
                        VectorUtils.ScaleBy(ref Centroids[i], (Float)(1.0 / ClusterSizes[i]));

                    if (isAccelerated)
                    {
                        Float clusterDelta = MathUtils.Sqrt(VectorUtils.L2DistSquared(ref Centroids[i], ref centroids[i]));

                        deltas[i] = clusterDelta;
                        if (deltaMax < clusterDelta)
                            deltaMax = clusterDelta;
                    }

                    centroidL2s[i] = VectorUtils.NormSquared(Centroids[i]);
                }

                for (int i = 0; i < K; i++)
                    Utils.Swap(ref centroids[i], ref Centroids[i]);
            }

            public void ReportProgress(IProgressChannel pch, int iteration, int maxIterations)
            {
                pch.Checkpoint(AverageScore, NumChanged, NumUnchangedMissed + GloballyFiltered,
                    GloballyFiltered, iteration, maxIterations);
            }
        }

        private sealed class SharedState
        {
            private readonly KMeansAcceleratedRowMap _acceleratedRowMap;
            public int MaxInstancesToAccelerate => _acceleratedRowMap.MaxInstancesToAccelerate;
            public bool IsAccelerated => _acceleratedRowMap.IsAccelerated;
            public KMeansUtils.RowIndexGetter RowIndexGetter => _acceleratedRowMap.RowIndexGetter;

            public int Iteration;

            // YinYang  data.

            // the distance between the old and the new center for each cluster
            public readonly Float[] Delta;
            // max value of delta[i] for 0 <= i < _k
            public Float DeltaMax;

            // Per instance structures

            public int GetBestCluster(int idx)
            {
                return _bestCluster[idx];
            }

            // closest cluster for an instance
            private readonly int[] _bestCluster;

            // upper bound on the distance of an instance to its bestCluster
            private readonly Float[] _upperBound;
            // lower bound on the distance of an instance to every cluster other than its bestCluster
            private readonly Float[] _lowerBound;

            public SharedState(FeatureFloatVectorCursor.Factory factory, IChannel ch, long baseMaxInstancesToAccelerate, int k,
                bool isParallel, long totalTrainingInstances)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(factory);
                ch.Assert(k > 0);
                ch.Assert(totalTrainingInstances > 0);

                _acceleratedRowMap = new KMeansAcceleratedRowMap(factory, ch, baseMaxInstancesToAccelerate, totalTrainingInstances, isParallel);
                ch.Assert(MaxInstancesToAccelerate >= 0,
                    "MaxInstancesToAccelerate cannot be negative as KMeansAcceleratedRowMap sets it to 0 when baseMaxInstancesToAccelerate is negative");

                if (MaxInstancesToAccelerate > 0)
                {
                    // allocate data structures
                    Delta = new Float[k];

                    _bestCluster = new int[MaxInstancesToAccelerate];
                    _upperBound = new Float[MaxInstancesToAccelerate];
                    _lowerBound = new Float[MaxInstancesToAccelerate];
                }
            }

            /// <summary>
            /// When assigning an accelerated row to a cluster, we store away the distance
            /// to its closer and second closed cluster, as well as the identity of the new
            /// closest cluster. This method returns the last known closest cluster.
            /// </summary>
            public int SetYinYangCluster(int n, ref VBuffer<Float> features, Float minDistance, int minCluster, Float secMinDistance)
            {
                if (n == -1)
                    return -1;

                // update upper and lower bound
                // updates have to be true distances to use triangular inequality
                Float instanceNormSquared = VectorUtils.NormSquared(features);
                _upperBound[n] = MathUtils.Sqrt(instanceNormSquared + minDistance);
                _lowerBound[n] = MathUtils.Sqrt(instanceNormSquared + secMinDistance);
                int previousCluster = _bestCluster[n];
                _bestCluster[n] = minCluster;
                return previousCluster;
            }

            /// <summary>
            /// Updates the known YinYang bounds for the given row using the centroid position
            /// deltas from the previous iteration.
            /// </summary>
            public void UpdateYinYangBounds(int n)
            {
                Contracts.Assert(n != -1);
                _upperBound[n] += Delta[_bestCluster[n]];
                _lowerBound[n] -= DeltaMax;
            }

            /// <summary>
            /// Determines if the triangle distance inequality still applies to the given row,
            /// allowing us to avoid per-cluster distance computation.
            /// </summary>
            public bool IsYinYangGloballyBound(int n)
            {
                return _upperBound[n] < _lowerBound[n];
            }

#if DEBUG
            public void AssertValidYinYangBounds(int n, ref VBuffer<Float> features, VBuffer<Float>[] centroids)
            {
                // Assert that the global filter is indeed doing the right thing
                Float bestDistance = MathUtils.Sqrt(VectorUtils.L2DistSquared(ref features, ref centroids[_bestCluster[n]]));
                Contracts.Assert(KMeansLloydsYinYangTrain.AlmostLeq(bestDistance, _upperBound[n]));
                for (int j = 0; j < centroids.Length; j++)
                {
                    if (j == _bestCluster[n])
                        continue;
                    Float distance = MathUtils.Sqrt(VectorUtils.L2DistSquared(ref features, ref centroids[j]));

                    Contracts.Assert(AlmostLeq(_lowerBound[n], distance));
                }
            }
#endif
        }

        public static void Train(IHost host, int numThreads, IChannel ch, FeatureFloatVectorCursor.Factory cursorFactory,
            long totalTrainingInstances, int k, int dimensionality, int maxIterations,
            long accelMemBudgetInMb, Float convergenceThreshold, VBuffer<Float>[] centroids)
        {
            SharedState state;
            WorkChunkState[] workState;
            ReducedWorkChunkState reducedState;
            Initialize(ch, cursorFactory, totalTrainingInstances, numThreads, k, dimensionality, accelMemBudgetInMb,
                out state, out workState, out reducedState);
            Float[] centroidL2s = new Float[k];

            for (int i = 0; i < k; i++)
                centroidL2s[i] = VectorUtils.NormSquared(centroids[i]);

            using (var pch = host.StartProgressChannel("KMeansTrain"))
            {
                pch.SetHeader(new ProgressHeader(
                    new[] { "Average Score", "# of Examples with Reassigned Cluster",
                            "# of Examples with Same Cluster", "Globally Filtered" },
                    new[] { "iterations" }),
                    (e) => e.SetProgress(0, state.Iteration, maxIterations));

                bool isConverged = false;
                while (!isConverged && state.Iteration < maxIterations)
                {
                    // assign instances to clusters and calculate total score
                    reducedState.Clear(keepCachedSums: true);

                    if (numThreads > 1)
                    {
                        // Build parallel cursor set and run....
                        var set = cursorFactory.CreateSet(numThreads);
                        Action[] ops = new Action[set.Length];
                        for (int i = 0; i < ops.Length; i++)
                        {
                            int chunkId = i;
                            ops[i] = new Action(() =>
                            {
                                using (var cursor = set[chunkId])
                                    ProcessChunk(cursor, state, workState[chunkId], k, centroids, centroidL2s);
                            });
                        }

                        Parallel.Invoke(new ParallelOptions()
                        {
                            MaxDegreeOfParallelism = numThreads
                        }, ops);
                    }
                    else
                    {
                        using (var cursor = cursorFactory.Create())
                            ProcessChunk(cursor, state, reducedState, k, centroids, centroidL2s);
                    }

                    WorkChunkState.Reduce(workState, reducedState);

                    reducedState.ReportProgress(pch, state.Iteration, maxIterations);

#if DEBUG
                    if (state.IsAccelerated)
                    {
                        // Assert that cachedSum[i] is equal to the sum of the first maxInstancesToAccelerate instances assigned to cluster i
                        var cachedSumCopy = new VBuffer<Float>[k];
                        for (int i = 0; i < k; i++)
                            cachedSumCopy[i] = VBufferUtils.CreateDense<Float>(dimensionality);

                        using (var cursor = cursorFactory.Create())
                        {
                            int numCounted = 0;
                            while (cursor.MoveNext() && numCounted < state.MaxInstancesToAccelerate)
                            {
                                int id = state.RowIndexGetter(cursor);
                                if (id != -1)
                                {
                                    VectorUtils.Add(ref cursor.Features, ref cachedSumCopy[state.GetBestCluster(id)]);
                                    numCounted++;
                                }
                            }
                        }

                        for (int i = 0; i < k; i++)
                        {
                            for (int j = 0; j < dimensionality; j++)
                                Contracts.Assert(AlmostEq(reducedState.CachedSumDebug[i].Values[j], cachedSumCopy[i].Values[j]));
                        }
                    }
#endif
                    reducedState.UpdateClusters(centroids, centroidL2s, state.Delta, ref state.DeltaMax);
                    isConverged = reducedState.AverageScoreDelta < convergenceThreshold;
                    state.Iteration++;

                    if (state.Iteration % 100 == 0)
                        KMeansUtils.VerifyModelConsistency(centroids);
                }
            }
        }

        private static void Initialize(
            IChannel ch, FeatureFloatVectorCursor.Factory factory, long totalTrainingInstances,
            int numThreads, int k, int dimensionality, long accelMemBudgetMb,
            out SharedState state,
            out WorkChunkState[] perThreadWorkState, out ReducedWorkChunkState reducedWorkState)
        {
            // In the case of a single thread, we use a single WorkChunkState instance
            // and skip the reduce step, in the case of a parallel implementation we use
            // the last WorkChunkState to reduce the per-thread chunks into a single
            // result prior to the KMeans update step.
            int neededPerThreadWorkStates = numThreads == 1 ? 0 : numThreads;

            // Accelerating KMeans requires the following data structures.
            // The algorithm is based on the YinYang KMeans algorithm [ICML'15], http://research.microsoft.com/apps/pubs/default.aspx?id=252149
            // These data structures are allocated only as allowed by the _accelMemBudgetMb parameter
            // if _accelMemBudgetMb is zero, then the algorithm below reduces to the original KMeans++ implementation
            int bytesPerCluster =
                sizeof(Float) +                                 // for delta
                sizeof(Float) * dimensionality * (neededPerThreadWorkStates + 1);  // for cachedSum

            int bytesPerInstance =
                sizeof(int) +                             // for bestCluster
                sizeof(Float) +                           // for upperBound
                sizeof(Float) +                           // for lowerBound
                (numThreads > 1 ? sizeof(int) + 16 : 0); // for parallel rowCursor index lookup HashArray storage (16 bytes for RowId, 4 bytes for internal 'next' index)

            long maxInstancesToAccelerate = Math.Max(0, (accelMemBudgetMb * 1024 * 1024 - bytesPerCluster * k) / bytesPerInstance);

            state = new SharedState(factory, ch, maxInstancesToAccelerate, k, numThreads > 1, totalTrainingInstances);
            WorkChunkState.Initialize(maxInstancesToAccelerate, k, dimensionality, numThreads,
                out reducedWorkState, out perThreadWorkState);
        }

        /// <summary>
        /// Performs the 'update' step of KMeans. This method is passed a WorkChunkState. In the parallel version
        /// this chunk will be one of _numThreads chunks and the RowCursor will be part of a RowCursorSet. In the
        /// unthreaded version, this chunk will be the final chunk and hold state for the entire data set.
        /// </summary>
        private static void ProcessChunk(FeatureFloatVectorCursor cursor, SharedState state, WorkChunkStateBase chunkState, int k, VBuffer<Float>[] centroids, Float[] centroidL2s)
        {
            while (cursor.MoveNext())
            {
                int n = state.RowIndexGetter(cursor);
                bool firstIteration = state.Iteration == 0;

                // We cannot accelerate the first iteration. In other iterations, we can only accelerate the first maxInstancesToAccelerate
                if (!firstIteration && n != -1)
                {
                    state.UpdateYinYangBounds(n);
                    if (state.IsYinYangGloballyBound(n))
                    {
                        chunkState.KeepYinYangAssignment(state.GetBestCluster(n));
#if DEBUG
                        state.AssertValidYinYangBounds(n, ref cursor.Features, centroids);
#endif
                        continue;
                    }
                }

                Float minDistance;
                Float secMinDistance;
                int cluster;
                int secCluster;
                KMeansUtils.FindBestCluster(ref cursor.Features, centroids, centroidL2s, k, false, out minDistance, out cluster, out secMinDistance, out secCluster);

                if (n == -1)
                    chunkState.UpdateClusterAssignment(ref cursor.Features, cluster, minDistance);
                else
                {
                    int prevCluster = state.SetYinYangCluster(n, ref cursor.Features, minDistance, cluster, secMinDistance);
                    chunkState.UpdateClusterAssignment(firstIteration, ref cursor.Features, cluster, prevCluster, minDistance);
                }
            }
        }

#if DEBUG
        private const Double FloatingPointErrorThreshold = 0.1F;

        private static bool AlmostEq(Float af, Float bf)
        {
            Double a = (Double)af;
            Double b = (Double)bf;

            // First check if a and b are close together
            if (Math.Abs(a - b) < FloatingPointErrorThreshold)
                return true;

            // Else, it could simply mean that a and b are large, so check for relative error
            // Note: a and b being large means that we are not dividing by a small number below
            // Also, dividing by the larger number ensures that there is no division by zero problem
            Double relativeError;
            if (Math.Abs(a) > Math.Abs(b))
                relativeError = Math.Abs((a - b) / a);
            else
                relativeError = Math.Abs((a - b) / b);

            if (relativeError < FloatingPointErrorThreshold)
                return true;

            return false;
        }

        private static bool AlmostLeq(Float a, Float b)
        {
            if (AlmostEq(a, b))
                return true;

            return ((Double)a - (Double)b) < 0;
        }
#endif
    }

    internal static class KMeansUtils
    {
        public struct WeightedPoint
        {
            public double Weight;
            public VBuffer<Float> Point;
        }

        public struct RowStats
        {
            public long MissingFeatureCount;
            public long TotalTrainingInstances;
        }

        public delegate Float WeightFunc(ref VBuffer<Float> point, int pointRowIndex);

        /// <summary>
        /// Performs a multithreaded version of weighted reservior sampling, returning
        /// an array of numSamples, where each sample has been selected from the
        /// data set with a probability of numSamples/N * weight/(sum(weight)). Buffer
        /// is sized to the number of threads plus one and stores the minheaps needed to
        /// perform the per-thread reservior samples.
        ///
        /// This method assumes that the numSamples is much smaller than the full dataset as
        /// it expects to be able to sample numSamples * numThreads.
        ///
        /// This is based on the 'A-Res' algorithm in 'Weighted Random Sampling', 2005; Efraimidis, Spirakis:
        /// http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
        /// </summary>
        public static RowStats ParallelWeightedReservoirSample(
            IHost host, int numThreads,
            int numSamples, FeatureFloatVectorCursor.Factory factory,
            WeightFunc weightFn,
            RowIndexGetter rowIndexGetter,
            ref VBuffer<Float>[] dst,
            ref Heap<WeightedPoint>[] buffer)
        {
            host.AssertValue(host);
            host.AssertValue(factory);
            host.AssertValue(weightFn);
            host.Assert(numSamples > 0);
            host.Assert(numThreads > 0);

            Heap<WeightedPoint> outHeap = null;
            var rowStats = ParallelMapReduce<Heap<WeightedPoint>, Heap<WeightedPoint>>(
                numThreads, host, factory, rowIndexGetter,
                (ref Heap<WeightedPoint> heap) =>
                {
                    if (heap == null)
                        heap = new Heap<WeightedPoint>((x, y) => x.Weight > y.Weight, numSamples);
                    else
                        heap.Clear();
                },
                (ref VBuffer<Float> point, int pointRowIndex, Heap<WeightedPoint> heap, IRandom rand) =>
                {
                    // We use distance as a proxy for 'is the same point'. By excluding
                    // all points that lie within a very small distance of our current set of
                    // centroids we force the algorithm to explore more broadly and avoid creating a
                    // set of centroids containing the same, or very close to the same, point
                    // more than once.
                    Float sameClusterEpsilon = (Float)1e-15;

                    Float weight = weightFn(ref point, pointRowIndex);

                    // If numeric instability has forced it to zero, then we bound it to epsilon to
                    // keep the key valid and avoid NaN, (although the math does tend to work out regardless:
                    // 1 / 0 => Inf, base ^ Inf => 0, when |base| < 1)
                    if (weight == 0)
                        weight = Float.Epsilon;

                    if (weight <= sameClusterEpsilon)
                        return;

                    double key = Math.Log(rand.NextDouble()) / weight;
                    // If we are less than all of the samples in the heap and the heap
                    // is full already, early return.
                    if (heap.Count == numSamples && key <= heap.Top.Weight)
                        return;

                    WeightedPoint wRow;
                    if (heap.Count == numSamples)
                        wRow = heap.Pop();
                    else
                        wRow = new WeightedPoint();

                    wRow.Weight = key;
                    Utils.Swap(ref wRow.Point, ref point);
                    heap.Add(wRow);
                },
                (Heap<WeightedPoint>[] heaps, IRandom rand, ref Heap<WeightedPoint> finalHeap) =>
                {
                    host.Assert(finalHeap == null);
                    finalHeap = new Heap<WeightedPoint>((x, y) => x.Weight > y.Weight, numSamples);
                    for (int i = 0; i < heaps.Length; i++)
                    {
                        host.AssertValue(heaps[i]);
                        host.Assert(heaps[i].Count <= numSamples, "heaps[i].Count must not be greater than numSamples");
                        while (heaps[i].Count > 0)
                        {
                            var row = heaps[i].Pop();
                            if (finalHeap.Count < numSamples)
                                finalHeap.Add(row);
                            else if (row.Weight > finalHeap.Top.Weight)
                            {
                                finalHeap.Pop();
                                finalHeap.Add(row);
                            }
                        }
                    }
                }, ref buffer, ref outHeap);

            if (outHeap.Count != numSamples)
                throw host.Except("Failed to initialize clusters: too few examples");

            // Keep in mind that the distribution of samples in dst will not be random. It will
            // have the residual minHeap ordering.
            Utils.EnsureSize(ref dst, numSamples);
            for (int i = 0; i < numSamples; i++)
            {
                var row = outHeap.Pop();
                Utils.Swap(ref row.Point, ref dst[i]);
            }

            return rowStats;
        }

        public delegate void InitAction<TPartitionState>(ref TPartitionState val);
        public delegate int RowIndexGetter(FeatureFloatVectorCursor cur);
        public delegate void MapAction<TPartitionState>(ref VBuffer<Float> point, int rowIndex, TPartitionState state, IRandom rand);
        public delegate void ReduceAction<TPartitionState, TGlobalState>(TPartitionState[] intermediates, IRandom rand, ref TGlobalState result);

        /// <summary>
        /// Takes a data cursor and perform an in-memory parallel aggregation operation on it. This
        /// helper wraps some of the behavior common to parallel operations over a IRowCursor set,
        /// including building the set, creating separate IRandom instances, and IRowCursor disposal.
        /// </summary>
        /// <typeparam name="TPartitionState">The type that each parallel cursor will be expected to aggregate to.</typeparam>
        /// <typeparam name="TGlobalState">The type of the final output from combining each per-thread instance of TInterAgg.</typeparam>
        /// <param name="numThreads"></param>
        /// <param name="baseHost"></param>
        /// <param name="factory"></param>
        /// <param name="rowIndexGetter"></param>
        /// <param name="initChunk">Initializes an instance of TInterAgg, or prepares/clears it if it is already allocated.</param>
        /// <param name="mapper">Invoked for every row, should update TInterAgg using row cursor data.</param>
        /// <param name="reducer">Invoked after all row cursors have completed, combines the entire array of TInterAgg instances into a final TAgg result.</param>
        /// <param name="buffer">A reusable buffer array of TInterAgg.</param>
        /// <param name="result">A reusable reference to the final result.</param>
        /// <returns></returns>
        public static RowStats ParallelMapReduce<TPartitionState, TGlobalState>(
            int numThreads,
            IHost baseHost,
            FeatureFloatVectorCursor.Factory factory,
            RowIndexGetter rowIndexGetter,
            InitAction<TPartitionState> initChunk,
            MapAction<TPartitionState> mapper,
            ReduceAction<TPartitionState, TGlobalState> reducer,
            ref TPartitionState[] buffer, ref TGlobalState result)
        {
            var set = factory.CreateSet(numThreads);
            int numCursors = set.Length;
            Action[] workArr = new Action[numCursors];
            Utils.EnsureSize(ref buffer, numCursors, numCursors);

            for (int i = 0; i < numCursors; i++)
            {
                int ii = i;
                var cur = set[i];
                initChunk(ref buffer[i]);
                var innerWorkState = buffer[i];
                IRandom rand = RandomUtils.Create(baseHost.Rand);
                workArr[i] = () =>
                {
                    using (cur)
                    {
                        while (cur.MoveNext())
                            mapper(ref cur.Features, rowIndexGetter(cur), innerWorkState, rand);
                    }
                };
            }
            Parallel.Invoke(new ParallelOptions()
            {
                MaxDegreeOfParallelism = numThreads
            }, workArr);

            reducer(buffer, baseHost.Rand, ref result);
            return new RowStats()
            {
                MissingFeatureCount = set.Select(cur => cur.BadFeaturesRowCount).Sum(),
                TotalTrainingInstances = set.Select(cur => cur.KeptRowCount).Sum()
            };
        }

        public static int FindBestCluster(ref VBuffer<Float> features, VBuffer<Float>[] centroids, Float[] centroidL2s)
        {
            Float discard1;
            Float discard2;
            int discard3;
            int cluster;
            FindBestCluster(ref features, centroids, centroidL2s, centroids.Length, false, out discard1, out cluster, out discard2, out discard3);
            return cluster;
        }

        public static int FindBestCluster(ref VBuffer<Float> features, VBuffer<Float>[] centroids, Float[] centroidL2s, int centroidCount, bool realWeight, out Float minDistance)
        {
            Float discard1;
            int discard2;
            int cluster;
            FindBestCluster(ref features, centroids, centroidL2s, centroidCount, realWeight, out minDistance, out cluster, out discard1, out discard2);
            return cluster;
        }

        /// <summary>
        /// Given a point and a set of centroids this method will determine the closest centroid
        /// using L2 distance. It will return a value equivalent to that distance, the index of the
        /// closest cluster, and a value equivalent to the distance to the second-nearest cluster.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="centroids"></param>
        /// <param name="centroidL2s">The L2 norms of the centroids. Used for efficiency and expected to be computed up front.</param>
        /// <param name="centroidCount">The number of centroids. Must be less than or equal to the length of the centroid array.</param>
        /// <param name="needRealDistance">Whether to return a real L2 distance, or a value missing the L2 norm of <paramref name="features"/>.</param>
        /// <param name="minDistance">The distance between <paramref name="features"/> and the nearest centroid in <paramref name="centroids" />.</param>
        /// <param name="cluster">The index of the nearest centroid.</param>
        /// <param name="secMinDistance">The second nearest distance, or PosInf if <paramref name="centroids" /> only contains a single point.</param>
        /// <param name="secCluster">The index of the second nearest centroid, or -1 if <paramref name="centroids" /> only contains a single point.</param>
        public static void FindBestCluster(
            ref VBuffer<Float> features,
            VBuffer<Float>[] centroids, Float[] centroidL2s, int centroidCount, bool needRealDistance,
            out Float minDistance, out int cluster, out Float secMinDistance, out int secCluster)
        {
            Contracts.Assert(centroids.Length >= centroidCount && centroidL2s.Length >= centroidCount && centroidCount > 0);
            Contracts.Assert(features.Length == centroids[0].Length);

            minDistance = Float.PositiveInfinity;
            secMinDistance = Float.PositiveInfinity;
            cluster = 0; // currently assigned cluster to the instance
            secCluster = -1;

            for (int j = 0; j < centroidCount; j++)
            {
                // this is not a real distance, since we don't add L2 norm of the instance
                // This won't affect minimum calculations, and total score will just be lowered by sum(L2 norms)
                Float distance = -2 * VectorUtils.DotProduct(ref features, ref centroids[j]) + centroidL2s[j];

                if (distance <= minDistance)
                {
                    // Note the equal to in the branch above. This is important when secMinDistance == minDistance
                    secMinDistance = minDistance;
                    secCluster = cluster;
                    minDistance = distance;
                    cluster = j;
                }
                else if (distance < secMinDistance)
                {
                    secMinDistance = distance;
                    secCluster = j;
                }
            }

            Contracts.Assert(FloatUtils.IsFinite(minDistance));
            Contracts.Assert(centroidCount == 1 || (FloatUtils.IsFinite(secMinDistance) && secCluster >= 0));
            Contracts.Assert(minDistance <= secMinDistance);

            if (needRealDistance)
            {
                Float l2 = VectorUtils.NormSquared(features);
                minDistance += l2;
                if (secCluster != -1)
                    secMinDistance += l2;
            }
        }

        /// <summary>
        /// Checks that all coordinates of all centroids are finite, and throws otherwise
        /// </summary>
        public static void VerifyModelConsistency(VBuffer<Float>[] centroids)
        {
            foreach (var centroid in centroids)
                Contracts.Check(centroid.Items().Select(x => x.Value).All(FloatUtils.IsFinite), "Model training failed: non-finite coordinates are generated");
        }
    }
}
