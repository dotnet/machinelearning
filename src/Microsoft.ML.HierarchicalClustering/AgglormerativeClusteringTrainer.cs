// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Numeric;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Internal.Internallearn;
using System.Collections;

[assembly: LoadableClass(AgglomerativeClusteringTrainer.Summary, typeof(AgglomerativeClusteringTrainer), typeof(AgglomerativeClusteringTrainer.Options),
    new[] { typeof(SignatureClusteringTrainer), typeof(SignatureTrainer) },
    AgglomerativeClusteringTrainer.UserNameValue,
    AgglomerativeClusteringTrainer.LoadNameValue,
    AgglomerativeClusteringTrainer.ShortName, "AgglomerativeClustering")]

[assembly: LoadableClass(typeof(void), typeof(AgglomerativeClusteringTrainer), null, typeof(SignatureEntryPointModule), "AgglomerativeClustering")]

namespace Microsoft.ML.Trainers
{
    public class AgglomerativeClusteringTrainer : TrainerEstimatorBase<ClusteringPredictionTransformer<AgglomerativeClusteringModelParameters>, AgglomerativeClusteringModelParameters>
    {
        internal const string LoadNameValue = "AgglomerativeClustering";
        internal const string UserNameValue = "Agglomerative Clustering";
        internal const string ShortName = "AggClust";
        internal const string Summary = "Hierarchical clustering is a method of cluster analysis which build a hierarchy of clusters."
            + "Agglomerative Clustering is a bottom-up approach: each observation starts in its own cluster, "
            + "and pairs of clusters are merged as one moves up the hierarchy.";

        [BestFriend]
        internal static class Defaults
        {
            /// <value>The number of clusters.</value>
            public const int NumberOfClusters = 5;

            /// <value>Linkage algorithm.</value>
            public const LinkageCriterion Linkage = LinkageCriterion.Average;
        }

        /// <summary>
        /// Options for the <see cref="AgglomerativeClusteringTrainer"/> as used in [AgglomerativeClusteringTrainer(Options)](xref:Microsoft.ML.HierarchicalClusteringExtensions.AgglomerativeClustering(Microsoft.ML.ClusteringCatalog.ClusteringTrainers,Microsoft.ML.Trainers.AgglomerativeClusteringTrainer.Options)).
        /// </summary>
        public sealed class Options : UnsupervisedTrainerInputBaseWithWeight
        {
            /// <summary>
            /// The number of clusters.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of clusters", SortOrder = 50, Name = "K")]
            public int NumberOfClusters = Defaults.NumberOfClusters;

            /// <summary>
            /// Cluster Linkage algorithm.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Linkage algorithm", ShortName = "link")]
            public LinkageCriterion LinkageCriterion = LinkageCriterion.Average;
        }

        private readonly int _k;
        private readonly LinkageCriterion _linkCriteria;
        private readonly Dictionary<int, ClusterInfo> _clusters = new Dictionary<int, ClusterInfo>();
        private readonly List<DataSimilarity> _similarities = new List<DataSimilarity>();
        private readonly List<int> _dendrogram = new List<int>();
        private ILinkageAlgorithm _linkageAlgorithm;

        public override TrainerInfo Info { get; }
        private protected override PredictionKind PredictionKind => PredictionKind.Clustering;

        /// <summary>
        /// Initializes a new instance of <see cref="AgglomerativeClusteringTrainer"/>
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> to use.</param>
        /// <param name="options">The advanced options of the algorithm.</param>
        internal AgglomerativeClusteringTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), default, TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName))
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckUserArg(options.NumberOfClusters > 0, nameof(options.NumberOfClusters), "Must be positive");

            _k = options.NumberOfClusters;

            _linkCriteria = options.LinkageCriterion;

            Info = new TrainerInfo();
        }

        private protected override AgglomerativeClusteringModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector(out int dimensionality);
            Contracts.Assert(dimensionality > 0);

            using (var ch = Host.Start("Training"))
            {
                return TrainCore(ch, data);
            }
        }

        private AgglomerativeClusteringModelParameters TrainCore(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            var cursorFactory = new FeatureFloatVectorCursor.Factory(data);

            this.Initialize(cursorFactory);
            this.SetLinkageAlgorithm();
            int totalTrainingInstances = _similarities.Count;
            Train(totalTrainingInstances);

            ch.Info("Model trained successfully on {0} instances", totalTrainingInstances);

            return new AgglomerativeClusteringModelParameters(Host, _clusters, _dendrogram, copyIn: false);
        }

        private void Train(int instances)
        {
            int iterations = instances - _k;
            for (int i = 0; i < iterations; i++)
            {
                this._linkageAlgorithm.UpdateCluster(out var clusterIdx1, out var clusterIdx2);
                _dendrogram[clusterIdx2] = clusterIdx1;
            }
        }

        /// <summary>
        /// Read data. Compute initial set of clusters and distance with all other clusters
        /// Initially every data point lies in its own cluster of size 1.
        /// </summary>
        /// <param name="cursorFactory"></param>
        private void Initialize(FeatureFloatVectorCursor.Factory cursorFactory)
        {
            int instances = 0;
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    var data = new DataSimilarity();
                    var cluster = new ClusterInfo();
                    cursor.Features.CopyTo(ref data.DataPoint);
                    data.NormL2S = VectorUtils.NormSquared(cursor.Features);

                    if (instances != 1)
                    {
                        data.Similarity = GetSimilarity(data.DataPoint, data.NormL2S);
                        for (var i = 0; i < instances; i++)
                        {
                            cluster.Similarity.Add(i, data.Similarity[i]);
                            _clusters[i].Similarity[instances] = data.Similarity[i];
                        }
                    }
                    _clusters.Add(instances, cluster);
                    _similarities.Add(data);
                    _dendrogram.Add(instances);
                    instances++;
                }
            }
        }

        /// <summary>
        /// Select a linkage algorithm according the user option.
        /// </summary>
        private void SetLinkageAlgorithm()
        {
            switch (_linkCriteria)
            {
                case LinkageCriterion.Average:
                    _linkageAlgorithm = new AverageLinkage(this._clusters);
                    break;
                case LinkageCriterion.Complete:
                    _linkageAlgorithm = new CompleteLinkage(this._clusters);
                    break;
                case LinkageCriterion.Single:
                    _linkageAlgorithm = new SingleLinkage(this._clusters);
                    break;

            }
        }

        /// <summary>
        /// Returns List of euclidean distance from point to all other points.
        /// </summary>
        /// <param name="point"></param>
        /// <param name="normL2S">Squared L2 norm of point </param>
        /// <returns></returns>
        private List<float> GetSimilarity(VBuffer<float> point, float normL2S)
        {
            var result = new List<float>();
            foreach (var kv in _similarities)
            {
                var distance = -2 * VectorUtils.DotProduct(in point, in kv.DataPoint)
                    + normL2S + kv.NormL2S;
                result.Add(distance);
            }
            return result;
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Vector,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),

                new SchemaShape.Column(DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.UInt32,
                        true,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override ClusteringPredictionTransformer<AgglomerativeClusteringModelParameters> MakeTransformer(AgglomerativeClusteringModelParameters model, DataViewSchema trainSchema)
        => new ClusteringPredictionTransformer<AgglomerativeClusteringModelParameters>(Host, model, trainSchema, null);
    }

    public enum LinkageCriterion
    {
        Average = 0,
        Complete = 1,
        Single = 2
    }

    /// <summary>
    /// Class for maintaining training data along with squared l2 norm and euclidean distance with all other training data points
    /// </summary>
    internal class DataSimilarity
    {
        // Vector representing single row of data
        public VBuffer<float> DataPoint;

        // Squared L2 norm of the vector
        public float NormL2S;

        // Distance with all other rows of data
        public List<float> Similarity = new List<float>();
    }

    /// <summary>
    /// Class for cluster information. List of members of cluster. Similarity with other clusters.
    /// </summary>
    internal class ClusterInfo
    {
        // List of members of cluster.
        public readonly List<int> Members = new List<int>();

        //Similarity with other clusters.
        public readonly Dictionary<int, float> Similarity = new Dictionary<int, float>();
    }
}
