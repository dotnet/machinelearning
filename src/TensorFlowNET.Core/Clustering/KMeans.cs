using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow.Clustering
{
    /// <summary>
    /// Creates the graph for k-means clustering.
    /// </summary>
    public class KMeans
    {
        public const string CLUSTERS_VAR_NAME = "clusters";

        public const string SQUARED_EUCLIDEAN_DISTANCE = "squared_euclidean";
        public const string COSINE_DISTANCE = "cosine";
        public const string RANDOM_INIT = "random";
        public const string KMEANS_PLUS_PLUS_INIT = "kmeans_plus_plus";
        public const string KMC2_INIT = "kmc2";

        Tensor[] _inputs;
        int _num_clusters;
        string _initial_clusters;
        string _distance_metric;
        bool _use_mini_batch;
        int _mini_batch_steps_per_iteration;
        int _random_seed;
        int _kmeans_plus_plus_num_retries;
        int _kmc2_chain_length;

        public KMeans(Tensor inputs,
            int num_clusters,
            string initial_clusters = RANDOM_INIT,
            string distance_metric = SQUARED_EUCLIDEAN_DISTANCE,
            bool use_mini_batch = false,
            int mini_batch_steps_per_iteration = 1,
            int random_seed = 0,
            int kmeans_plus_plus_num_retries = 2,
            int kmc2_chain_length = 200)
        {
            _inputs = new Tensor[] { inputs };
            _num_clusters = num_clusters;
            _initial_clusters = initial_clusters;
            _distance_metric = distance_metric;
            _use_mini_batch = use_mini_batch;
            _mini_batch_steps_per_iteration = mini_batch_steps_per_iteration;
            _random_seed = random_seed;
            _kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries;
            _kmc2_chain_length = kmc2_chain_length;
        }

        public object training_graph()
        {
            var initial_clusters = _initial_clusters;
            var num_clusters = ops.convert_to_tensor(_num_clusters);
            var inputs = _inputs;
            var vars = _create_variables(num_clusters);
            var cluster_centers_var = vars[0];
            var cluster_centers_initialized = vars[1];
            var total_counts = vars[2];
            var cluster_centers_updated = vars[3];
            var update_in_steps = vars[4];

            var init_op = new _InitializeClustersOpFactory(_inputs, num_clusters, initial_clusters, _distance_metric,
                _random_seed, _kmeans_plus_plus_num_retries,
                _kmc2_chain_length, cluster_centers_var, cluster_centers_updated,
                cluster_centers_initialized).op();

            throw new NotImplementedException("KMeans training_graph");
        }

        private RefVariable[] _create_variables(Tensor num_clusters)
        {
            var init_value = constant_op.constant(new float[0], dtype: TF_DataType.TF_FLOAT);
            var cluster_centers = tf.Variable(init_value, name: CLUSTERS_VAR_NAME, validate_shape: false);
            var cluster_centers_initialized = tf.Variable(false, dtype: TF_DataType.TF_BOOL, name: "initialized");
            RefVariable update_in_steps = null;
            if (_use_mini_batch && _mini_batch_steps_per_iteration > 1)
                throw new NotImplementedException("KMeans._create_variables");
            else
            {
                var cluster_centers_updated = cluster_centers;
                var ones = array_ops.ones(new Tensor[] { num_clusters }, dtype: TF_DataType.TF_INT64);
                var cluster_counts = _use_mini_batch ? tf.Variable(ones) : null;
                return new RefVariable[]
                {
                    cluster_centers,
                    cluster_centers_initialized,
                    cluster_counts,
                    cluster_centers_updated,
                    update_in_steps
                };
            }
        }
    }
}
