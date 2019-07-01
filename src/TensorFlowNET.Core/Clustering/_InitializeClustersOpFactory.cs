using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow.Clustering
{
    /// <summary>
    /// Internal class to create the op to initialize the clusters.
    /// </summary>
    public class _InitializeClustersOpFactory
    {
        Tensor[] _inputs;
        Tensor _num_clusters;
        string _initial_clusters;
        string _distance_metric;
        int _random_seed;
        int _kmeans_plus_plus_num_retries;
        int _kmc2_chain_length;
        RefVariable _cluster_centers;
        RefVariable _cluster_centers_updated;
        RefVariable _cluster_centers_initialized;
        Tensor _num_selected;
        Tensor _num_remaining;
        Tensor _num_data;

        public _InitializeClustersOpFactory(Tensor[] inputs,
            Tensor num_clusters,
            string initial_clusters,
            string distance_metric,
            int random_seed,
            int kmeans_plus_plus_num_retries,
            int kmc2_chain_length,
            RefVariable cluster_centers,
            RefVariable cluster_centers_updated,
            RefVariable cluster_centers_initialized)
        {
            _inputs = inputs;
            _num_clusters = num_clusters;
            _initial_clusters = initial_clusters;
            _distance_metric = distance_metric;
            _random_seed = random_seed;
            _kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries;
            _kmc2_chain_length = kmc2_chain_length;
            _cluster_centers = cluster_centers;
            _cluster_centers_updated = cluster_centers_updated;
            _cluster_centers_initialized = cluster_centers_initialized;

            _num_selected = array_ops.shape(_cluster_centers).slice(0);
            _num_remaining = _num_clusters - _num_selected;

            _num_data = math_ops.add_n(_inputs.Select(i => array_ops.shape(i).slice(0)).ToArray());
        }

        private Tensor _initialize()
        {
            return with(ops.control_dependencies(new Operation[]
            {
                check_ops.assert_positive(_num_remaining)
            }), delegate
            {
                var num_now_remaining = _add_new_centers();
                return control_flow_ops.cond(math_ops.equal(num_now_remaining, 0),
                  () =>
                  {
                      return state_ops.assign(_cluster_centers_initialized, true);
                  },
                  () =>
                  {
                      return control_flow_ops.no_op().output.slice(0);
                  });
            });
        }

        public Tensor op()
        {
            var x = control_flow_ops.cond(gen_math_ops.equal(_num_remaining, 0),
                () => 
                {
                    return check_ops.assert_equal(_cluster_centers_initialized, true);
                },
                _initialize);

            return x;
        }

        private Tensor _add_new_centers()
        {
            // Adds some centers and returns the number of centers remaining.
            var new_centers = _choose_initial_centers();
            if (_distance_metric == KMeans.COSINE_DISTANCE)
                new_centers = nn_impl.l2_normalize(new_centers.slice(0), axis: 1);

            // If cluster_centers is empty, it doesn't have the right shape for concat.
            var all_centers = control_flow_ops.cond(math_ops.equal(_num_selected, 0),
                () => new Tensor[] { new_centers },
                () => new Tensor[] { array_ops.concat(new Tensor[] { _cluster_centers, new_centers }, 0) });

            var a = state_ops.assign(_cluster_centers, all_centers, validate_shape: false);

            return _num_clusters - array_ops.shape(a).slice(0);
        }

        private Tensor _choose_initial_centers()
        {
            return _greedy_batch_sampler().slice(0);
        }

        private Tensor _greedy_batch_sampler()
        {
            return control_flow_ops.cond(_num_data <= _num_remaining,
                () =>
                {
                    return array_ops.concat(_inputs, 0);
                },
                () =>
                {
                    return _random();
                });
        }

        private Tensor _random()
        {
            var reshape = array_ops.reshape(_num_remaining, new int[] { -1 });
            var cast = math_ops.cast(_num_data, TF_DataType.TF_INT64);
            var indices = random_ops.random_uniform(
                reshape,
                minval: 0,
                maxval: cast,
                seed: _random_seed,
                dtype: TF_DataType.TF_INT64);
            return embedding_ops.embedding_lookup(_inputs, indices, partition_strategy: "div");
        }
    }
}
