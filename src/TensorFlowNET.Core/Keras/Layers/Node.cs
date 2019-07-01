using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// A `Node` describes the connectivity between two layers.
    /// </summary>
    public class Node
    {
        public InputLayer outbound_layer;
        public Layer[] inbound_layers;
        public int[] node_indices;
        public int[] tensor_indices;
        public Tensor[] input_tensors;
        public Tensor[] output_tensors;
        public int[][] input_shapes;
        public int[][] output_shapes;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="outbound_layer">
        /// the layer that takes
        /// `input_tensors` and turns them into `output_tensors`
        /// (the node gets created when the `call`
        /// method of the layer was called).
        /// </param>
        /// <param name="inbound_layers">
        /// a list of layers, the same length as `input_tensors`,
        /// the layers from where `input_tensors` originate.
        /// </param>
        /// <param name="node_indices">
        /// a list of integers, the same length as `inbound_layers`.
        /// `node_indices[i]` is the origin node of `input_tensors[i]`
        /// (necessary since each inbound layer might have several nodes,
        /// e.g. if the layer is being shared with a different data stream).
        /// </param>
        /// <param name="tensor_indices"></param>
        /// <param name="input_tensors">list of input tensors.</param>
        /// <param name="output_tensors">list of output tensors.</param>
        public Node(InputLayer outbound_layer, 
            Layer[] inbound_layers, 
            int[] node_indices,
            int[] tensor_indices,
            Tensor[] input_tensors,
            Tensor[] output_tensors)
        {
            this.outbound_layer = outbound_layer;
            this.inbound_layers = inbound_layers;
            this.node_indices = node_indices;
            this.tensor_indices = tensor_indices;
            this.input_tensors = input_tensors;
            this.output_tensors = output_tensors;

            input_shapes = input_tensors.Select(x => x._shape_tuple()).ToArray();
            output_shapes = output_tensors.Select(x => x._shape_tuple()).ToArray();

            // Add nodes to all layers involved.
            foreach (var layer in inbound_layers)
            {
                if (layer != null)
                    layer.outbound_nodes.Add(this);
            }

            outbound_layer.inbound_nodes.Add(this);
        }
    }
}
