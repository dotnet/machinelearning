### Download compiler from https://github.com/protocolbuffers/protobuf/releases
Work in command line

```shell
cd tensorflow

set SRC_DIR=D:/Projects/tensorflow
set DST_DIR=D:/Projects/TensorFlow.NET/src/TensorFlowNET.Core/Protobuf

protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow/core/framework/resource_handle.proto
... tensorflow/core/framework/tensor_shape.proto
... tensorflow/core/framework/types.proto
... tensorflow/core/framework/tensor.proto
... tensorflow/core/framework/attr_value.proto
... tensorflow/core/framework/node_def.proto
... tensorflow/core/framework/versions.proto
... tensorflow/core/framework/function.proto
... tensorflow/core/framework/graph.proto
... tensorflow/core/framework/variable.proto
... tensorflow/core/framework/cost_graph.proto
... tensorflow/core/framework/step_stats.proto
... tensorflow/core/framework/allocation_description.proto
... tensorflow/core/framework/tensor_description.proto
... tensorflow/core/framework/api_def.proto
... tensorflow/core/framework/device_attributes.proto
... tensorflow/core/framework/graph_transfer_info.proto
... tensorflow/core/framework/kernel_def.proto
... tensorflow/core/framework/iterator.proto
... tensorflow/core/framework/log_memory.proto
... tensorflow/core/framework/tensor_slice.proto
... tensorflow/core/framework/summary.proto
... tensorflow/core/protobuf/saver.proto
... tensorflow/core/protobuf/meta_graph.proto
... tensorflow/core/protobuf/cluster.proto
... tensorflow/core/protobuf/config.proto
... tensorflow/core/protobuf/debug.proto
... tensorflow/core/protobuf/rewriter_config.proto
... tensorflow/core/protobuf/control_flow.proto
... tensorflow/core/util/event.proto
... tensorflow/python/training/checkpoint_state.proto
```

