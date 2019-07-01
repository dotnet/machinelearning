using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow.Layers
{
    public class Layer : Keras.Layers.Layer
    {
        protected Graph _graph;
        
        protected VariableScope _scope;
        protected VariableScope _current_scope;
        
        protected bool? _reuse;
        protected bool _use_resource_variables;
        protected bool _keras_style;

        public Layer(bool trainable = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool? _reuse = null) : base(trainable: trainable, name: name, dtype: dtype)
        {
            // For backwards compatibility, legacy layers do not use `ResourceVariable`
            // by default.
            this._use_resource_variables = false;
            this._reuse = _reuse;

            // Avoid an incorrect lint error
            _trainable_weights = new List<RefVariable>();
            this.built = false;
            _keras_style = false;
        }

        public virtual Tensor apply(Tensor inputs, Tensor training = null)
        {
            return __call__(inputs, training: training);
        }

        public Tensor __call__(Tensor inputs,
            Tensor training = null,
            VariableScope scope = null)
        {
            _set_scope(scope);
            _graph = ops._get_graph_from_inputs(new Tensor[] { inputs }, graph: _graph);

            variable_scope scope_context_manager = null;
            if (built)
            {

            }
            else
            {
                scope_context_manager = tf.variable_scope(_scope,
                    reuse: _reuse,
                    auxiliary_name_scope: false);
            }

            Tensor outputs = null;
            with(scope_context_manager, scope2 =>
            {
                _current_scope = scope2;
                // Actually call layer
                outputs = base.__call__(new Tensor[] { inputs }, training: training);
            });


            // Update global default collections.
            _add_elements_to_collection(_updates.ToArray(), new string[] { ops.GraphKeys.UPDATE_OPS });

            return outputs;
        }

        protected virtual void _add_elements_to_collection(Operation[] elements, string[] collection_list)
        {
            foreach(var name in collection_list)
            {
                var collection = ops.get_collection_ref(name) as List<object>;

                foreach (var element in elements)
                    if (!collection.Contains(element))
                        collection.Add(element);
            }
        }

        /// <summary>
        /// Adds a new variable to the layer, or gets an existing one; returns it.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="initializer"></param>
        /// <param name="trainable"></param>
        /// <param name="synchronization"></param>
        /// <param name="aggregation"></param>
        /// <returns></returns>
        protected virtual RefVariable add_weight(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            var default_graph = ops.get_default_graph();
            Graph init_graph = null;
            VariableV1[] existing_variables = null;

            if (default_graph.building_function)
            {
                throw new NotImplementedException("add_weight");
            }
            else
            {
                init_graph = default_graph;
                existing_variables = variables.global_variables().ToArray();
            }

            if(dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            _set_scope();
            var reuse = built || (_reuse != null && _reuse.Value);
            return with(tf.variable_scope(_scope,
                reuse: reuse,
                auxiliary_name_scope: false), scope =>
                {
                    _current_scope = scope;
                    return with(ops.name_scope(_name_scope()), delegate
                    {
                        var variable = base.add_weight(name,
                            shape,
                            dtype: dtype,
                            initializer: initializer,
                            trainable: trainable,
                            getter: (name1, shape1, dtype1, initializer1, trainable1) =>
                                tf.get_variable(name1,
                                    shape: new TensorShape(shape1),
                                    dtype: dtype1,
                                    initializer: initializer1,
                                    trainable: trainable1)
                            );

                        //if (init_graph != null)
                            //var trainable_variables = variables.trainable_variables();
                        
                        return variable;
                    });
                });
        }



        protected override string _name_scope()
        {
            return _current_scope.original_name_scope;
        }

        private void _set_scope(VariableScope scope = null)
        {
            if (_scope == null)
            {
                if(_reuse.HasValue && _reuse.Value)
                {
                    throw new NotImplementedException("_set_scope _reuse.HasValue");
                    /*with(tf.variable_scope(scope == null ? _base_name : scope),
                        captured_scope => _scope = captured_scope);*/
                }
                else
                {
                    with(tf.variable_scope(scope, default_name: _base_name), captured_scope =>
                    {
                        // convert variable_scope to VariableScope
                        _scope = captured_scope;
                    });
                }
            }
        }
    }
}
