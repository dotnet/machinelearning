using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    /// <summary>
    /// Variable scope object to carry defaults to provide to `get_variable`
    /// </summary>
    public class VariableScope
    {
        public bool use_resource { get; set; }
        private _ReuseMode _reuse;
        public bool resue;

        private TF_DataType _dtype;
        string _name;
        public string name => _name;
        public string _name_scope { get; set; }
        public string original_name_scope => _name_scope;

        public VariableScope(bool reuse, 
            string name = "", 
            string name_scope = "",
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            _name = name;
            _name_scope = name_scope;
            _reuse = _ReuseMode.AUTO_REUSE;
            _dtype = dtype;
        }

        public RefVariable get_variable(_VariableStore var_store, 
            string name, 
            TensorShape shape = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            object initializer = null, // IInitializer or Tensor
            bool? trainable = null,
            bool? use_resource = null,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation= VariableAggregation.None)
        {
            string full_name = !string.IsNullOrEmpty(this.name) ? this.name + "/" + name : name;
            return with(ops.name_scope(null), scope =>
            {
                if (dtype == TF_DataType.DtInvalid)
                    dtype = _dtype;

                return var_store.get_variable(full_name, 
                    shape: shape, 
                    dtype: dtype,
                    initializer: initializer,
                    reuse: resue,
                    trainable: trainable,
                    synchronization: synchronization,
                    aggregation: aggregation);
            });
        }
    }
}
