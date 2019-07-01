using System;
using System.Collections.Generic;
using Tensorflow;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class Normal : Distribution
    {
        public Tensor _loc { get; set; }
        public Tensor _scale { get; set; }

        public Dictionary<string, object> parameters = new Dictionary<string, object>();
        /// <summary>
        /// The Normal distribution with location `loc` and `scale` parameters.
        /// Mathematical details
        /// The probability density function(pdf) is,
        /// '''
        /// pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
        /// Z = (2 pi sigma**2)**0.5
        /// '''
        /// where `loc = mu` is the mean, `scale = sigma` is the std.deviation, and, `Z`
        /// is the normalization constant.
        /// </summary>
        /// <param name="loc"></param>
        /// <param name="scale"></param>
        /// <param name="validate_args"></param>
        /// <param name="allow_nan_stats"></param>
        /// <param name="name"></param>
        public Normal (Tensor loc, Tensor scale, bool validate_args=false, bool allow_nan_stats=true, string name="Normal") 
        {
            parameters.Add("name", name);
            parameters.Add("loc", loc);
            parameters.Add("scale", scale);
            parameters.Add("validate_args", validate_args);
            parameters.Add("allow_nan_stats", allow_nan_stats);

            with(ops.name_scope(name, "", new { loc, scale }), scope => 
            {
                with(ops.control_dependencies(validate_args ? new Operation[] { scale.op} : new Operation[] { }), cd =>
                {
                    this._loc = array_ops.identity(loc, name);
                    this._scale = array_ops.identity(scale, name);
                    base._dtype = this._scale.dtype;
                    // base._reparameterization_type = new ReparameterizationType("FULLY_REPARAMETERIZED");
                    base._validate_args = validate_args;
                    base._allow_nan_stats = allow_nan_stats;
                    base._parameters = parameters;
                    base._graph_parents = new List<Tensor>(new Tensor[] { this._loc, this._scale });
                    base._name = name;
                });

            });
                
        }
        /// <summary>
        /// Distribution parameter for the mean.
        /// </summary>
        /// <returns></returns>
        public Tensor loc()
        {
            return _loc;
        }
        /// <summary>
        /// Distribution parameter for standard deviation."
        /// </summary>
        /// <returns></returns>
        public Tensor scale()
        {
            return _scale;
        }

        public Tensor _batch_shape_tensor()
        {
            return array_ops.broadcast_dynamic_shape(array_ops.shape(_loc), array_ops.shape(_scale));
        }

        public Tensor _batch_shape()
        {
            return array_ops.broadcast_static_shape(new Tensor(_loc.shape), new Tensor(_scale.shape));
        }

        protected override Tensor _log_prob(Tensor x)
        {
            var log_prob = _log_unnormalized_prob(x);
            var log_norm = _log_normalization();
            return tf.sub(log_prob, log_norm);
        }

        private Tensor _log_unnormalized_prob (Tensor x)
        {
            return -0.5 * math_ops.square(_z(x));
        }
        /// <summary>
        /// Standardize input `x` to a unit normal.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private Tensor _z (Tensor x)
        {
            return tf.divide(tf.sub(x, this._loc), this._scale);
        }

        private Tensor _log_normalization()
        {
            Tensor t1 = ops.convert_to_tensor(Math.Log(2.0 * Math.PI), TF_DataType.TF_FLOAT);
            Tensor t2 = tf.multiply(ops.convert_to_tensor(0.5, TF_DataType.TF_FLOAT), t1);
            return  tf.add(t2, math_ops.log(this._scale));
        }
    }
}