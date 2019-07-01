//Base classes for probability distributions.
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;


namespace Tensorflow
{
    public class _BaseDistribution
    {
        // Abstract base class needed for resolving subclass hierarchy.
    }

    /// <summary>
    /// A generic probability distribution base class.
    /// Distribution is a base class for constructing and organizing properties
    /// (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian). 
    /// </summary>
    public class Distribution : _BaseDistribution
    {
        public TF_DataType _dtype {get;set;}
        //public ReparameterizationType _reparameterization_type {get;set;}
        public bool _validate_args {get;set;}
        public bool _allow_nan_stats {get;set;}
        public Dictionary<string, object> _parameters  {get;set;}
        public List<Tensor> _graph_parents  {get;set;}
        public string _name  {get;set;}


        /// <summary>
        /// Log probability density/mass function.
        /// </summary>
        /// <param name="value"> `Tensor`.</param>
        /// <param name="name"> Python `str` prepended to names of ops created by this function.</param>
        /// <returns>log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type `self.dtype`.</returns>

        
        public Tensor log_prob(Tensor value, string name = "log_prob")
        {
            return _call_log_prob(value, name);
        }

        private Tensor _call_log_prob (Tensor value, string name)
        {
            return with(ops.name_scope(name, "moments", new { value }), scope =>
            {
                try
                {
                    return _log_prob(value);
                }
                catch (Exception e1)
                {
                    try
                    {
                        return math_ops.log(_prob(value));
                    } catch (Exception e2)
                    {
                        throw new NotImplementedException();
                    }
                }
            });
        }

        protected virtual Tensor _log_prob(Tensor value)
        {
            throw new NotImplementedException();
        }

        private Tensor _prob(Tensor value)
        {
            throw new NotImplementedException();
        }

        public TF_DataType dtype()
        {
            return this._dtype;
        }
        

        /// <summary>
        /// Constructs the `Distribution'     
        /// **This is a private method for subclass use.**
        /// </summary>
        /// <param name="dtype"> The type of the event samples. `None` implies no type-enforcement.</param>
        /// <param name="reparameterization_type"> Instance of `ReparameterizationType`.
        /// If `distributions.FULLY_REPARAMETERIZED`, this `Distribution` can be reparameterized
        /// in terms of some standard distribution with a function whose Jacobian is constant for the support 
        /// of the standard distribution. If `distributions.NOT_REPARAMETERIZED`,
        /// then no such reparameterization is available.</param>
        /// <param name="validate_args"> When `True` distribution parameters are checked for validity despite
        /// possibly degrading runtime performance. When `False` invalid inputs silently render incorrect outputs.</param>
        /// <param name="allow_nan_stats"> When `True`, statistics (e.g., mean, mode, variance) use the value "`NaN`" 
        /// to indicate the result is undefined. When `False`, an exception is raised if one or more of the statistic's
        /// batch members are undefined.</param>
        /// <param name = "parameters"> `dict` of parameters used to instantiate this `Distribution`.</param>
        /// <param name = "graph_parents"> `list` of graph prerequisites of this `Distribution`.</param>
        /// <param name = "name"> Name prefixed to Ops created by this class. Default: subclass name.</param>
        /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>

        /*
        private Distribution (
                TF_DataType dtype,
                ReparameterizationType reparameterization_type,
                bool validate_args,
                bool allow_nan_stats,
                Dictionary<object, object> parameters=null,
                List<object> graph_parents=null,
                string name= null)
                {
                    this._dtype = dtype;
                    this._reparameterization_type = reparameterization_type;
                    this._allow_nan_stats = allow_nan_stats;
                    this._validate_args = validate_args;
                    this._parameters = parameters;
                    this._graph_parents = graph_parents;
                    this._name = name;
                }
        */




    }

    /// <summary>
    /// Instances of this class represent how sampling is reparameterized.
    /// Two static instances exist in the distributions library, signifying
    /// one of two possible properties for samples from a distribution:
    /// `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    /// reparameterized, and straight-through gradients are supported.
    /// `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    /// reparameterized, and straight-through gradients are either partially
    /// unsupported or are not supported at all. In this case, for purposes of
    /// e.g. RL or variational inference, it is generally safest to wrap the
    /// sample results in a `stop_gradients` call and use policy
    /// gradients / surrogate loss instead.
    /// </summary>
    class ReparameterizationType
    {
        public string _rep_type { get; set; }
        public ReparameterizationType(string rep_type)
        {
            this._rep_type = rep_type;
        }

        public void repr()
        {
            Console.WriteLine($"<Reparameteriation Type: {this._rep_type}>" );
        }

        public bool eq (ReparameterizationType other)
        {
            return this.Equals(other);
        }
    }


}