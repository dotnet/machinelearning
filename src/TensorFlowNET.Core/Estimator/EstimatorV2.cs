using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Data;

namespace Tensorflow.Estimator
{
    /// <summary>
    /// Estimator class to train and evaluate TensorFlow models.
    /// <see cref="tensorflow_estimator\python\estimator\estimator.py"/>
    /// </summary>
    public class EstimatorV2 : IEstimator
    {
        public EstimatorV2(string model_dir = null)
        {

        }

        /// <summary>
        /// Calls the input function.
        /// </summary>
        /// <param name="mode"></param>
        public void call_input_fn(string mode = null)
        {

        }

        public void train_model_default(Func<string, string, HyperParams, bool, DatasetV1Adapter> input_fn)
        {

        }

        public void get_features_and_labels_from_input_fn()
        {

        }
    }
}
