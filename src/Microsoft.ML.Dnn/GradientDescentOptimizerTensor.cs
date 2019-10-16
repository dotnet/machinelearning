using Tensorflow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Optimizer that implements the gradient descent algorithm.
    /// </summary>
    public class GradientDescentOptimizerTensor : Optimizer
    {
        /// <summary>
        /// Construct a new gradient descent optimizer.
        /// </summary>
        /// <param name="learningRateTensor">A Tensor indicating the learning rate to use.</param>
        /// <param name="useLocking">If true use locks for update operations.</param>
        /// <param name="name">Optional name prefix for the operations created when applying
        /// gradients.Defaults to "GradientDescent".</param>
        /// <remarks>
        /// When eager execution is enabled, `learning_rate` can be a callable that
        /// takes no arguments and returns the actual value to use.This can be useful
        /// for changing these values across different invocations of optimizer
        /// functions.
        /// </remarks>
        public GradientDescentOptimizerTensor(Tensor learningRateTensor, bool useLocking = false, string name = "GradientDescent")
            : base(learningRateTensor, useLocking, name)
        {
            _lr_t = learningRateTensor;
        }

        public override void _prepare()
        {
        }

    }
}