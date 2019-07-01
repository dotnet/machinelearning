using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Initializers;

namespace Tensorflow.Keras
{
    public class Initializers
    {
        /// <summary>
        /// He normal initializer.
        /// </summary>
        /// <param name="seed"></param>
        /// <returns></returns>
        public IInitializer he_normal(int? seed = null)
        {
            return new VarianceScaling(scale: 2.0f, mode: "fan_in", distribution: "truncated_normal", seed: seed);
        }
    }
}
