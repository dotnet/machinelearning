using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;

namespace Tensorflow
{
    public static partial class keras
    {
        public static Preprocessing preprocessing => new Preprocessing();
        public static Sequence sequence = new Sequence();
        public static Sequential Sequential() => new Sequential();
    }
}
