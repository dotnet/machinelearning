using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IPyClass
    {
        /// <summary>
        /// Called when the instance is created.
        /// </summary>
        /// <param name="args"></param>
        void __init__(IPyClass self, dynamic args);

        void __enter__(IPyClass self);

        void __exit__(IPyClass self);

        void __del__(IPyClass self);
    }
}
