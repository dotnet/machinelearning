using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    public class RegisterGradient : Attribute
    {
        public string Name { get; set; }

        public RegisterGradient(string name)
        {
            Name = name;
        }
    }
}
