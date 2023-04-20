using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML.Samples
{
    public static class ParameterExample
    {
        public static void Run()
        {
            // Parameter is essentially a wrapper class over Json.
            // Therefore it supports all json types, like integar, number, boolearn, string, etc..

            // To create parameter over existing value, use Parameter.From
            var intParam = Parameter.FromInt(10);
            var doubleParam = Parameter.FromDouble(20);
            var boolParam = Parameter.FromBool(false);

            // To cast parameter to specific type, use Parameter.AsType
            // NOTE: Casting to a wrong type will trigger an argumentException.
            var i = intParam.AsType<int>(); // i == 10
            var d = doubleParam.AsType<double>(); // d == 20
            var b = boolParam.AsType<bool>(); // b == false
        }
    }
}
