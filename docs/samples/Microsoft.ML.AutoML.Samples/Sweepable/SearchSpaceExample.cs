using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.AutoML.Samples
{
    public static class SearchSpaceExample
    {
        public static void Run()
        {
            // The following code shows how to create a SearchSpace for MyParameter.
            var myParameterSearchSpace = new SearchSpace<MyParameter>();

            // Equivalently, you can also create myParameterSearchSpace from scratch.
            var myParameterSearchSpace2 = new SearchSpace.SearchSpace();

            // numeric options
            myParameterSearchSpace2["IntOption"] = new UniformIntOption(min: -10, max: 10, logBase: false, defaultValue: 0);
            myParameterSearchSpace2["SingleOption"] = new UniformSingleOption(min: 1, max: 10, logBase: true, defaultValue: 1);
            myParameterSearchSpace2["DoubleOption"] = new UniformDoubleOption(min: -10, max: 10, logBase: false, defaultValue: 0);

            // choice options
            myParameterSearchSpace2["BoolOption"] = new ChoiceOption(true, false);
            myParameterSearchSpace2["StrOption"] = new ChoiceOption("a", "b", "c");

            // nest options
            var nestedSearchSpace = new SearchSpace.SearchSpace();
            nestedSearchSpace["IntOption"] = new UniformIntOption(min: -10, max: 10, logBase: false, defaultValue: 0);
            myParameterSearchSpace2["Nest"] = nestedSearchSpace;

            // the two search space should be equal
            Debug.Assert(myParameterSearchSpace.GetHashCode() == myParameterSearchSpace2.GetHashCode());
        }

        public class MyParameter
        {
            [Range((int)-10, 10, 0, false)]
            public int IntOption { get; set; }

            [Range(1f, 10f, 1f, true)]
            public float SingleOption { get; set; }

            [Range(-10, 10, false)]
            public double DoubleOption { get; set; }

            [BooleanChoice]
            public bool BoolOption { get; set; }

            [Choice("a", "b", "c")]
            public string StrOption { get; set; }

            [NestOption]
            public NestParameter Nest { get; set; }
        }

        public class NestParameter
        {
            [Range((int)-10, 10, 0, false)]
            public int IntOption { get; set; }
        }
    }
}
