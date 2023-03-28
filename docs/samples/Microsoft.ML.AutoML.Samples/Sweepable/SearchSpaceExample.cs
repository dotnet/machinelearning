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
            myParameterSearchSpace2["IntOption"] = new UniformIntOption(-10, 10, false, 0);
            myParameterSearchSpace2["SingleOption"] = new UniformSingleOption(1, 10, true, 1);
            myParameterSearchSpace2["DoubleOption"] = new UniformDoubleOption(-10, 10, false, 0);

            // choice options
            myParameterSearchSpace2["BoolOption"] = new ChoiceOption(true, false);
            myParameterSearchSpace2["StrOption"] = new ChoiceOption("a", "b", "c");

            // nest options
            var nestedSearchSpace = new SearchSpace.SearchSpace();
            nestedSearchSpace["IntOption"] = new UniformIntOption(-10, 10, false, 0);
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
