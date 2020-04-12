using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal class ParameterAttribute : Attribute
    {
        private IList _value;
        private Type _meta;

        public ParameterAttribute(int min, int max, int step = 1)
        {
            _meta = typeof(int);
            var intList = new List<int>();
            for (var i = min; i <= max; i+=step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            _value = intList;
        }

        public ParameterAttribute(float min, float max, float step = 1f)
        {
            _meta = typeof(float);
            var intList = new List<float>();
            for (var i = min; i <= max; i += step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            _value = intList;
        }

        public ParameterAttribute(int[] candidates)
        {
            _meta = typeof(int);
            _value = candidates.ToList();
        }

        public ParameterAttribute(float[] candidates)
        {
            _meta = typeof(float);
            _value = candidates.ToList();
        }

        public ParameterAttribute(string[] candidates)
        {
            _meta = typeof(string);
            _value = candidates.ToList();
        }

        public IList Value { get => _value; }

        public Type Meta { get => _meta; }

    }
}
