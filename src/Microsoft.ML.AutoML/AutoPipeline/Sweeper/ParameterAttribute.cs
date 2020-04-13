using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Sweeper;

namespace Microsoft.ML.AutoPipeline
{
    internal class ParameterAttribute : Attribute
    {
        private IList _value;
        private Type _meta;

        public ParameterAttribute(string name, int min, int max, int step = 1)
        {
            _meta = typeof(int);
            var intList = new List<int>();
            for (var i = min; i <= max; i+=step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            _value = intList;

            var option = new LongParamOptions()
            {
                Name = name,
                Min = min,
                Max = max,
                StepSize = step,
            };

            ValueGenerator = new LongValueGenerator(option);
        }

        public ParameterAttribute(string name, float min, float max, float step = 1f)
        {
            _meta = typeof(float);
            var intList = new List<float>();
            for (var i = min; i <= max; i += step)
            {
                intList.Add(i);
            }

            intList.Add(max);
            _value = intList;

            var option = new FloatParamOptions()
            {
                Name = name,
                Min = min,
                Max = max,
                StepSize = step,
            };

            ValueGenerator = new FloatValueGenerator(option);
        }

        public ParameterAttribute(string name, string[] candidates)
        {
            _meta = typeof(string);
            _value = candidates.ToList();

            var option = new DiscreteParamOptions()
            {
                Name = name,
                Values = candidates
            };

            ValueGenerator = new DiscreteValueGenerator(option);
        }

        public IList Value { get => _value; }

        public Type Meta { get => _meta; }

        public Microsoft.ML.IValueGenerator ValueGenerator { get; }

    }
}
