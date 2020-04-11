using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal abstract class OptionBuilder<TOption>
        where TOption: class
    {
        public TOption CreateDefaultOption()
        {
            var assem = typeof(TOption).Assembly;
            return assem.CreateInstance(typeof(TOption).FullName) as TOption;
        }

        public TOption BuildOption(SweeperOutput input)
        {
            var option = CreateDefaultOption();
            foreach(var kv in input)
            {
                var value = kv.Value;
                typeof(TOption).GetProperty(kv.Key)?.SetValue(option, value);
            }

            return option;
        }

        public ISweeper BuildRandomSweeper(int maximum = 100)
        {
            var paramaters = this.GetType().GetProperties()
                                 .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null);

            var paramatersDictionary = new Dictionary<string, ParameterAttribute>();
            foreach( var param in paramaters)
            {
                paramatersDictionary.Add(param.Name, Attribute.GetCustomAttribute(param, typeof(ParameterAttribute)) as ParameterAttribute);
            }

            return new RandomSweeper(paramatersDictionary, maximum);
        }
    }
}
