using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal abstract class OptionBuilder<TOption>
        where TOption: class
    {

        public Dictionary<string, ParameterAttribute> ParameterAttributes { get => this.GetParameterAttributes(); }

        public TOption CreateDefaultOption()
        {
            var assem = typeof(TOption).Assembly;
            var option = assem.CreateInstance(typeof(TOption).FullName) as TOption;

            // set up fields
            var fields = this.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            foreach(var field in fields)
            {
                var value = field.GetValue(this);
                option.GetType().GetField(field.Name)?.SetValue(option, value);
            }

            return option;
        }

        public TOption BuildOption(SweeperOutput input)
        {
            var option = CreateDefaultOption();
            foreach(var kv in input)
            {
                var value = kv.Value;
                typeof(TOption).GetField(kv.Key)?.SetValue(option, value);
            }

            return option;
        }

        private Dictionary<string, ParameterAttribute> GetParameterAttributes()
        {
            var paramaters = this.GetType().GetFields()
                     .Where(x => Attribute.GetCustomAttribute(x, typeof(ParameterAttribute)) != null);

            var paramatersDictionary = new Dictionary<string, ParameterAttribute>();
            foreach (var param in paramaters)
            {
                paramatersDictionary.Add(param.Name, Attribute.GetCustomAttribute(param, typeof(ParameterAttribute)) as ParameterAttribute);
            }

            return paramatersDictionary;
        }
    }
}
