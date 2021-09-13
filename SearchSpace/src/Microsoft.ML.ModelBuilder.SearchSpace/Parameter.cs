// <copyright file="Parameter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Converter;
using Newtonsoft.Json;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    // TODO
    // Add tests
    [JsonConverter(typeof(ParameterConverter))]
    public class Parameter : Dictionary<string, Parameter>
    {
        public Parameter(object value)
        {
            this.Value = value;
        }

        public Parameter()
            : base()
        {
        }

        public object Value { get; }

        public T AsType<T>()
        {
            try
            {
                if (this.Value != null)
                {
                    return (T)this.Value;
                }
                else
                {
                    var json = JsonConvert.SerializeObject(this);
                    return JsonConvert.DeserializeObject<T>(json);
                }
            }
            catch (InvalidCastException)
            {
                var json = JsonConvert.SerializeObject(this);
                return JsonConvert.DeserializeObject<T>(json);
                throw;
            }
        }

        internal static Parameter CreateFromInstance<T>(T instance)
        {
            return CreateFromInstance(instance, typeof(T));
        }

        private static Parameter CreateFromInstance(object instance, Type instanceType)
        {
            var res = new Parameter();

            // handle property
            var props = instanceType.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            foreach (var prop in props)
            {
                var attributes = prop.GetCustomAttributes(false);
                foreach (var attr in attributes)
                {
                    if(attr is ChoiceAttribute || attr is RangeAttribute)
                    {
                        res.Add(prop.Name, new Parameter(prop.GetValue(instance)));
                        break;
                    }
                    else if (attr is OptionAttribute)
                    {
                        res.Add(prop.Name, CreateFromInstance(prop.GetValue(instance), prop.PropertyType));
                        break;
                    }
                    else
                    {
                        continue;
                    }
                }
            }

            // field
            var fields = instanceType.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            foreach (var field in fields)
            {
                var attributes = field.GetCustomAttributes(false);
                foreach (var attr in attributes)
                {
                    if (attr is ChoiceAttribute || attr is RangeAttribute)
                    {
                        res.Add(field.Name, new Parameter(field.GetValue(instance)));
                        break;
                    }
                    else if (attr is OptionAttribute)
                    {
                        res.Add(field.Name, CreateFromInstance(field.GetValue(instance), field.FieldType));
                        break;
                    }
                    else
                    {
                        continue;
                    }
                }
            }

            return res;
        }
    }
}
