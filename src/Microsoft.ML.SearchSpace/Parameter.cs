// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace Microsoft.ML.SearchSpace
{
    public class Parameter : IParameter
    {
        private readonly JsonSerializerSettings _settings = new JsonSerializerSettings()
        {
            Formatting = Formatting.Indented,
            Culture = System.Globalization.CultureInfo.InvariantCulture,
            NullValueHandling = NullValueHandling.Ignore,
            Converters = new JsonConverter[]
            {
                new StringEnumConverter(),
            },
        };

        private readonly object _value;

        private Parameter(object value, ParameterType type)
        {
            this._value = value;
            this.ParameterType = type;
        }

        public static Parameter FromDouble(double value)
        {
            return new Parameter(value, ParameterType.Float);
        }

        public static Parameter FromFloat(float value)
        {
            return new Parameter(value, ParameterType.Float);
        }

        public static Parameter FromLong(long value)
        {
            return new Parameter(value, ParameterType.Integer);
        }

        public static Parameter FromInt(int value)
        {
            return new Parameter(value, ParameterType.Integer);
        }

        public static Parameter FromString(string value)
        {
            return new Parameter(value, ParameterType.String);
        }

        public static Parameter FromBool(bool value)
        {
            return new Parameter(value, ParameterType.Bool);
        }

        public static Parameter FromEnum<T>(T value) where T : struct, Enum
        {
            return Parameter.FromEnum(value, typeof(T));
        }

        public static Parameter FromIEnumerable<T>(IEnumerable<T> values)
        {
            // check T
            return Parameter.FromIEnumerable(values as IEnumerable);
        }

        private static Parameter FromIEnumerable(IEnumerable values)
        {
            return new Parameter(values, ParameterType.Array);
        }

        private static Parameter FromEnum(Enum e, Type t)
        {
            return Parameter.FromString(Enum.GetName(t, e));
        }

        public static Parameter FromObject<T>(T value) where T : class
        {
            return Parameter.FromObject(value, typeof(T));
        }

        private static Parameter FromObject(object value, Type type)
        {
            var param = value switch
            {
                int i => Parameter.FromInt(i),
                long l => Parameter.FromLong(l),
                double d => Parameter.FromDouble(d),
                float f => Parameter.FromFloat(f),
                string s => Parameter.FromString(s),
                bool b => Parameter.FromBool(b),
                IEnumerable vs => Parameter.FromIEnumerable(vs),
                Enum e => Parameter.FromEnum(e, e.GetType()),
                _ => null,
            };

            if (param != null)
            {
                return param;
            }
            else
            {
                var parameter = Parameter.CreateNestedParameter();
                var properties = type.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
                        .Where(p => p.CanRead && p.CanWrite);
                foreach (var property in properties)
                {
                    var name = property.Name;
                    var pValue = property.GetValue(value);
                    if (pValue != null)
                    {
                        var parameter = Parameter.FromObject(pValue, property.PropertyType);

                        if (parameter?.Count != 0)
                        {
                            parameter[name] = parameter;
                        }
                    }
                }

                return parameter;
            }
        }

        public static Parameter CreateNestedParameter(params KeyValuePair<string, IParameter>[] parameters)
        {
            var parameter = new Parameter(new Dictionary<string, IParameter>(), ParameterType.Object);
            foreach (var param in parameters)
            {
                parameter[param.Key] = param.Value;
            }

            return parameter;
        }

        public object Value { get => this._value; }

        public int Count => this.ParameterType == ParameterType.Object ? (this._value as Dictionary<string, IParameter>).Count : 1;

        public bool IsReadOnly
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, IParameter>).IsReadOnly;
            }
        }

        public ParameterType ParameterType { get; }

        ICollection<IParameter> IDictionary<string, IParameter>.Values
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, IParameter>).Values;
            }
        }

        public ICollection<string> Keys
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, IParameter>).Keys;
            }
        }

        public IParameter this[string key]
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, IParameter>)[key];
            }

            set
            {
                this.VerifyIfParameterIsObjectType();
                (this._value as IDictionary<string, IParameter>)[key] = value;
            }
        }

        public T AsType<T>()
        {
            if (this._value is T t)
            {
                return t;
            }
            else
            {
                var json = JsonConvert.SerializeObject(this._value, this._settings);
                return JsonConvert.DeserializeObject<T>(json, this._settings);
            }
        }

        public void Clear()
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, IParameter>).Clear();
        }

        public void Add(string key, IParameter value)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, IParameter>).Add(key, value);
        }

        public bool TryGetValue(string key, out IParameter value)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as Dictionary<string, IParameter>).TryGetValue(key, out value);
        }

        public void Add(KeyValuePair<string, IParameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, IParameter>).Add(item.Key, item.Value);
        }

        public bool Contains(KeyValuePair<string, IParameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as Dictionary<string, IParameter>).Contains(item);
        }

        public bool Remove(KeyValuePair<string, IParameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, IParameter>).Remove(item);
        }

        IEnumerator<KeyValuePair<string, IParameter>> IEnumerable<KeyValuePair<string, IParameter>>.GetEnumerator()
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, IParameter>).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, IParameter>).GetEnumerator();
        }

        private void VerifyIfParameterIsObjectType()
        {
            Contracts.Check(this.ParameterType == ParameterType.Object, "parameter is not object type.");
        }

        public void CopyTo(KeyValuePair<string, IParameter>[] array, int arrayIndex)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as IDictionary<string, IParameter>).CopyTo(array, arrayIndex);
        }

        public bool ContainsKey(string key)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, IParameter>).ContainsKey(key);
        }

        public bool Remove(string key)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, IParameter>).Remove(key);
        }
    }
}
