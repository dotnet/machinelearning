// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace.Converter;

namespace Microsoft.ML.SearchSpace
{
    public enum ParameterType
    {
        Integer = 0,
        Float = 1,
        Bool = 2,
        String = 3,
        Object = 4,
        Array = 5,
    }

    [JsonConverter(typeof(ParameterConverter))]
    public class Parameter : IDictionary<string, Parameter>
    {
        private readonly JsonSerializerOptions _settings = new JsonSerializerOptions()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        private readonly object _value;

        private Parameter(object value, ParameterType type)
        {
            this._value = value;
            this.ParameterType = type;
            this._settings.Converters.Add(new JsonStringEnumConverter());
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
                        var p = Parameter.FromObject(pValue, property.PropertyType);

                        if (p?.Count != 0)
                        {
                            parameter[name] = p;
                        }
                    }
                }

                return parameter;
            }
        }

        public static Parameter CreateNestedParameter(params KeyValuePair<string, Parameter>[] parameters)
        {
            var parameter = new Parameter(new Dictionary<string, Parameter>(), ParameterType.Object);
            foreach (var param in parameters)
            {
                parameter[param.Key] = param.Value;
            }

            return parameter;
        }

        public object Value { get => this._value; }

        public int Count => this.ParameterType == ParameterType.Object ? (this._value as Dictionary<string, Parameter>).Count : 1;

        public bool IsReadOnly
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, Parameter>)?.IsReadOnly ?? false;
            }
        }

        public ParameterType ParameterType { get; }

        ICollection<Parameter> IDictionary<string, Parameter>.Values
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, Parameter>).Values;
            }
        }

        public ICollection<string> Keys
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, Parameter>).Keys;
            }
        }

        public Parameter this[string key]
        {
            get
            {
                this.VerifyIfParameterIsObjectType();
                return (this._value as IDictionary<string, Parameter>)[key];
            }

            set
            {
                this.VerifyIfParameterIsObjectType();
                (this._value as IDictionary<string, Parameter>)[key] = value;
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
                var json = JsonSerializer.Serialize(this._value, this._settings);
                return JsonSerializer.Deserialize<T>(json, this._settings);
            }
        }

        public void Clear()
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, Parameter>).Clear();
        }

        public void Add(string key, Parameter value)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, Parameter>).Add(key, value);
        }

        public bool TryGetValue(string key, out Parameter value)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as Dictionary<string, Parameter>).TryGetValue(key, out value);
        }

        public void Add(KeyValuePair<string, Parameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as Dictionary<string, Parameter>).Add(item.Key, item.Value);
        }

        public bool Contains(KeyValuePair<string, Parameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as Dictionary<string, Parameter>).Contains(item);
        }

        public bool Remove(KeyValuePair<string, Parameter> item)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, Parameter>).Remove(item);
        }

        IEnumerator<KeyValuePair<string, Parameter>> IEnumerable<KeyValuePair<string, Parameter>>.GetEnumerator()
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, Parameter>).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, Parameter>).GetEnumerator();
        }

        private void VerifyIfParameterIsObjectType()
        {
            Contracts.Check(this.ParameterType == ParameterType.Object, "parameter is not object type.");
        }

        public void CopyTo(KeyValuePair<string, Parameter>[] array, int arrayIndex)
        {
            this.VerifyIfParameterIsObjectType();
            (this._value as IDictionary<string, Parameter>).CopyTo(array, arrayIndex);
        }

        public bool ContainsKey(string key)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, Parameter>).ContainsKey(key);
        }

        public bool Remove(string key)
        {
            this.VerifyIfParameterIsObjectType();
            return (this._value as IDictionary<string, Parameter>).Remove(key);
        }
    }
}
