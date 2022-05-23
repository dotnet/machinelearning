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
    /// <summary>
    /// Parameter type. This type is used to determine the type of <see cref="Parameter"/> and is associated to corresponded Json token when serializing/deserializing.
    /// </summary>
    public enum ParameterType
    {
        /// <summary>
        /// Json int type.
        /// </summary>
        Integer = 0,

        /// <summary>
        /// Json number type.
        /// </summary>
        Number = 1,

        /// <summary>
        /// Json boolean type.
        /// </summary>
        Bool = 2,

        /// <summary>
        /// Json string type.
        /// </summary>
        String = 3,

        /// <summary>
        /// Json object type.
        /// </summary>
        Object = 4,

        /// <summary>
        /// Json array type.
        /// </summary>
        Array = 5,
    }

    /// <summary>
    /// <see cref="Parameter"/> is used to save sweeping result from tuner and is used to restore mlnet pipeline from sweepable pipline.
    /// </summary>
    [JsonConverter(typeof(ParameterConverter))]
    public sealed class Parameter : IDictionary<string, Parameter>, IEquatable<Parameter>
    {
        private readonly JsonSerializerOptions _settings = new JsonSerializerOptions()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        private readonly object _value;

        private Parameter(object value, ParameterType type)
        {
            _value = value;
            ParameterType = type;
            _settings.Converters.Add(new JsonStringEnumConverter());
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="double"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Number"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromDouble(double value)
        {
            return new Parameter(value, ParameterType.Number);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="float"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Number"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromFloat(float value)
        {
            return new Parameter(value, ParameterType.Number);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="long"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Integer"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromLong(long value)
        {
            return new Parameter(value, ParameterType.Integer);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="int"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Integer"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromInt(int value)
        {
            return new Parameter(value, ParameterType.Integer);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="string"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.String"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromString(string value)
        {
            return new Parameter(value, ParameterType.String);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="bool"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Bool"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromBool(bool value)
        {
            return new Parameter(value, ParameterType.Bool);
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="Enum"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.String"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter FromEnum<T>(T value) where T : struct, Enum
        {
            return Parameter.FromEnum(value, typeof(T));
        }

        /// <summary>
        /// Create a <see cref="Parameter"/> from a <see cref="IEnumerable"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Array"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
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

        /// <summary>
        /// Create a <see cref="Parameter"/> from an <see cref="object"/> value. The <see cref="ParameterType"/> will be <see cref="ParameterType.Object"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
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

        /// <summary>
        /// Create a <see cref="Parameter"/> from <paramref name="parameters"/>. The <see cref="ParameterType"/> will be <see cref="ParameterType.Object"/>.
        /// </summary>
        /// <returns><see cref="Parameter"/></returns>
        public static Parameter CreateNestedParameter(params KeyValuePair<string, Parameter>[] parameters)
        {
            var parameter = new Parameter(new Dictionary<string, Parameter>(), ParameterType.Object);
            foreach (var param in parameters)
            {
                parameter[param.Key] = param.Value;
            }

            return parameter;
        }

        internal object Value { get => _value; }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public int Count => ParameterType == ParameterType.Object ? (_value as Dictionary<string, Parameter>).Count : 1;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool IsReadOnly
        {
            get
            {
                VerifyIfParameterIsObjectType();
                return (_value as IDictionary<string, Parameter>)?.IsReadOnly ?? false;
            }
        }

        /// <summary>
        /// Get <see cref="ParameterType"/> of this <see cref="ParameterType"/>
        /// </summary>
        public ParameterType ParameterType { get; }

        ICollection<Parameter> IDictionary<string, Parameter>.Values
        {
            get
            {
                VerifyIfParameterIsObjectType();
                return (_value as IDictionary<string, Parameter>).Values;
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public ICollection<string> Keys
        {
            get
            {
                VerifyIfParameterIsObjectType();
                return (_value as IDictionary<string, Parameter>).Keys;
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public Parameter this[string key]
        {
            get
            {
                VerifyIfParameterIsObjectType();
                return (_value as IDictionary<string, Parameter>)[key];
            }

            set
            {
                VerifyIfParameterIsObjectType();
                (_value as IDictionary<string, Parameter>)[key] = value;
            }
        }

        /// <summary>
        /// Cast <see cref="ParameterType"/> to <typeparamref name="T"/>. This method will return immediately if the underlying value is of type <typeparamref name="T"/>, otherwise it uses <see cref="JsonSerializer"/> to
        /// convert its value to <typeparamref name="T"/>.
        /// </summary>
        public T AsType<T>()
        {
            if (_value is T t)
            {
                return t;
            }
            else
            {
                var json = JsonSerializer.Serialize(_value, _settings);
                return JsonSerializer.Deserialize<T>(json, _settings);
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void Clear()
        {
            VerifyIfParameterIsObjectType();
            (_value as Dictionary<string, Parameter>).Clear();
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void Add(string key, Parameter value)
        {
            VerifyIfParameterIsObjectType();
            (_value as Dictionary<string, Parameter>).Add(key, value);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool TryGetValue(string key, out Parameter value)
        {
            VerifyIfParameterIsObjectType();
            return (_value as Dictionary<string, Parameter>).TryGetValue(key, out value);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void Add(KeyValuePair<string, Parameter> item)
        {
            VerifyIfParameterIsObjectType();
            (_value as Dictionary<string, Parameter>).Add(item.Key, item.Value);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Contains(KeyValuePair<string, Parameter> item)
        {
            VerifyIfParameterIsObjectType();
            return (_value as Dictionary<string, Parameter>).Contains(item);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Remove(KeyValuePair<string, Parameter> item)
        {
            VerifyIfParameterIsObjectType();
            return (_value as IDictionary<string, Parameter>).Remove(item);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        IEnumerator<KeyValuePair<string, Parameter>> IEnumerable<KeyValuePair<string, Parameter>>.GetEnumerator()
        {
            VerifyIfParameterIsObjectType();
            return (_value as IDictionary<string, Parameter>).GetEnumerator();
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        IEnumerator IEnumerable.GetEnumerator()
        {
            VerifyIfParameterIsObjectType();
            return (_value as IDictionary<string, Parameter>).GetEnumerator();
        }

        private void VerifyIfParameterIsObjectType()
        {
            Contracts.Check(ParameterType == ParameterType.Object, "parameter is not object type.");
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void CopyTo(KeyValuePair<string, Parameter>[] array, int arrayIndex)
        {
            VerifyIfParameterIsObjectType();
            (_value as IDictionary<string, Parameter>).CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool ContainsKey(string key)
        {
            VerifyIfParameterIsObjectType();
            return (_value as IDictionary<string, Parameter>).ContainsKey(key);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Remove(string key)
        {
            VerifyIfParameterIsObjectType();
            return (_value as IDictionary<string, Parameter>).Remove(key);
        }

        public bool Equals(Parameter other)
        {
            //Check whether the compared object is null.
            if (Object.ReferenceEquals(other, null)) return false;

            //Check whether the compared object references the same data.
            if (Object.ReferenceEquals(this, other)) return true;

            var thisJson = JsonSerializer.Serialize(this);
            var otherJson = JsonSerializer.Serialize(other);

            return thisJson == otherJson;
        }

        public override int GetHashCode()
        {
            var thisJson = JsonSerializer.Serialize(this);
            return thisJson.GetHashCode();
        }
    }
}
