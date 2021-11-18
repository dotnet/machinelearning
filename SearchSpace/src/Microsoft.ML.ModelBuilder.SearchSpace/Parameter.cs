// <copyright file="Parameter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Converter;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    // TODO
    // Add tests
    [JsonConverter(typeof(ParameterConverter))]
    public class Parameter
    {
        private JsonSerializerSettings settings = new JsonSerializerSettings()
        {
            Formatting = Formatting.Indented,
            Culture = System.Globalization.CultureInfo.InvariantCulture,
            NullValueHandling = NullValueHandling.Ignore,
            Converters = new JsonConverter[]
            {
                new StringEnumConverter(),
            },
        };

        private JsonSerializer jsonSerializer;
        private JToken jtoken;
        private Type type;

        public Parameter(object value)
        {
            this.jsonSerializer = JsonSerializer.Create(this.settings);
            this.jtoken = JToken.FromObject(value, this.jsonSerializer);
            this.type = value.GetType();
        }

        public Parameter()
        {
            this.jsonSerializer = JsonSerializer.Create(this.settings);
            this.jtoken = JObject.Parse("{}");
            this.type = typeof(JObject);
        }

        public object Value { get => this.jtoken; }

        public ICollection<string> Keys
        {
            get
            {
                if (this.jtoken?.HasValues is null or false)
                {
                    return new string[0];
                }

                return this.jtoken.ToObject<JObject>(this.jsonSerializer).Properties().Select(prop => prop.Name).ToArray();
            }
        }

        public ICollection<Parameter> Values
        {
            get
            {
                if (this.Keys.Count == 0)
                {
                    return new Parameter[0];
                }

                Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");
                var jobject = this.jtoken as JObject;
                return this.Keys.Select(k => new Parameter(jobject[k])).ToArray();
            }
        }

        public int Count => this.Keys.Count();

        public bool IsReadOnly => true;

        public Parameter this[string key]
        {
            get
            {
                if (this.ContainsKey(key))
                {
                    return new Parameter(this.jtoken.ToObject<JObject>(this.jsonSerializer).GetValue(key));
                }
                else
                {
                    throw new KeyNotFoundException($"{key} not found");
                }
            }
            set => (this.jtoken as JObject).Add(key, value.jtoken);
        }

        public T AsType<T>()
        {
            if(this.jtoken.Type == JTokenType.Object)
            {
                var json = JsonConvert.SerializeObject(this.jtoken, this.settings);
                return JsonConvert.DeserializeObject<T>(json, this.settings);
            }
            else
            {
                return this.jtoken.ToObject<T>(this.jsonSerializer);
            }
        }

        public void Add(string key, Parameter value)
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");
            this[key] = value;
        }

        public bool ContainsKey(string key)
        {
            if (this.Count == 0)
            {
                return false;
            }

            var jobject = this.jtoken as JObject;
            return jobject.TryGetValue(key, out var _);
        }

        public bool Remove(string key)
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");
            var jobject = this.jtoken as JObject;

            return jobject.Remove(key);
        }

        public bool TryGetValue(string key, out Parameter value)
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");
            var jobject = this.jtoken as JObject;

            var res = jobject.TryGetValue(key, out var token);
            value = new Parameter(token);

            return res;
        }

        public void Add(KeyValuePair<string, Parameter> item)
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");

            this.Add(item.Key, item.Value);
        }

        public void Clear()
        {
            this.jtoken = JObject.Parse("{}");
        }

        public bool Contains(KeyValuePair<string, Parameter> item)
        {
            if (this.Count == 0)
            {
                return false;
            }

            return this.ContainsKey(item.Key) && item.Value == this[item.Key];
        }

        public void CopyTo(KeyValuePair<string, Parameter>[] array, int arrayIndex)
        {
            foreach (var kv in this)
            {
                array[arrayIndex++] = kv;
            }
        }

        public bool Remove(KeyValuePair<string, Parameter> item)
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");

            if (this.Contains(item))
            {
                return this.Remove(item.Key);
            }

            return false;
        }

        public IEnumerator<KeyValuePair<string, Parameter>> GetEnumerator()
        {
            Contract.Requires(this.jtoken is JObject, "jtoken is not JObject");

            foreach (var key in this.Keys)
            {
                yield return new KeyValuePair<string, Parameter>(key, this[key]);
            }
        }

        public void Merge(Parameter p)
        {
            Contract.Requires(this.jtoken.Type == JTokenType.Object, "jtoken type is not object");
            var left = this.jtoken.Root;
            var right = p.jtoken.Root;

            this.jtoken = this.Merge(left, right);
        }

        private JToken Merge(JToken left, JToken right)
        {
            if (left.Type != right.Type)
            {
                return left;
            }

            if (left.Type is JTokenType.Object)
            {
                var rightObject = (JObject)right;
                var leftObject = (JObject)left;
                foreach (var property in rightObject.Properties())
                {
                    var name = property.Name;
                    if (leftObject.ContainsKey(name))
                    {
                        leftObject[name] = this.Merge(leftObject[name], property.Value);
                    }
                    else
                    {
                        leftObject.Add(name, property.Value);
                    }
                }

                return left;
            }
            else
            {
                return right;
            }
        }
    }
}
