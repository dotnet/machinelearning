// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Newtonsoft.Json;

namespace Microsoft.ML.AutoML
{
    internal class Pipeline
    {
        public PipelineNode[] Nodes { get; set; }
        public bool CacheBeforeTrainer { get; set; }

        public Pipeline(PipelineNode[] nodes, bool cacheBeforeTrainer = false)
        {
            Nodes = nodes;
            CacheBeforeTrainer = cacheBeforeTrainer;
        }

        // (used by Newtonsoft)
        internal Pipeline()
        {
        }

        public IEstimator<ITransformer> ToEstimator(MLContext context)
        {
            var inferredPipeline = SuggestedPipeline.FromPipeline(context, this);
            return inferredPipeline.ToEstimator();
        }
    }

    internal class PipelineNode
    {
        private IDictionary<string, string> _serializedProperty;

        public string Name { get; set; }
        public PipelineNodeType NodeType { get; set; }
        public string[] InColumns { get; set; }
        public string[] OutColumns { get; set; }

        [JsonIgnore]
        public IDictionary<string, object> Properties { get; set; }

        public PipelineNode(string name, PipelineNodeType nodeType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties = null)
        {
            Name = name;
            NodeType = nodeType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties ?? new Dictionary<string, object>();
        }

        public PipelineNode(string name, PipelineNodeType nodeType,
            string inColumn, string outColumn, IDictionary<string, object> properties = null) :
            this(name, nodeType, new string[] { inColumn }, new string[] { outColumn }, properties)
        {
        }

        public PipelineNode(string name, PipelineNodeType nodeType,
            string[] inColumns, string outColumn, IDictionary<string, object> properties = null) :
            this(name, nodeType, inColumns, new string[] { outColumn }, properties)
        {
        }

        // (used by Newtonsoft)
        internal PipelineNode()
        {
        }

        [JsonProperty]
        internal IDictionary<string, string> SerializedProperties
        {
            get
            {
                if (_serializedProperty != null)
                {
                    return _serializedProperty;
                }

                if (Properties == null)
                {
                    return null;
                }

                var res = new Dictionary<string, string>();

                foreach(var kv in Properties)
                {
                    if (kv.Value == null)
                        continue;

                    var type = kv.Value.GetType();
                    if (type == typeof(bool))
                    {
                        //True to true
                        res[kv.Key] = ((bool)kv.Value).ToString(CultureInfo.InvariantCulture).ToLowerInvariant();
                    }
                    if (type == typeof(float))
                    {
                        //0.0 to 0.0f
                        res[kv.Key] = ((float)kv.Value).ToString(CultureInfo.InvariantCulture) + "f";
                    }

                    if (type == typeof(int))
                    {
                        res[kv.Key] = ((int)kv.Value).ToString(CultureInfo.InvariantCulture);
                    }

                    if (type == typeof(double))
                    {
                        res[kv.Key] = ((double)kv.Value).ToString(CultureInfo.InvariantCulture);
                    }

                    if (type == typeof(long))
                    {
                        res[kv.Key] = ((long)kv.Value).ToString(CultureInfo.InvariantCulture);
                    }

                    if (type == typeof(string))
                    {
                        var val = kv.Value.ToString();
                        if (val == "<Auto>")
                            continue; // This is temporary fix and needs to be fixed in AutoML SDK
                        // string to @"string"
                        res[kv.Key] = $"@\"{val.Replace("\"","\"\"")}\"";
                    }

                    if (type == typeof(PipelineNode))
                    {
                        var val = JsonConvert.SerializeObject(kv.Value);
                        res[kv.Key] = val;
                    }

                    if (type.IsEnum)
                    {
                        //example: "MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression"
                        res[kv.Key] = $"{type.ReflectedType.Name}.{type.Name}.{kv.Value.ToString()}";
                    }

                    if (type == typeof(CustomProperty))
                    {
                        res[kv.Key] = ((CustomProperty)kv.Value).ToString();
                    }
                }
                return res;
            }
            set
            {
                _serializedProperty = value;
            }
        }
    }

    internal enum PipelineNodeType
    {
        Transform,
        Trainer
    }

    internal class CustomProperty
    {
        public const string Seperator = "=";
        public string Name { get; set; }
        public IDictionary<string, object> Properties { get; set; }

        public CustomProperty(string name, IDictionary<string, object> properties)
        {
            Name = name;
            Properties = properties;
        }

        internal CustomProperty()
        {
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach(var kv in Properties)
            {
                sb.Append(kv.Key);
                sb.Append(Seperator);
                sb.Append(kv.Value.ToString());
                sb.Append(",");
            }
            if(sb.Length > 0)
                sb.Remove(sb.Length - 1, 1); //remove the last ,
            return $"new {Name}(){{{sb}}}";
        }
    }

    internal class PipelineScore
    {
        public readonly double Score;

        /// <summary>
        /// This setting is true if the pipeline run succeeded and ran to completion.
        /// Else, it is false if some exception was thrown before the run could complete.
        /// </summary>
        public readonly bool RunSucceeded;

        internal readonly Pipeline Pipeline;

        internal PipelineScore(Pipeline pipeline, double score, bool runSucceeded)
        {
            Pipeline = pipeline;
            Score = score;
            RunSucceeded = runSucceeded;
        }
    }
}
