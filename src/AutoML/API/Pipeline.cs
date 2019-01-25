using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Newtonsoft.Json;

namespace Microsoft.ML.Auto
{
    public class Pipeline
    {
        public PipelineNode[] Nodes { get; set; }

        public Pipeline(PipelineNode[] nodes)
        {
            Nodes = nodes;
        }

        // (used by Newtonsoft)
        internal Pipeline()
        {
        }
        
        public IEstimator<ITransformer> ToEstimator()
        {
            var inferredPipeline = InferredPipeline.FromPipeline(this);
            return inferredPipeline.ToEstimator();
        }
    }

    public class PipelineNode
    {
        public string Name { get; set; }
        public PipelineNodeType NodeType { get; set; }
        public string[] InColumns { get; set; }
        public string[] OutColumns { get; set; }
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

        public override bool Equals(object obj)
        {
            var other = obj as PipelineNode;
            if(other == null)
            {
                return false;
            }
            if(this.Name != other.Name)
            {
                return false;
            }
            if(this.NodeType != other.NodeType)
            {
                return false;
            }
            if(!ColumnSetsAreEqual(this.InColumns, other.InColumns) ||
                !ColumnSetsAreEqual(this.OutColumns, other.OutColumns))
            {
                return false;
            }
            return PropertiesAreEqual(this.Properties, other.Properties);
        }

        public override int GetHashCode()
        {
            return JsonConvert.SerializeObject(this).GetHashCode();
        }

        // (used by Newtonsoft)
        internal PipelineNode()
        {
        }

        private static bool ColumnSetsAreEqual(string[] set1, string[] set2)
        {
            if(set1 == null)
            {
                return set2 == null;
            }
            if(set2 == null)
            {
                return false;
            }
            if(set1.Length != set2.Length)
            {
                return false;
            }
            for(var i = 0; i < set1.Length; i++)
            {
                if(!set1[i].Equals(set2[i]))
                {
                    return false;
                }
            }
            return true;
        }

        private static bool PropertiesAreEqual(IDictionary<string, object> props1, IDictionary<string, object> props2)
        {
            if(props1 == null)
            {
                return props2 == null;
            }
            if(props2 == null)
            {
                return false;
            }
            if(props1.Keys.Count != props2.Keys.Count)
            {
                return false;
            }
            foreach(var key in props1.Keys)
            {
                var value1 = props1[key];
                if(!props2.TryGetValue(key, out var value2))
                {
                    return false;
                }
                if(!value1.Equals(value2))
                {
                    return false;
                }
            }
            return true;
        }
    }

    public enum PipelineNodeType
    {
        Transform,
        Trainer
    }

    public class CustomProperty
    {
        public readonly string Name;
        public readonly IDictionary<string, object> Properties;
    }

    public class PipelineRunResult
    {
        public readonly Pipeline Pipeline;
        public readonly double Score;

        /// <summary>
        /// This setting is true if the pipeline run succeeded & ran to completion.
        /// Else, it is false if some exception was thrown before the run could complete.
        /// </summary>
        public readonly bool RunSucceded;

        public PipelineRunResult(Pipeline pipeline, double score, bool runSucceeded)
        {
            Pipeline = pipeline;
            Score = score;
            RunSucceded = runSucceeded;
        }
    }
}
