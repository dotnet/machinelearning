// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Core.Data;

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

        // (used by Newtonsoft)
        internal PipelineNode()
        {
        }
    }

    public enum PipelineNodeType
    {
        Transform,
        Trainer
    }

    public class CustomProperty
    {
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
