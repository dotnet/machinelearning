// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

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

    internal enum PipelineNodeType
    {
        Transform,
        Trainer
    }

    internal class CustomProperty
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
