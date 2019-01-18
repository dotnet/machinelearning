using System.Collections.Generic;

namespace Microsoft.ML.Auto
{
    public class Pipeline
    {
        public readonly PipelineNode[] Elements;

        public Pipeline(PipelineNode[] elements)
        {
            Elements = elements;
        }
    }

    public class PipelineNode
    {
        public readonly string Name;
        public readonly PipelineNodeType ElementType;
        public readonly string[] InColumns;
        public readonly string[] OutColumns;
        public readonly IDictionary<string, object> Properties;

        public PipelineNode(string name, PipelineNodeType elementType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties = null)
        {
            Name = name;
            ElementType = elementType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties ?? new Dictionary<string, object>();
        }

        public PipelineNode(string name, PipelineNodeType elementType, 
            string inColumn, string outColumn, IDictionary<string, object> properties = null) :
            this(name, elementType, new string[] { inColumn }, new string[] { outColumn }, properties)
        {
        }

        public PipelineNode(string name, PipelineNodeType elementType,
            string[] inColumns, string outColumn, IDictionary<string, object> properties = null) :
            this(name, elementType, inColumns, new string[] { outColumn }, properties)
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
        public readonly string Name;
        public readonly IDictionary<string, object> Properties;
    }
}
