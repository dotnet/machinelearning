using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class PipelineTests
    {
        [TestMethod]
        public void PipelineNodeEquality()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            Assert.AreEqual(node1, node2);
        }

        [TestMethod]
        public void PipelineNodeInequalityNull()
        {
            var node1 = BuildSamplePipelineNode();
            PipelineNode node2 = null;
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityNullInCols()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.InColumns = null;
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentInColNames()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.InColumns[0] = "imdifferent";
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentInColCount()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.InColumns = new string[] { "hello", "world" };
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentOutColCount()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.OutColumns = new string[] { "hello", "world" };
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentPropCount()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.Properties["Key2"] = "Value1";
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentPropKeys()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.Properties = new Dictionary<string, object>()
            {
                {"different", "different" }
            };
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentPropValues()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.Properties["Key1"] = "Value2";
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityNullProps()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.Properties = null;
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityName()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.Name = "different";
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityType()
        {
            var node1 = BuildSamplePipelineNode();
            var node2 = BuildSamplePipelineNode();
            node2.NodeType = PipelineNodeType.Transform;
            Assert.AreNotEqual(node1, node2);
            Assert.AreNotEqual(node2, node1);
        }

        [TestMethod]
        public void PipelineNodeInequalityDifferentType()
        {
            var node1 = BuildSamplePipelineNode();
            Assert.AreNotEqual(node1, 1);
            Assert.AreNotEqual(1, node1);
        }

        private static PipelineNode BuildSamplePipelineNode()
        {
            return new PipelineNode("name", PipelineNodeType.Trainer,
                new string[] { "In1" },
                new string[] { "Out1" },
                new Dictionary<string, object>() {
                    {"Key1", "Value1" }
                });
        }
    }
}
