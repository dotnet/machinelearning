using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator.Parameter;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.CodeGenerator.Tests
{
    [TestClass]
    public class ParameterTest
    {
        [TestMethod]
        public void NameParameterTest()
        {
            var nameParameter = new NameParameter()
            {
                ParameterName = "ParamName",
                ParameterValue = "ParamValue",
            };
            var expectString = "ParamValue";
            Assert.AreEqual(expectString, nameParameter.ToParameter());
        }

        [TestMethod]
        public void OptionalParameterTest()
        {
            var optionParameter = new OptionalParameter()
            {
                ParameterName = "ParamName",
                ParameterValue = "ParamValue",
            };
            var expectString = "ParamName:ParamValue";
            Assert.AreEqual(expectString, optionParameter.ToParameter());
        }

        [TestMethod]
        public void NameArrayParameterTest()
        {
            var nameArrayParameter = new NameArrayParameter()
            {
                ParameterName = "ParamName",
                ArrayParameterValue = new string[] {
                    "ParamValue0",
                    "ParamValue1",
                    "ParamValue2",
                },
            };
            var expectString = "new []{ParamValue0,ParamValue1,ParamValue2}";
            Assert.AreEqual(expectString, nameArrayParameter.ToParameter());
        }

        [TestMethod]
        public void InputOutputColumnPairParameterTest()
        {
            var inputOutputColumnPairParameter = new InputOutputColumnPairParameter()
            {
                InputColumns = new string[] { "input1", "input2", "input3" },
                OutputColumns = new string[] { "output1", "output2", "output3" },
            };
            var expectString = "new []{new InputOutputColumnPair(output1,input1),new InputOutputColumnPair(output2,input2),new InputOutputColumnPair(output3,input3)}";
            Assert.AreEqual(expectString, inputOutputColumnPairParameter.ToParameter());
        }
    }
}
