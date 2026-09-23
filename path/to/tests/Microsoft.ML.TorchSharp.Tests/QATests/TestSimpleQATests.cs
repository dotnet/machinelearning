using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Threading;

namespace Microsoft.ML.TorchSharp.Tests.QATests
{
    [TestClass]
    public class TestSimpleQATests
    {
        [TestMethod]
        public void TestSimpleQA_Succeeds()
        {
            // Create a new MLContext
            var mlContext = new MLContext();

            // Create a new TorchSharp model
            var model = new TorchSharpModel(mlContext);

            // Set a timeout for the test
            var timeout = TimeSpan.FromSeconds(30);

            try
            {
                // Run the test with a timeout
                model.RunTest(timeout);
            }
            catch (OperationCanceledException ex)
            {
                // Handle the timeout exception
                Assert.Fail($"Test timed out: {ex.Message}");
            }
            catch (Exception ex)
            {
                // Handle any other exceptions
                Assert.Fail($"Test failed: {ex.Message}");
            }
        }
    }
}