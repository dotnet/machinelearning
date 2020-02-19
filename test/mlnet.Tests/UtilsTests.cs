using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.ML.CLI.Utilities;

namespace mlnet.Tests
{
    [TestClass]
    public class UtilsTests
    {

        [DataTestMethod]
        [DataRow("NameWithoutSpaces", '_', "NameWithoutSpaces")]
        [DataRow("Name With Spaces", '.', "Name.With.Spaces")]
        [DataRow("학습유형들에", '_', "학습유형들에")]
        public void SanitizeTest(string input, char replacement, string expected)
        {
            Assert.AreEqual(Utils.Sanitize(input, replacement), expected);
        }

    }
}
