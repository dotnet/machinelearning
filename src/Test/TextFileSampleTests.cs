using System.IO;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class TextFileSampleTests
    {
        [TestMethod]
        public void CanParseLargeRandomStream()
        {
            using (var stream = new MemoryStream())
            {
                const int numRows = 100000;
                const int rowSize = 100;

                for (var i = 0; i < numRows; i++)
                {
                    var row = new byte[rowSize];
                    AutoMlUtils.Random.NextBytes(row);

                    // ensure byte array has no 0s, so text file sampler doesn't
                    // think file is encoded with UTF-16 or UTF-32 without a BOM
                    for (var k = 0; k < row.Length; k++)
                    {
                        if(row[k] == 0)
                        {
                            row[k] = 1;
                        }
                    }
                    stream.Write(row);
                    stream.Write(Encoding.UTF8.GetBytes("\r\n"));
                }

                stream.Seek(0, SeekOrigin.Begin);

                var sample = TextFileSample.CreateFromFullStream(stream);
                Assert.IsNotNull(sample);
                Assert.IsTrue(sample.FullFileSize > 0);
            }
        }
    }
}
