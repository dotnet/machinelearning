using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class ColumnInferenceTests
    {
        [TestMethod]
        public void UnGroupColumnsTest()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            var columnInferenceWithoutGrouping = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, true, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.Columns)
            {
                Assert.IsFalse(col.Item1.Source.Length > 1 || col.Item1.Source[0].Min != col.Item1.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, true, groupColumns: true);
            Assert.IsTrue(columnInferenceWithGrouping.Columns.Count() < columnInferenceWithoutGrouping.Columns.Count());
        }
    }
}