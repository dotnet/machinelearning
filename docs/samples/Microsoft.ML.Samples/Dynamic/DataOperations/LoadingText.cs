using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.DataOperations
{
    public static class LoadingText
    {
        // This examples shows all the ways to load data with TextLoader.
        public static void Example()
        {
            // Create 5 data files to illustrate different loading methods.
            var dataFiles = new List<string>();
            var random = new Random(1);
            var dataDirectoryName = "DataDir";
            Directory.CreateDirectory(dataDirectoryName);
            for (int i = 0; i < 5; i++)
            {
                var fileName = Path.Combine(dataDirectoryName, $"Data_{i}.csv");
                dataFiles.Add(fileName);
                using (var fs = File.CreateText(fileName))
                {
                    // Write without header with 10 random columns, forcing
                    // approximately 80% of values to be 0.
                    for (int line = 0; line < 10; line++)
                    {
                        var sb = new StringBuilder();
                        for (int pos = 0; pos < 10; pos++)
                        {
                            var value = random.NextDouble();
                            sb.Append((value < 0.8 ? 0 : value).ToString() + '\t');
                        }
                        fs.WriteLine(sb.ToString(0, sb.Length - 1));
                    }
                }
            }

            // Create a TextLoader.
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Features", DataKind.Single, 0, 9)
                },
                hasHeader: false
            );

            // Load a single file from path.
            var singleFileData = loader.Load(dataFiles[0]);
            PrintRowCount(singleFileData);

            // Expected Output:
            //   10


            // Load all 5 files from path.
            var multipleFilesData = loader.Load(dataFiles.ToArray());
            PrintRowCount(multipleFilesData);

            // Expected Output:
            //   50


            // Load all files using path wildcard.
            var multipleFilesWildcardData =
                loader.Load(Path.Combine(dataDirectoryName, "Data_*.csv"));
            PrintRowCount(multipleFilesWildcardData);

            // Expected Output:
            //   50


            // Create a TextLoader with user defined type.
            var loaderWithCustomType =
                mlContext.Data.CreateTextLoader<Data>(hasHeader: false);

            // Load a single file from path.
            var singleFileCustomTypeData = loaderWithCustomType.Load(dataFiles[0]);
            PrintRowCount(singleFileCustomTypeData);

            // Expected Output:
            //   10


            // Create a TextLoader with unknown column length to illustrate
            // how a data sample may be used to infer column size.
            var dataSample = new MultiFileSource(dataFiles[0]);
            var loaderWithUnknownLength = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Features",
                                          DataKind.Single,
                                          new[] { new TextLoader.Range(0, null) })
                },
                dataSample: dataSample
            );

            var dataWithInferredLength = loaderWithUnknownLength.Load(dataFiles[0]);
            var featuresColumn = dataWithInferredLength.Schema.GetColumnOrNull("Features");
            if (featuresColumn.HasValue)
                Console.WriteLine(featuresColumn.Value.ToString());

            // Expected Output:
            //   Features: Vector<Single, 10>
            //
            // ML.NET infers the correct length of 10 for the Features column,
            // which is of type Vector<Single>.

            PrintRowCount(dataWithInferredLength);

            // Expected Output:
            //   10


            // Save the data with 10 rows to a text file to illustrate the use of
            // sparse format.
            var sparseDataFileName = Path.Combine(dataDirectoryName, "saved_data.tsv");
            using (FileStream stream = new FileStream(sparseDataFileName, FileMode.Create))
                mlContext.Data.SaveAsText(singleFileData, stream);

            // Since there are many zeroes in the data, it will be saved in a sparse
            // representation to save disk space. The data may be forced to be saved
            // in a dense representation by setting forceDense to true. The sparse
            // data will look like the following:
            //
            //   10 7:0.943862259
            //   10 3:0.989767134
            //   10 0:0.949778438   8:0.823028445   9:0.886469543
            //
            // The sparse representation of the first row indicates that there are
            // 10 columns, the column 7 (8-th column) has value 0.943862259, and other
            // omitted columns have value 0.

            // Create a TextLoader that allows sparse input.
            var sparseLoader = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Features", DataKind.Single, 0, 9)
                },
                allowSparse: true
            );

            // Load the saved sparse data.
            var sparseData = sparseLoader.Load(sparseDataFileName);
            PrintRowCount(sparseData);

            // Expected Output:
            //   10


            // Create a TextLoader without any column schema using TextLoader.Options.
            // Since the sparse data file was saved with ML.NET, it has the schema
            // enoded in its header that the loader can understand:
            //
            // #@ TextLoader{
            // #@   sep=tab
            // #@   col=Features:R4:0-9
            // #@ }
            //
            // The schema syntax is unimportant since it is only used internally. In
            // short, it tells the loader that the values are separated by tabs, and
            // that columns 0-9 in the text file are to be read into one column named
            // "Features" of type Single (internal type R4).

            var options = new TextLoader.Options()
            {
                AllowSparse = true,
            };
            var dataSampleWithSchema = new MultiFileSource(sparseDataFileName);
            var sparseLoaderWithSchema =
                mlContext.Data.CreateTextLoader(options, dataSample: dataSampleWithSchema);

            // Load the saved sparse data.
            var sparseDataWithSchema = sparseLoaderWithSchema.Load(sparseDataFileName);
            PrintRowCount(sparseDataWithSchema);

            // Expected Output:
            //   10
        }

        private static void PrintRowCount(IDataView idv)
        {
            // IDataView is lazy so we need to iterate through it
            // to get the number of rows.
            long rowCount = 0;
            using (var cursor = idv.GetRowCursor(idv.Schema))
                while (cursor.MoveNext())
                    rowCount++;

            Console.WriteLine(rowCount);
        }

        private class Data
        {
            [LoadColumn(0, 9)]
            public float[] Features { get; set; }
        }
    }
}
