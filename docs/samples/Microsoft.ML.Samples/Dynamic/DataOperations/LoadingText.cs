using System;
using System.Collections.Generic;
using System.IO;
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
            var random = new Random();
            var dataDirectoryName = "DataDir";
            Directory.CreateDirectory(dataDirectoryName);
            for (int i = 0; i < 5; i++)
            {
                var fileName = Path.Combine(dataDirectoryName, $"Data_{i}.csv");
                dataFiles.Add(fileName);
                using (var fs = File.CreateText(fileName))
                    // Write without header with a random column, followed by five
                    // tab separated columns of 0.
                    for (int line = 0; line < 10; line++)
                        fs.WriteLine(random.NextDouble().ToString()+ "\t0\t0\t0\t0\t0");
            }

            // Create a TextLoader.
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Features", DataKind.Single, 0, 5)
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


            // Save the data with 10 rows to a text file to illustrate the use of
            // sparse format, and data loading with schema inferred from a data sample.
            var sparseDataFileName = Path.Combine(dataDirectoryName, "saved_data.tsv");
            using (FileStream stream = new FileStream(sparseDataFileName, FileMode.Create))
                mlContext.Data.SaveAsText(singleFileData, stream, schema: true);

            // The data will be saved with the schema encoded in the header. Since
            // there are many zeroes in the data, it will be saved in a sparse
            // representation to save disk space. The data may be forced to be saved
            // in a dense representation by setting forceDense to true. The sparse
            // data, along with the schema, will look like the following:
            //
            //   #@ TextLoader{
            //   #@   sep=tab
            //   #@   col=Features:R4:0-5
            //   #@ }
            //   6   0:0.286606282
            //   6   0:0.1535183
            //   6   0:0.499382764
            //   6   0:0.5140711
            //
            // The schema header indicates that values are separated by tabs, and
            // that columns 0 through 5 are to read into a vector called Features
            // of type DataKind.Single (unimportant: internally represented as R4).
            // The sparse representation of the first row indicates that there are
            // 6 columns, the zero-th column has value 0.286606282, and other omitted
            // columns have value 0.

            // Create a TextLoader that allows sparse input. Since the data file inlcudes
            // the schema in the header, columns may be left as null instead of providing
            // schema information, and the data file itself may be provided as a data
            // sample so that the schema can be inferred from the header.
            var dataSample = new MultiFileSource(sparseDataFileName);
            var sparseLoader = mlContext.Data.CreateTextLoader(
                columns: null,
                allowSparse: true,
                dataSample: dataSample
            );

            // Load the saved sparse data.
            var dataFromSparseLoader = sparseLoader.Load(sparseDataFileName);
            PrintRowCount(dataFromSparseLoader);

            // Expected Output:
            //   10


            // Create a TextLoader that allows sparse input using TextLoader.Options.
            // The schema will again be inferred from the data sample.
            var options = new TextLoader.Options()
            {
                AllowSparse = true,
            };
            var sparseLoaderWithOptions = mlContext.Data.CreateTextLoader(options, dataSample);

            // Load the saved sparse data.
            var dataFromSparseLoaderWithOptions = sparseLoaderWithOptions.Load(sparseDataFileName);
            PrintRowCount(dataFromSparseLoaderWithOptions);

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
            [LoadColumn(0, 5)]
            public float[] Features { get; set; }
        }
    }
}
