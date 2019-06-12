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
            var random = new Random();
            var dataDirectoryName = "DataDir";
            Directory.CreateDirectory(dataDirectoryName);
            for (int i = 0; i < 5; i++)
            {
                var fileName = Path.Combine(dataDirectoryName, $"Data_{i}.csv");
                dataFiles.Add(fileName);
                using (var fs = File.CreateText(fileName))
                    // Write random lines without header
                    for (int line = 0; line < 10; line++)
                        fs.WriteLine(random.NextDouble().ToString());
            }

            // Create a TextLoader.
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("RandomFeature", DataKind.Single, 0)
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
                loader.Load(Path.Combine(dataDirectoryName, "*"));
            PrintRowCount(multipleFilesWildcardData);

            // Expected Output:
            //   50
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
    }
}
