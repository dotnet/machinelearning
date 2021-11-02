using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.DataOperations
{
    public static class LoadingSvmLight
    {
        // This examples shows how to load data with SvmLightLoader.
        public static void Example()
        {
            // Create a random SVM light format file.
            var random = new Random(42);
            var dataDirectoryName = "DataDir";
            Directory.CreateDirectory(dataDirectoryName);
            var fileName = Path.Combine(dataDirectoryName, $"SVM_Data.csv");
            using (var fs = File.CreateText(fileName))
            {
                // Write random lines in SVM light format
                for (int line = 0; line < 10; line++)
                {
                    var sb = new StringBuilder();
                    if (random.NextDouble() > 0.5)
                        sb.Append("1 ");
                    else
                        sb.Append("-1 ");
                    if (line % 2 == 0)
                        sb.Append("cost:1 ");
                    else
                        sb.Append("cost:2 ");
                    for (int i = 1; i <= 10; i++)
                    {
                        if (random.NextDouble() > 0.5)
                            continue;
                        sb.Append($"{i}:{random.NextDouble()} ");
                    }
                    fs.WriteLine(sb.ToString());
                }
            }

            // Create an SvmLightLoader.
            var mlContext = new MLContext();
            var file = new MultiFileSource(fileName);
            var loader = mlContext.Data.CreateSvmLightLoader(dataSample: file);

            // Load a single file from path.
            var svmData = loader.Load(file);

            PrintSchema(svmData);

            // Expected Output:
            // Column Label type Single
            // Column Weight type Single
            // Column GroupId type Key<UInt64, 0 - 18446744073709551613>
            // Column Comment type String
            // Column Features type Vector<Single, 10>

            PrintData(svmData);

            // Expected Output:
            // 1 1 0 0 0.2625927 0 0 0.7612506 0.2573214 0 0.3809696 0.5174511
            // -1 1 0 0 0 0.7051522 0 0 0.7111546 0.9062127 0 0
            // -1 1 0 0 0 0.535722 0 0 0.1491191 0.05100901 0 0
            // -1 1 0 0.6481459 0.04449836 0 0 0.4203662 0 0 0.01325378 0.2674384
            // -1 1 0 0 0.7978093 0.5134962 0.008952909 0 0.003074009 0.6541431 0.9135142 0
            // -1 1 0 0.3727672 0.4369507 0 0 0.2973725 0 0 0 0.8816807
            // 1 1 0 0.1031429 0.3332489 0 0.1346936 0.5916625 0 0 0 0
            // 1 1 0 0 0 0.3454075 0 0.2197472 0.03848049 0.5923384 0.09373277 0
            // -1 1 0 0.7511514 0 0.0420841 0 0 0.9262196 0 0.545344 0
            // 1 1 0 0.02958358 0.9334617 0 0 0.8833956 0.2947684 0 0 0

            // If the loader is created without a data sample we need to specify the number of features expected in the file.
            loader = mlContext.Data.CreateSvmLightLoader(inputSize: 10);
            svmData = loader.Load(file);

            PrintSchema(svmData);
            PrintData(svmData);
        }

        private static void PrintSchema(IDataView svmData)
        {
            foreach (var col in svmData.Schema)
                Console.WriteLine($"Column {col.Name} type {col.Type}");
        }

        private static void PrintData(IDataView svmData)
        {
            using (var cursor = svmData.GetRowCursor(svmData.Schema))
            {
                var labelGetter = cursor.GetGetter<float>(svmData.Schema["Label"]);
                var weightGetter = cursor.GetGetter<float>(svmData.Schema["Weight"]);
                var featuresGetter = cursor.GetGetter<VBuffer<float>>(svmData.Schema["Features"]);

                VBuffer<float> features = default;
                while (cursor.MoveNext())
                {
                    float label = default;
                    labelGetter(ref label);

                    float weight = default;
                    weightGetter(ref weight);

                    featuresGetter(ref features);

                    Console.WriteLine($"{label} {weight} {string.Join(' ', features.DenseValues())}");
                }
            }
        }
    }
}
