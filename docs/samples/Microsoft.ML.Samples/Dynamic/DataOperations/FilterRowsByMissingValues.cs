using System;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic
{
    public class FilterRowsByMissingValues
    {
        /// <summary>
        /// Sample class showing how to use FilterRowsByMissingValues.
        /// </summary>
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var dataEnumerable = DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(10, naRate: 0.05);
            var data = mlContext.Data.LoadFromEnumerable(dataEnumerable);

            // Look at the original dataset
            Console.WriteLine($"Label\tFeatures");
            foreach (var row in dataEnumerable)
            {
                Console.WriteLine($"{row.Label}\t({string.Join(", ", row.Features)})");
            }
            Console.WriteLine();
            // Expected output:
            //  Label Features
            //  0 (0.9680227, 0.4060332, 1.106027, 1.17755, 0.4919063, 0.8326591, 1.182151, NaN, 1.195347, 0.5145918)
            //  1 (0.9919022, NaN, 0.5262842, 0.6876203, 0.08110995, 0.4533272, 0.9885438, 0.7629636, NaN, 0.3431419)
            //  1 (0.7159725, 0.2734515, 0.7947656, 0.4572088, 0.2213147, 0.7187268, 0.4879681, 0.8781915, 0.7353975, 0.679749)
            //  0 (1.095362, 0.2865799, 0.3701428, 1.026814, 1.199973, 0.8522052, 1.009463, 0.929094, 0.3255273, 0.3891238)
            //  1 (0.3255007, 0.4683977, 0.8092038, 0.764506, 0.2949968, 0.6633928, 0.2867224, 0.2225179, 0.06851885, 0.693045)
            //  1 (0.221342, 0.0665216, 0.6785055, 0.1490974, 0.6098703, 0.4906252, 0.6776115, 0.2254031, 0.005082198, 0.850485)
            //  0 (0.9049759, 1.188812, 0.7227401, 0.7065761, 0.2570084, 0.6960788, 0.8131579, 0.942329, 1.133393, 0.8996523)
            //  0 (0.8851265, 0.3727676, 0.8091109, 1.197115, 0.2634366, 1.04256, 0.8459901, 1.170127, 0.7129673, 1.013653)
            //  1 (0.5528619, 0.9945465, 0.06445368, 0.4830741, 0.0716896, 0.1508327, 0.4510793, NaN, 0.8160448, 0.9136292)
            //  1 (0.9628896, 0.01686989, 0.2783295, 0.5877925, 0.324167, 0.974933, 0.9728873, 0.1322647, 0.1782212, 0.5446572)

            // Filter out any row with an NA value
            var filteredData = mlContext.Data.FilterRowsByMissingValues(data, "Features");

            // Take a look at the resulting dataset and note that the Feature vectors with NaNs are missing.
            var enumerable = mlContext.Data.CreateEnumerable<DatasetUtils.FloatLabelFloatFeatureVectorSample>(filteredData, reuseRowObject: true);
            Console.WriteLine($"Label\tFeatures");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Label}\t({string.Join(", ", row.Features)})");
            }
            // Expected output:
            //  Label Features
            //  1 (0.7159725, 0.2734515, 0.7947656, 0.4572088, 0.2213147, 0.7187268, 0.4879681, 0.8781915, 0.7353975, 0.679749)
            //  0 (1.095362, 0.2865799, 0.3701428, 1.026814, 1.199973, 0.8522052, 1.009463, 0.929094, 0.3255273, 0.3891238)
            //  1 (0.3255007, 0.4683977, 0.8092038, 0.764506, 0.2949968, 0.6633928, 0.2867224, 0.2225179, 0.06851885, 0.693045)
            //  1 (0.221342, 0.0665216, 0.6785055, 0.1490974, 0.6098703, 0.4906252, 0.6776115, 0.2254031, 0.005082198, 0.850485)
            //  0 (0.9049759, 1.188812, 0.7227401, 0.7065761, 0.2570084, 0.6960788, 0.8131579, 0.942329, 1.133393, 0.8996523)
            //  0 (0.8851265, 0.3727676, 0.8091109, 1.197115, 0.2634366, 1.04256, 0.8459901, 1.170127, 0.7129673, 1.013653)
            //  1 (0.9628896, 0.01686989, 0.2783295, 0.5877925, 0.324167, 0.974933, 0.9728873, 0.1322647, 0.1782212, 0.5446572)
        }
    }
}
