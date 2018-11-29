using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public class LdaTransformExample
    {
        public static void LdaTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTopicsData> data = SamplesUtils.DatasetUtils.GetTopicsData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of one of the columns of the the topics data. 
            // The Review column contains the keys associated with a particular body of text.  
            //
            // Review                               
            // "animals birds cats dogs fish horse" 
            // "horse birds house fish duck cats"   
            // "car truck driver bus pickup"       
            // "car truck driver bus pickup horse"

            // A pipeline for featurizing the "Review" column
            string ldaFeatures = "LdaFeatures";
            var pipeline = ml.Transforms.Text.ProduceWordBags("Review").
                Append(ml.Transforms.Text.LatentDirichletAllocation("Review", ldaFeatures, numTopic:3));

            // The transformed data
            var transformer = pipeline.Fit(trainData);
            var transformed_data = transformer.Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var featureRow in column)
                {
                    foreach (var value in featureRow.GetValues())
                        Console.Write($"{value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Preview of the column obtained after processing the input.
            var defaultColumn = transformed_data.GetColumn<VBuffer<float>>(ml, ldaFeatures);
            printHelper(ldaFeatures, defaultColumn);

            // LdaFeatures column obtained post-transformation.
            // For LDA, we had specified numTopic:3. Hence each row of text has been featurized as a vector of floats with length 3.

            //0.1818182 0.4545455 0.3636364
            //0.3636364 0.1818182 0.4545455
            //0.2222222 0.2222222 0.5555556
            //0.2727273 0.09090909 0.6363636
        }
    }
}
