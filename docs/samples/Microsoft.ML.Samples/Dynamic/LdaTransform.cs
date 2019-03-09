using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class LatentDirichletAllocationTransform
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and then read it as a ML.NET data set.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTopicsData> data = SamplesUtils.DatasetUtils.GetTopicsData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of one of the columns of the the topics data. 
            // The Review column contains the keys associated with a particular body of text.  
            //
            // Review                               
            // "animals birds cats dogs fish horse" 
            // "horse birds house fish duck cats"   
            // "car truck driver bus pickup"       
            // "car truck driver bus pickup horse"

            string review = nameof(SamplesUtils.DatasetUtils.SampleTopicsData.Review);
            string ldaFeatures = "LdaFeatures";

            // A pipeline for featurizing the "Review" column
            var pipeline = ml.Transforms.Text.ProduceWordBags(review).
                Append(ml.Transforms.Text.LatentDirichletAllocation(review, ldaFeatures, numberOfTopics: 3));

            // The transformed data
            var transformer = pipeline.Fit(trainData);
            var transformed_data = transformer.Transform(trainData);

            // Column obtained after processing the input.
            var ldaFeaturesColumn = transformed_data.GetColumn<VBuffer<float>>(transformed_data.Schema[ldaFeatures]);

            Console.WriteLine($"{ldaFeatures} column obtained post-transformation.");
            foreach (var featureRow in ldaFeaturesColumn)
            {
                foreach (var value in featureRow.GetValues())
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            Console.WriteLine("===================================================");

            // LdaFeatures column obtained post-transformation.
            // For LDA, we had specified numTopic:3. Hence each row of text has been featurized as a vector of floats with length 3.

            //0.1818182 0.4545455 0.3636364
            //0.3636364 0.1818182 0.4545455
            //0.2222222 0.2222222 0.5555556
            //0.2727273 0.09090909 0.6363636
        }
    }
}
