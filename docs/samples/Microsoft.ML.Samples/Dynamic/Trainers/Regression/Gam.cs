using System;
using System.Linq;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class Gam
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();
            
            // Read the Housing regression dataset
            var data = DatasetUtils.LoadHousingRegressionDataset(mlContext);

            var labelName = "MedianHomeValue";
            var featureNames = data.Schema
                .Select(column => column.Name) // Get the column names
                .Where(name => name != labelName) // Drop the Label
                .ToArray();

            // Create a pipeline.
            var pipeline =
                // Concatenate the features to create a Feature vector.
                mlContext.Transforms.Concatenate("Features", featureNames)
                // Append a GAM regression trainer, setting the "MedianHomeValue" column as the label of the dataset,
                // the "Features" column produced by concatenation as the features column,
                // and use a small number of bins to make it easy to visualize in the console window.
                // For real applications, it is recommended to start with the default number of bins.
                .Append(mlContext.Regression.Trainers.Gam(labelColumnName: labelName, featureColumnName: "Features", maximumBinCountPerFeature: 16));

            // Train the pipeline.
            var trainedPipeline = pipeline.Fit(data);

            // Extract the model from the pipeline.
            var gamModel = trainedPipeline.LastTransformer.Model;

            // Now investigate the bias and shape functions of the GAM model.
            // The bias represents the average prediction for the training data.
            Console.WriteLine($"Average predicted cost: {gamModel.Bias:0.00}");

            // Expected output:
            //   Average predicted cost: 22.53

            // Let's take a look at the features that the model built. Similar to a linear model, we have
            // one response per feature. Unlike a linear model, this response is a function instead of a line.
            // Each feature response represents the deviation from the average prediction as a function of the 
            // feature value.

            // Let's investigate the TeacherRatio variable. This is the ratio of students to teachers,
            // so the higher it is, the more students a teacher has in their classroom.
            // First, let's get the index of the variable we want to look at.
            var studentTeacherRatioIndex = featureNames.ToList().FindIndex(str => str.Equals("TeacherRatio"));

            // Next, let's get the array of histogram bin upper bounds from the model for this feature.
            // For each feature, the shape function is calculated at `MaxBins` locations along the range of 
            // values that the feature takes, and the resulting shape function can be seen as a histogram of
            // effects.
            var teacherRatioBinUpperBounds = gamModel.GetBinUpperBounds(studentTeacherRatioIndex);
            // And the array of bin effects; these are the effect size for each bin.
            var teacherRatioBinEffects = gamModel.GetBinEffects(studentTeacherRatioIndex);

            // Now, write the function to the console. The function is a set of bins, and the corresponding
            // function values. You can think of GAMs as building a bar-chart lookup table.
            Console.WriteLine("Student-Teacher Ratio");
            for (int i = 0; i < teacherRatioBinUpperBounds.Count; i++)
                Console.WriteLine($"x < {teacherRatioBinUpperBounds[i]:0.00} => {teacherRatioBinEffects[i]:0.000}");

            //  Expected output:
            //    Student-Teacher Ratio
            //    x < 14.55 =>  2.105
            //    x < 14.75 =>  2.326
            //    x < 15.40 =>  0.903
            //    x < 16.50 =>  0.651
            //    x < 17.15 =>  0.587
            //    x < 17.70 =>  0.624
            //    x < 17.85 =>  0.684
            //    x < 18.35 => -0.315
            //    x < 18.55 => -0.542
            //    x < 18.75 => -0.083
            //    x < 19.40 => -0.442
            //    x < 20.55 => -0.649
            //    x < 21.05 => -1.579
            //    x <   ∞   =>  0.318

            // Let's consider this output. To score a given example, we look up the first bin where the inequality
            // is satisfied for the feature value. We can look at the whole function to get a sense for how the
            // model responds to the variable on a global level. For the student-teacher-ratio variable, we can see
            // that smaller class sizes are predictive of a higher house value, while student-teacher ratios higher 
            // than about 18 lead to lower predictions in house value. This makes intuitive sense, as smaller class 
            // sizes are desirable and also indicative of better-funded schools, which could make buyers likely to
            // pay more for the house.
            
            // Another thing to notice is that these feature functions can be noisy. See student-teacher ratios > 21.05.
            // Common practice is to use resampling methods to estimate a confidence interval at each bin. This will
            // help to determine if the effect is real or just sampling noise. See for example 
            // Tan, Caruana, Hooker, and Lou. "Distill-and-Compare: Auditing Black-Box Models Using Transparent Model 
            // Distillation." <a href='https://arxiv.org/abs/1710.06169'>arXiv:1710.06169</a>."
        }
    }
}
