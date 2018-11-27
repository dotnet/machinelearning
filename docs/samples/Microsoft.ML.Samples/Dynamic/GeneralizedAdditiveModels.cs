using Microsoft.ML.Runtime.Data;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public class GeneralizedAdditiveModels_RegressionExample
    {
        public static void RunExample()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a sentiment.tsv file in the filesystem.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("MedianHomeValue", DataKind.R4, 0),
                        new TextLoader.Column("CrimesPerCapita", DataKind.R4, 1),
                        new TextLoader.Column("PercentResidental", DataKind.R4, 2),
                        new TextLoader.Column("PercentNonRetail", DataKind.R4, 3),
                        new TextLoader.Column("CharlesRiver", DataKind.R4, 4),
                        new TextLoader.Column("NitricOxides", DataKind.R4, 5),
                        new TextLoader.Column("RoomsPerDwelling", DataKind.R4, 6),
                        new TextLoader.Column("PercentPre40s", DataKind.R4, 7),
                        new TextLoader.Column("EmploymentDistance", DataKind.R4, 8),
                        new TextLoader.Column("HighwayDistance", DataKind.R4, 9),
                        new TextLoader.Column("TaxRate", DataKind.R4, 10),
                        new TextLoader.Column("TeacherRatio", DataKind.R4, 11),
                    }
                });
            
            // Read the data
            var data = reader.Read(dataFile);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Then append a gam regressor, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features column,
            // and use a small number of bins to make it easy to visualize in the console window.
            // For real appplications, it is recommended to start with the default number of bins.
            var labelName = "MedianHomeValue";
            var featureNames = data.Schema.GetColumns()
                .Select(tuple => tuple.column.Name) // Get the column names
                .Where(name => name != labelName) // Drop the Label
                .ToArray();
            var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
                    .Append(mlContext.Regression.Trainers.GeneralizedAdditiveModels(
                        labelColumn: labelName, featureColumn: "Features", maxBins: 16));
            var fitPipeline = pipeline.Fit(data);

            // Extract the model from the pipeline
            var gamModel = fitPipeline.LastTransformer.Model;

            // Step 3: Investigate the properties of the model

            // The intercept for the GAM models represent the average prediction for the training data
            var intercept = gamModel.Intercept;
            // Expected output: Average predicted cost: 22.53
            Console.WriteLine($"Average predicted cost: {intercept:0.00}");

            // Let's take a look at the features that the model built. Similar to a linear model, we have
            // one response per feature. Unlike a linear model, this response is a function instead of a line.
            // Each feature response represents the deviation from the average prediction as a function of the 
            // feature value.

            // Let's investigate the TeacherRatio variable. This is the ratio of students to teachers,
            // so the higher it is, the more students a teacher has in their classroom.
            // First, let's get the index of the variable we want to look at
            var studentTeacherRatioIndex = featureNames.ToList().FindIndex(str => str.Equals("TeacherRatio"));

            // Next, let's get the array of bin upper bounds from the model for this feature
            var teacherRatioBinUpperBounds = gamModel.GetFeatureBinUpperBounds(studentTeacherRatioIndex);
            // And the array of bin weights; these are the effect size for each bin
            var teacherRatioFeatureWeights = gamModel.GetFeatureWeights(studentTeacherRatioIndex);

            // Now, write the function to the console. The function is a set of bins, and the corresponding
            // function values. You can think of GAMs as building a bar-chart lookup table.
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
            //
            // Let's consider this output. To score a given example, we look up the first bin where the inequality
            // is satisfied for the feature value. We can look at the whole function to get a sense for how the
            // model responds to the variable on a global level. For the student-teacher-ratio variable, we can see
            // that smaller class sizes are predictive of a higher house value, while student-teacher ratios higher 
            // than about 18 lead to lower predictions in house value. This makes intuitive sense, as smaller class 
            // sizes are desirable and also indicative of better-funded schools, which could make buyers likely to
            // pay more for the house.
            //
            // Another thing to notice is that these feature functions can be noisy. See student-teacher ratios > 21.05.
            // Common practice is to use resampling methods to estimate a confidence interval at each bin. This will
            // help to determine if the effect is real or just sampling noise. See for example 
            // Tan, Caruana, Hooker, and Lou. "Distill-and-Compare: Auditing Black-Box Models Using Transparent Model 
            // Distillation." <a href='https://arxiv.org/abs/1710.06169'>arXiv:1710.06169</a>."
            Console.WriteLine();
            Console.WriteLine("Student-Teacher Ratio");
            for (int i = 0; i < teacherRatioBinUpperBounds.Length; i++)
            {
                Console.WriteLine($"x < {teacherRatioBinUpperBounds[i]:0.00} => {teacherRatioFeatureWeights[i]:0.000}");
            }
            Console.WriteLine();
        }
    }
}
