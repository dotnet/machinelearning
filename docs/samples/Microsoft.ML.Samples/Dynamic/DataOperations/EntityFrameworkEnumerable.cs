using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.EntityFrameworkCore;

namespace Microsoft.ML.Samples.Dynamic
{
    public class HousingData
    {
        // Entity Framework requires the object to have a primary key
        public int HousingDataId { get; set; }
        public float MedianHomeValue { get; set; }
        public float CrimesPerCapita { get; set; }
        public float PercentResidental { get; set; }
        public float PercentNonRetail { get; set; }
        public float CharlesRiver { get; set; }
        public float NitricOxides { get; set; }
        public float RoomsPerDwelling { get; set; }
        public float PercentPre40s { get; set; }
        public float EmploymentDistance { get; set; }
        public float HighwayDistance { get; set; }
        public float TaxRate { get; set; }
        public float TeacherRatio { get; set; }
    }

    public partial class MLNetExampleContext : DbContext
    {
        public DbSet<HousingData> Housing { get; set; }

        public MLNetExampleContext(DbContextOptions<MLNetExampleContext> options)
            : base(options)
        {
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                optionsBuilder.UseSqlServer("Server=(localdb)\\mssqllocaldb;Database=EFProviders.InMemory;Trusted_Connection=True;ConnectRetryCount=0");
            }
        }
    }

    public static class EntityFrameworkEnumerable
    {
        private static IEnumerable<HousingData> housingData;

        public static void Example()
        {
            // Download the housing data set and get the file name.
            var housingFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Read data from file into an IEnumerable.
            var data = File.ReadAllLines(housingFile)
                .Skip(1) // Skip the header row
                .Select(l => l.Split('\t'))
                .Select(i => new HousingData
                {
                    MedianHomeValue = float.Parse(i[0]),
                    CrimesPerCapita = float.Parse(i[1]),
                    PercentResidental = float.Parse(i[2]),
                    PercentNonRetail = float.Parse(i[3]),
                    CharlesRiver = float.Parse(i[4]),
                    NitricOxides = float.Parse(i[5]),
                    RoomsPerDwelling = float.Parse(i[6]),
                    PercentPre40s = float.Parse(i[7])
                });

            // Options to pass to the DbContext to use an in-memory database for
            // Entity Framework and to give the database a name.
            var options = new DbContextOptionsBuilder<MLNetExampleContext>()
                .UseInMemoryDatabase(databaseName: "TestData")
                .Options;

            // Add the data to the in-memory database via Entity Framework.
            using (var dbContext = new MLNetExampleContext(options))
            {
                foreach (var dataItem in data)
                {
                    dbContext.Add(dataItem);
                }

                var rowsAffected = dbContext.SaveChanges();

                housingData = dbContext.Housing.ToList();
            }

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Load the housing dataset into an IDataView from the IEnumerable that was
            // retrieved from the database.
            var dataView = mlContext.Data.LoadFromEnumerable(housingData);

            //////////////////// Data Preview ////////////////////
            // MedianHomeValue    CrimesPerCapita    PercentResidental    PercentNonRetail    CharlesRiver    NitricOxides    RoomsPerDwelling    PercentPre40s
            // 24.00              0.00632            18.00                2.310               0               0.5380          6.5750              65.20
            // 21.60              0.02731            00.00                7.070               0               0.4690          6.4210              78.90
            // 34.70              0.02729            00.00                7.070               0               0.4690          7.1850              61.10

            var split = mlContext.Regression.TrainTestSplit(dataView, testFraction: 0.2);

            // Create the estimator. We only concatenate the features into a single vector column
            // and run the FastTree trainer on the data.
            var pipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s")
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "MedianHomeValue"));

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(split.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, label: "MedianHomeValue");

            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);
            Console.ReadLine();

            // Expected output
            //   Mean Absolute Error: 2.75
            //   Mean Square dError: 12.61
            //   Root Mean Squared Error: 3.55
            //   RSquared: 0.83
        }
    }
}
