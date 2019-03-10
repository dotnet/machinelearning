# Explainability CookBook

This document serves to provide different ways to use the explainability features of ML.NET.

All of these samples will use the [housting data](https://github.com/dotnet/machinelearning/blob/master/test/data/housing.txt) and will reference the below data schema class.

```csharp
public class HousingData
{
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
```

## How do I look at the global feature importance?
The below snippet shows how to get a glimpse of the the feature importance, or how much each column of data impacts the performance of the model.

```csharp
MLContext context = new MLContext();

IDataView data = context.Data.LoadFromTextFile("./housing.txt", new[]
{
    new TextLoader.Column("Label", DataKind.Single, 0),
    new TextLoader.Column("CrimesPerCapita", DataKind.Single, 1),
    new TextLoader.Column("PercentResidental", DataKind.Single, 2),
    new TextLoader.Column("PercentNonRetail", DataKind.Single, 3),
    new TextLoader.Column("CharlesRiver", DataKind.Single, 4),
    new TextLoader.Column("NitricOxides", DataKind.Single, 5),
    new TextLoader.Column("RoomsPerDwelling", DataKind.Single, 6),
    new TextLoader.Column("PercentPre40s", DataKind.Single, 7),
    new TextLoader.Column("EmploymentDistance", DataKind.Single, 8),
    new TextLoader.Column("HighwayDistance", DataKind.Single, 9),
    new TextLoader.Column("TaxRate", DataKind.Single, 10),
    new TextLoader.Column("TeacherRatio", DataKind.Single, 11)
},
hasHeader: true);

var pipeline = context.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio")
    .Append(context.Regression.Trainers.FastTree());

var model = pipeline.Fit(data);
var transformedData = model.Transform(data);

var featureImportance = context.Regression.PermutationFeatureImportance(model.LastTransformer, transformedData);

foreach (var metricsStatistics in featureImportance)
{
    Console.WriteLine($"Root Mean Squared - {metricsStatistics.Rms.Mean}");
}
```

## How do I get a model's weights to look at the global feature importance?


## How do I look at the feature importance per row?

