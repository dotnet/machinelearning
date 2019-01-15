namespace Microsoft.ML.Auto
{
    public enum ColumnPurpose
    {
        Ignore = 0,
        Name = 1,
        Label = 2,
        NumericFeature = 3,
        CategoricalFeature = 4,
        TextFeature = 5,
        Weight = 6,
        Group = 7,
        ImagePath = 8
    }
}
