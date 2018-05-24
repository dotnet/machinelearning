using Microsoft.ML.Runtime.Api;

namespace MulticlassClassification_Iris
{
    public class IrisPrediction
    {
        [ColumnName("Score")] public float[] Score;
    }
}