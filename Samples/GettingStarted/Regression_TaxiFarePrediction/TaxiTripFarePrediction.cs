using Microsoft.ML.Runtime.Api;

namespace Regression_TaxiFarePrediction
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")] public float FareAmount;
    }
}