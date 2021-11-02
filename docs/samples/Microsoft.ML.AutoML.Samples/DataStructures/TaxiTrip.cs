using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public class TaxiTrip
    {
        [LoadColumn(0)]
        public string VendorId;

        [LoadColumn(1)]
        public float RateCode;

        [LoadColumn(2)]
        public float PassengerCount;

        [LoadColumn(3)]
        public float TripTimeInSeconds;

        [LoadColumn(4)]
        public float TripDistance;

        [LoadColumn(5)]
        public string PaymentType;

        [LoadColumn(6)]
        public float FareAmount;
    }
}
