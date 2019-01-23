namespace Microsoft.ML.Auto
{
    internal class ColumnDimensions
    {
        public int? Cardinality;
        public bool? HasMissing;

        public ColumnDimensions(int? cardinality, bool? hasMissing)
        {
            Cardinality = cardinality;
            HasMissing = hasMissing;
        }
    }
}
