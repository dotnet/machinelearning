namespace Microsoft.ML.TimeSeries
{
    internal abstract class DeseasonalityBase
    {
        public abstract void Deseasonality(ref double[] values, int period, ref double[] results);
    }
}
