namespace Microsoft.ML.TimeSeries
{
    internal sealed class StlDeseasonality : DeseasonalityBase
    {
        private readonly InnerStl _stl;

        public StlDeseasonality()
        {
            _stl = new InnerStl(true);
        }

        public override void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            bool success = _stl.Decomposition(values, period);
            if (success)
            {
                for (int i = 0; i < _stl.Residual.Count; ++i)
                {
                    results[i] = _stl.Residual[i];
                }
            }
            else
            {
                for (int i = 0; i < values.Length; ++i)
                {
                    results[i] = values[i];
                }
            }
        }
    }
}
