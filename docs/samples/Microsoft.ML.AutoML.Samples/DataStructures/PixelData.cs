using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public class PixelData
    {
        [LoadColumn(0, 63)]
        [VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64)]
        public float Number;
    }
}
