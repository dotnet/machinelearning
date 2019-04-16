using Samples.Dynamic;
using Samples.Dynamic.Trainers.BinaryClassification.Calibrators;

namespace Microsoft.ML.Samples
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            ConvertToGrayscale.Example();
            ConvertToImage.Example();
            ExtractPixels.Example();
            LoadImages.Example();
            ResizeImages.Example();
        }
    }
}
