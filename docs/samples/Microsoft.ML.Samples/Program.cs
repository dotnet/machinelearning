using System;
using Samples.Dynamic.Trainers.BinaryClassification;

namespace Microsoft.ML.Samples
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Prior");
            Prior.Example();
            Console.WriteLine("\nSdcaLogisticRegression");
            SdcaLogisticRegression.Example();
            Console.WriteLine("\nSdcaLogisticRegressionWithOptions");
            SdcaLogisticRegressionWithOptions.Example();
            Console.WriteLine("\nSdcaNonCalibrated");
            SdcaNonCalibrated.Example();
            Console.WriteLine("\nSdcaNonCalibratedWithOptions");
            SdcaNonCalibratedWithOptions.Example();
            Console.WriteLine("\nSgdCalibrated");
            SgdCalibrated.Example();
            Console.WriteLine("\nSgdCalibratedWithOptions");
            SgdCalibratedWithOptions.Example();
            Console.WriteLine("\nSgdNonCalibrated");
            SgdNonCalibrated.Example();
            Console.WriteLine("\nSgdNonCalibratedWithOptions");
            SgdNonCalibratedWithOptions.Example();
            Console.WriteLine("\nSymbolicSgdLogisticRegression");
            SymbolicSgdLogisticRegression.Example();
            Console.WriteLine("\nSymbolicSgdLogisticRegressionWithOptions");
            SymbolicSgdLogisticRegression.Example();

            Console.ReadLine();
        }
    }
}
