// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Xunit.Sdk;
namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    /// <summary>
    /// A theory for tests with varying tolerance levels
    /// <code>
    /// [Theory, VaryingTolerance(5)]
    /// public void VaryingToleranceTest(double tolerance)
    /// {
    ///     Console.WriteLine(String.Format("Tolerance: {0}", tolerance)); // "Tolerance: 1e-5
    ///     ...
    ///     Assert.InRange(actualMetric, expectedMetric - tolerance, expectedMetric + tolerance);
    /// }
    /// </code>
    /// </summary>
    public sealed class VaryingToleranceAttribute : DataAttribute
    {
        public VaryingToleranceAttribute(int tolerance)
        {
            Tolerance = tolerance;
        }

        public int Tolerance { get; }
        public override IEnumerable<object[]> GetData(MethodInfo testMethod)
        {
            Console.WriteLine(String.Format("Test \"{0}\" utilizes varying tolerances.", testMethod.Name));
            yield return new object[] { Tolerance };
        }
    }
}