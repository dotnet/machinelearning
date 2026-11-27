// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class LocalizationResourceTest
    {
        [Fact]
        public void PseudoLocalizedSatelliteResourcesAreSelected()
        {
            CultureInfo originalCulture = CultureInfo.CurrentUICulture;
            try
            {
                CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo("qps-ploc");
                Assert.Equal("[!! Psëûðø-løçålïzëð tëst mëssågë. !!]", Strings.PseudoLocalizationProbe);
            }
            finally
            {
                CultureInfo.CurrentUICulture = originalCulture;
            }
        }
    }
}
