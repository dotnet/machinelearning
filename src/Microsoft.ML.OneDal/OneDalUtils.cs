// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.OneDal
{
    [BestFriend]
    internal static class OneDalUtils
    {

#if false
        [BestFriend]
        internal static bool IsDispatchingEnabled()
        {
            return Environment.GetEnvironmentVariable("MLNET_BACKEND") == "ONEDAL" &&
                System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.X64;
        }
#else
        [BestFriend]
        internal static bool IsDispatchingEnabled()
        {
            if (Environment.GetEnvironmentVariable("MLNET_BACKEND") == "ONEDAL" &&
                System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.X64)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
#if NETFRAMEWORK
                // AppContext not available in the framework, user needs to set PATH manually
                // this will probably result in a runtime error where the user needs to set the PATH
#else
                    var currentDir = AppContext.BaseDirectory;
                    var nativeLibs = Path.Combine(currentDir, "runtimes", "win-x64", "native");
                    var originalPath = Environment.GetEnvironmentVariable("PATH");
                    Environment.SetEnvironmentVariable("PATH", nativeLibs + ";" + originalPath);
#endif
                }
                return true;
            }
            return false;
        }
#endif

        [BestFriend]
        internal static long GetTrainData(IChannel channel, FloatLabelCursor.Factory cursorFactory, ref List<float> featuresList, ref List<float> labelsList, int numberOfFeatures)
        {
            long n = 0;
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    // label
                    labelsList.Add(cursor.Label);

                    // features
                    var values = cursor.Features.GetValues();
                    if (cursor.Features.IsDense)
                    {
                        channel.Assert(values.Length == numberOfFeatures);

                        for (int j = 0; j < numberOfFeatures; ++j)
                        {
                            featuresList.Add(values[j]);
                        }
                    }
                    else
                    {
                        var indices = cursor.Features.GetIndices();
                        int i = 0;
                        for (int j = 0; j < indices.Length; ++j)
                        {
                            for (int k = i; k < indices[j]; ++k)
                            {
                                featuresList.Add(0);
                            }
                            featuresList.Add(values[indices[j]]);
                            i = indices[j] + 1;
                        }
                    }
                    n++;
                }
                channel.Check(n > 0, "No training examples in dataset.");
                if (cursor.BadFeaturesRowCount > 0)
                    channel.Warning("Skipped {0} instances with missing features/labelColumn during training", cursor.SkippedRowCount);
            }
            return n;
        }
    }
}
