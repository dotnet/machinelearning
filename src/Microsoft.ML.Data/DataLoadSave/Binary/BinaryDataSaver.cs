// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using System.IO;

namespace Microsoft.ML.Data.IO
{
    public static class BinaryDataSaver
    {
        public static void SaveData(IHostEnvironment env, IDataView data, Stream outputStream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(outputStream, nameof(outputStream));

            using (var ch = env.Start("Saving data"))
            {
                var saver = new BinarySaver(env, new BinarySaver.Arguments());
                DataSaverUtils.SaveDataView(ch, saver, data, outputStream);
            }
        }

        public static void SaveData(IHostEnvironment env, IDataView data, string fileName)
        {
            Contracts.CheckNonEmpty(fileName, nameof(fileName));
            using (var stream = File.Create(fileName))
                SaveData(env, data, stream);
        }
    }
}
