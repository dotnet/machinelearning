// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Console
    {
        public static int Main(string[] args)
        {
            string all = string.Join(" ", args);
            return Maml.MainAll(all);
        }

        public static unsafe int MainRaw(char* psz)
        {
            string args = new string(psz);
            return Maml.MainAll(args);
        }
    }
}
