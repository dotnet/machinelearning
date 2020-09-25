// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if !CPUMATH_INFRASTRUCTURE
// This namespace contains the PublicKey generally used in ML.NET project settings
namespace Microsoft.ML
#else
// CpuMath module has its own PublicKey for isolating itself from Microsoft.ML.Core
// Note that CpuMath uses its own BestFriend defined in Microsoft.ML.Internal.CpuMath.Core.
namespace Microsoft.ML.Internal.CpuMath.Core
#endif
{
    [BestFriend]
    internal static class PublicKey
    {
        public const string Value = ", PublicKey=00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbbad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
    }

    [BestFriend]
    internal static class InternalPublicKey
    {
        public const string Value = ", PublicKey=0024000004800000940000000602000000240000525341310004000001000100bd8dded65b44bf8183068bd6dae3b68ba499202b2909640604cf63c7c0ea95bec94a400af533d1132e0dba214f310f666486b50ea91f2697a4fe331eb6a8d7306029344e320dabb7c4c3617472e3088c28dbfcf761a3f1b954a2a64cb865aae873b1d3c3cab344661cd7d5929d1043912908b8dd321889ca11f29d6bf9b9b9a9";
    }
}
