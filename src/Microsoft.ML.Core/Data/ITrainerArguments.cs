// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime
{
    // This is basically a no-op interface put in primarily
    // for backward binary compat support for AFx.
    // REVIEW: This interface was removed in TLC 3.0 as part of the 
    // deprecation of the *Factory interfaces, but added back as a temporary
    // hack. Remove it asap.
    public interface ITrainerArguments
    {
    }
}
