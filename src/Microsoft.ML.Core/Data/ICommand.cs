// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Command
{
    /// <summary>
    /// The signature for commands.
    /// </summary>
    public delegate void SignatureCommand();

    public interface ICommand
    {
        void Run();
    }
}
