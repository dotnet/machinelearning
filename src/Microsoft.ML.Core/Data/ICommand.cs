// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Command
{
    /// <summary>
    /// The signature for commands.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureCommand();

    [BestFriend]
    internal interface ICommand
    {
        void Run();
    }
}
