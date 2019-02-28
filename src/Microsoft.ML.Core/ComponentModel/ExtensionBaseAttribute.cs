// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML
{
    /// <summary>
    /// The base attribute type for all attributes used for extensibility purposes.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public abstract class ExtensionBaseAttribute : Attribute
    {
        public string ContractName { get; }

        [BestFriend]
        private protected ExtensionBaseAttribute(string contractName)
        {
            ContractName = contractName;
        }
    }
}
