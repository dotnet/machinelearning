// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// This is a signature for classes that are 'holders' of entry points and components.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureEntryPointModule();

    /// <summary>
    /// A simplified assembly attribute for marking EntryPoint modules.
    /// </summary>
    [AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
    [BestFriend]
    internal sealed class EntryPointModuleAttribute : LoadableClassAttributeBase
    {
        public EntryPointModuleAttribute(Type loaderType)
            : base(null, typeof(void), loaderType, null, new[] { typeof(SignatureEntryPointModule) }, loaderType.FullName)
        { }
    }
}
