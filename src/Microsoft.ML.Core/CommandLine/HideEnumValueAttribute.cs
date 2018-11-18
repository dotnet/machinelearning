// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.CommandLine
{
    /// <summary>
    /// On an enum value - indicates that the value should not be shown in help or UI.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    [BestFriend]
    internal class HideEnumValueAttribute : Attribute
    {
        public HideEnumValueAttribute()
        {
        }
    }
}