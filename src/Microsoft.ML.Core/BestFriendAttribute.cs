// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Intended to be applied to types and members marked as internal to indicate that friend access of this
    /// internal item is OK from another assembly. This restriction applies only to assemblies that declare the
    /// <see cref="WantsToBeBestFriendsAttribute"/> assembly level attribute. Note that this attribute is not
    /// transferrable: an internal member with this attribute does not somehow make a containing internal type
    /// accessible. Conversely, neither does marking an internal type make any unmarked internal members accessible.
    /// </summary>
    [BestFriend]
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Field | AttributeTargets.Property | AttributeTargets.Constructor
        | AttributeTargets.Method | AttributeTargets.Interface | AttributeTargets.Enum | AttributeTargets.Delegate, AllowMultiple = false, Inherited = false)]
    internal sealed class BestFriendAttribute : Attribute
    {
    }

    /// <summary>
    /// This is an assembly level attribute to signal that friend accesses on this assembly should be checked
    /// for usage of <see cref="BestFriendAttribute"/>. If this attribute is missing, normal access rules for
    /// friends should apply.
    /// </summary>
    [BestFriend]
    [AttributeUsage(AttributeTargets.Assembly, AllowMultiple = false, Inherited = false)]
    internal sealed class WantsToBeBestFriendsAttribute : Attribute
    {
    }
}
