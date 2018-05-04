// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
#pragma warning disable TLC_GeneralName // This structure should be deprecated anyway.
    // REVIEW: Get rid of this. Everything should be in the ArgumentAttribute (or a class
    // derived from ArgumentAttribute).
    [AttributeUsage(AttributeTargets.Field)]
    public class TGUIAttribute : Attribute
#pragma warning restore TLC_GeneralName
    {
        // Display parameters
        public string Label { get; set; }
        public string Description { get; set; }
        public bool IsSaveFileName { get; set; }
        public bool IsFolder { get; set; }
        public bool ShowPreviewIcon { get; set; }
        public string OutputFilenameTemplate { get; set; }

        // REVIEW: this is not ideal as it'd be hard to improve the sweep syntax
        public string SuggestedSweeps { get; set; }

        public bool RegistryBacked { get; set; }

        public bool NotGui { get; set; }

        public bool NoSweep { get; set; }

        //Settings are automatically populated for fields that are classes.
        //The below is an extension of the framework to add settings for 
        //boolean type fields.
        public bool ShowSettingsForCheckbox { get; set; }
        public object Settings { get; set; }
    }
}
