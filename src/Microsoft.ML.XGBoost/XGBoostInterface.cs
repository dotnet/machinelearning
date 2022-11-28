// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.XGBoost
{
    public class XGBoostUtils
    {

        public static System.Version XgbMajorVersion()
        {
            int major;
            int minor;
            int patch;
            WrappedXGBoostInterface.XGBoostVersion(out major, out minor, out patch);
            return new System.Version(major, minor, patch);
        }

        // TODO: Should probably return a dictionary by parsing the JSON output
        public static string BuildInfo()
        {
            // should probably check this doesn't return an error
            unsafe
            {
                byte* resultPtr;
                WrappedXGBoostInterface.XGBuildInfo(&resultPtr);
                // this uses ANSI on Windows and non-ANSI on other OSs, so use Marshal.PtrToStringUTF8 instead
                // string result = new string((sbyte*)resultPtr);
                string result = Marshal.PtrToStringUTF8((nint)resultPtr) ?? "";
                return result;
            }
        }
    }

#pragma warning disable MSML_GeneralName
    public class XGBoostDLLException : Exception
#pragma warning restore MSML_GeneralName
    {
        public XGBoostDLLException()
        {
            /* empty */
        }

        public XGBoostDLLException(string message)
          : base(message)
        {
            /* empty */
        }

    }
}
