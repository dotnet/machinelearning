// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
