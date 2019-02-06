// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CommandLine;

namespace Microsoft.ML.Data
{
    public class Range
    {
        public Range() { }

        /// <summary>
        /// A range representing a single value. Will result in a scalar column.
        /// </summary>
        /// <param name="index">The index of the field of the text file to read.</param>
        public Range(int index)
        {
            Contracts.CheckParam(index >= 0, nameof(index), "Must be non-negative");
            Min = index;
            Max = index;
        }

        /// <summary>
        /// A range representing a set of values. Will result in a vector column.
        /// </summary>
        /// <param name="min">The minimum inclusive index of the column.</param>
        /// <param name="max">The maximum-inclusive index of the column. If <c>null</c>
        /// indicates that the caller should auto-detect the length
        /// of the lines, and read till the end.</param>
        public Range(int min, int? max)
        {
            Contracts.CheckParam(min >= 0, nameof(min), "Must be non-negative");
            Contracts.CheckParam(!(max < min), nameof(max), "If specified, must be greater than or equal to " + nameof(min));

            Min = min;
            Max = max;
            // Note that without the following being set, in the case where there is a single range
            // where Min == Max, the result will not be a vector valued but a scalar column.
            AutoEnd = max == null;
        }

        [Argument(ArgumentType.Required, HelpText = "First index in the range")]
        public int Min;

        // If max is specified, the fields autoEnd and variableEnd are ignored.
        // Otherwise, if autoEnd is true, then variableEnd is ignored.
        [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
        public int? Max;

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "This range extends to the end of the line, but should be a fixed number of items",
            ShortName = "auto")]
        public bool AutoEnd;

        [Argument(ArgumentType.AtMostOnce, HelpText = "This range includes only other indices not specified", ShortName = "other")]
        public bool AllOther;

        internal static Range Parse(string str)
        {
            Contracts.AssertNonEmpty(str);

            var res = new Range();
            if (res.TryParse(str))
                return res;
            return null;
        }

        private bool TryParse(string str)
        {
            Contracts.AssertNonEmpty(str);

            int ich = str.IndexOfAny(new char[] { '-', '~' });
            if (ich < 0)
            {
                // No "-" or "~". Single integer.
                if (!int.TryParse(str, out Min))
                    return false;
                Max = Min;
                return true;
            }

            AllOther = str[ich] == '~';

            if (ich == 0)
            {
                if (!AllOther)
                    return false;

                Min = 0;
            }
            else if (!int.TryParse(str.Substring(0, ich), out Min))
                return false;

            string rest = str.Substring(ich + 1);
            if (string.IsNullOrEmpty(rest) || rest == "*")
            {
                AutoEnd = true;
                return true;
            }

            int tmp;
            if (!int.TryParse(rest, out tmp))
                return false;
            Max = tmp;

            if (Min > Max)
                return false;

            return true;
        }
    }
}
