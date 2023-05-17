// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.Fairlearn
{
    public static class Utilities
    {
        public static SearchSpace.SearchSpace GenerateBinaryClassificationLambdaSearchSpace(Moment moment, float gridLimit = 10, bool negativeAllowed = true)
        {
            var searchSpace = new SearchSpace.SearchSpace();
            var sensitiveFeatureColumnValue = moment.SensitiveFeatureColumn.Cast<string>().Distinct();

            // for different_bound only
            // if sensitive feature column value is "a", "b", "c",
            // the search space will contains 6 options with name format {sensitive column value}_{pos/neg}
            // a_pos, a_neg, b_pos, b_neg, c_pos, c_neg.
            var rand = new Random();
            foreach (var p in from _groupValue in sensitiveFeatureColumnValue
                              from _indicator in new[] { "pos", "neg" }
                              select new { _groupValue, _indicator })
            {
                var option = new UniformSingleOption(-gridLimit, gridLimit, defaultValue: Convert.ToSingle(rand.NextDouble()) * 2.0f * gridLimit - gridLimit);
                var optionName = $"{p._groupValue}_{p._indicator}";
                searchSpace[optionName] = option;
            }

            return searchSpace;
        }
    }
}
