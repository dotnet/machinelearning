﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML.Utils
{
    public static class SummaryExtension
    {
        public static string Summary(this IEstimator<ITransformer> estimator)
        {
            var schema = estimator switch
            {
                ICanSummarize summarizable => summarizable.Summarize(),
                _ => new Schema
                {
                    Name = estimator.GetType().Name,
                },
            };

            return schema.ToString();
        }
    }
}
