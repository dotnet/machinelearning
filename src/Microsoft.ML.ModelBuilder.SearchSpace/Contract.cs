// <copyright file="Contract.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    internal static class Contract
    {
        public static void Requires(bool condition, string msg)
        {
            if (!condition)
            {
                throw new Exception(msg);
            }
        }
    }
}
