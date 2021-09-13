// <copyright file="SearchSpace{T}.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    public class SearchSpace<T> : SearchSpace
        where T : class, new()
    {
        public SearchSpace()
            : base(typeof(T))
        {
        }

        public new T SampleFromFeatureSpace(double[] feature)
        {
            var param = base.SampleFromFeatureSpace(feature);
            return param.AsType<T>();
        }

        public double[] MappingToFeatureSpace(T input)
        {
            var param = Parameter.CreateFromInstance(input);
            return this.MappingToFeatureSpace(param);
        }
    }
}
