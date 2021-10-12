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
        private T defaultOption = null;
        public SearchSpace()
            : base(typeof(T))
        {
        }

        public SearchSpace(T defaultOption)
            : base(typeof(T), new Parameter(defaultOption))
        {
            this.defaultOption = defaultOption;
        }

        public new T SampleFromFeatureSpace(double[] feature)
        {
            var param = base.SampleFromFeatureSpace(feature);
            var option = param.AsType<T>();

            return option;
        }

        public double[] MappingToFeatureSpace(T input)
        {
            var param = new Parameter(input);
            return this.MappingToFeatureSpace(param);
        }
    }
}
