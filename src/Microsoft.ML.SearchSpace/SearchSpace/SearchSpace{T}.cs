// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.SearchSpace
{
    public class SearchSpace<T> : SearchSpace
        where T : class, new()
    {
        private readonly T _defaultOption = null;

        public SearchSpace()
            : base(typeof(T))
        {
        }

        public SearchSpace(T defaultOption)
            : base(typeof(T), Parameter.FromObject(defaultOption))
        {
            this._defaultOption = defaultOption;
        }

        public new T SampleFromFeatureSpace(double[] feature)
        {
            var param = base.SampleFromFeatureSpace(feature);
            var option = param.AsType<T>();

            return option;
        }

        public double[] MappingToFeatureSpace(T input)
        {
            var param = Parameter.FromObject(input);
            return this.MappingToFeatureSpace(param);
        }
    }
}
