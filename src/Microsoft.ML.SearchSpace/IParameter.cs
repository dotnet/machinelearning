// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.ModelBuilder.SearchSpace.Converter;
using Newtonsoft.Json;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    public enum ParameterType
    {
        Integer = 0,
        Float = 1,
        Bool = 2,
        String = 3,
        Object = 4,
        Array = 5,
    }

    [JsonConverter(typeof(ParameterConverter))]
    public interface IParameter : IDictionary<string, IParameter>
    {
        ParameterType ParameterType { get; }

        object Value { get; }

        T AsType<T>();
    }
}
