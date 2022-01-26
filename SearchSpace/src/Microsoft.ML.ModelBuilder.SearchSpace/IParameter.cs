// <copyright file="Parameter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

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
