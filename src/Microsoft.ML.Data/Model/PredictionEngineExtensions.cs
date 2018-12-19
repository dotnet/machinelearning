// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods to create a prediction engine.
    /// </summary>
    public static class PredictionEngineExtensions
    {
        public static PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(this ITransformer transformer,
            IHostEnvironment env, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
            => new PredictionEngine<TSrc, TDst>(env, transformer, true, inputSchemaDefinition, outputSchemaDefinition);
    }
}
