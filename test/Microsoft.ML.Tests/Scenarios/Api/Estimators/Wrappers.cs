// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public class MyLambdaTransform<TSrc, TDst> : IEstimator<TransformWrapper>
         where TSrc : class, new()
            where TDst : class, new()
    {
        private readonly IHostEnvironment _env;
        private readonly Action<TSrc, TDst> _action;

        public MyLambdaTransform(IHostEnvironment env, Action<TSrc, TDst> action)
        {
            _env = env;
            _action = action;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = LambdaTransform.CreateMap(_env, input, _action);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public static class MyHelperExtensions
    {
        public static void SaveAsBinary(this IDataView data, IHostEnvironment env, Stream stream)
        {
            var saver = new BinarySaver(env, new BinarySaver.Arguments());
            using (var ch = env.Start("SaveData"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream);
        }

        public static IDataView FitAndTransform(this IEstimator<ITransformer> est, IDataView data) => est.Fit(data).Transform(data);

        public static IDataView FitAndRead<TSource>(this IDataReaderEstimator<TSource, IDataReader<TSource>> est, TSource source)
            => est.Fit(source).Read(source);
    }
}
