// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.AutoML
{
    internal class ModelContainer
    {
        private readonly MLContext _mlContext;
        private readonly FileInfo _fileInfo;

        internal ModelContainer(MLContext mlContext, FileInfo fileInfo, ITransformer model, DataViewSchema modelInputSchema)
        {
            _mlContext = mlContext;
            _fileInfo = fileInfo;

            // Write model to disk
            using (var fs = File.Create(fileInfo.FullName))
            {
                _mlContext.Model.Save(model, modelInputSchema, fs);
            }
            // Dispose model and free C Tensor objects as model has been saved to disk
            (model as IDisposable)?.Dispose();
        }

        public ITransformer GetModel()
        {
            // Load model from disk
            ITransformer model;
            using (var stream = new FileStream(_fileInfo.FullName, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                model = _mlContext.Model.Load(stream, out var modelInputSchema);
            }
            return model;
        }
    }
}
