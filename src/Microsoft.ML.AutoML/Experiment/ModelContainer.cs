// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.ML.AutoML
{
    internal class ModelContainer
    {
        private readonly MLContext _mlContext;
        private readonly FileInfo _fileInfo;
        private readonly ITransformer _model;

        internal ModelContainer(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
        }

        internal ModelContainer(MLContext mlContext, FileInfo fileInfo, ITransformer model, DataViewSchema modelInputSchema)
        {
            _mlContext = mlContext;
            _fileInfo = fileInfo;

            // Write model to disk
            using (var fs = File.Create(fileInfo.FullName))
            {
                _mlContext.Model.Save(model, modelInputSchema, fs);
            }
        }

        public ITransformer GetModel()
        {
            // If model stored in memory, return it
            if (_model != null)
            {
                return _model;
            }

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
