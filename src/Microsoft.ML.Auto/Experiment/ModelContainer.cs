// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
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

        internal ModelContainer(MLContext mlContext, FileInfo fileInfo, ITransformer model)
        {
            _mlContext = mlContext;
            _fileInfo = fileInfo;

            // Write model to disk
            using (var fs = File.Create(fileInfo.FullName))
            {
                model.SaveTo(mlContext, fs);
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
                model = _mlContext.Model.Load(stream);
            }
            return model;
        }
    }
}
