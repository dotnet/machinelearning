// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    public class MLContextOptions
    {
        public MLContextOptions()
        {
            MLContext = new MLContext();
        }

        public MLContext MLContext { get; set; }
    }

    public class PostMLContextOptionsConfiguration : IPostConfigureOptions<MLContextOptions>
    {
        private readonly ILogger<MLContext> _logger;

        public PostMLContextOptionsConfiguration(ILogger<MLContext> logger)
        {
            _logger = logger;
        }

        public void PostConfigure(string name, MLContextOptions options)
        {
            options.MLContext.Log += Log;
        }

        private void Log(object sender, LoggingEventArgs e)
        {
            _logger.LogTrace(e.Message);
        }
    }
}
