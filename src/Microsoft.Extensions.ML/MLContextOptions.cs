// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Provides an options class for MLContext objects.
    /// </summary>
    public class MLContextOptions
    {
        /// <summary>
        /// Initializes a new instance of <see cref="MLContextOptions"/>.
        /// </summary>
        public MLContextOptions()
        {
            MLContext = new MLContext();
        }

        /// <summary>
        /// The <see cref="MLContext "/> which all the ML.NET operations happen.
        /// </summary>
        public MLContext MLContext { get; set; }
    }

    /// <summary>
    /// Configures the <see cref="MLContextOptions"/> type.
    /// </summary>
    /// <remarks>
    /// Note: This is run after all <see cref="IConfigureOptions{MLContextOptions}"/>.
    /// </remarks>
    public class PostMLContextOptionsConfiguration : IPostConfigureOptions<MLContextOptions>
    {
        private readonly ILogger<MLContext> _logger;

        /// <summary>
        /// Initializes a new instance of <see cref="PostMLContextOptionsConfiguration"/>.
        /// </summary>
        /// <param name="logger">The <see cref="ILogger"/> to write to.</param>
        public PostMLContextOptionsConfiguration(ILogger<MLContext> logger)
        {
            _logger = logger;
        }

        /// <inheritdoc />
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
