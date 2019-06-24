// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Provides options for ML.NET objects.
    /// </summary>
    public class MLOptions
    {
        private MLContext _context;

        /// <summary>
        /// Initializes a new instance of <see cref="MLOptions"/>.
        /// </summary>
        public MLOptions()
        {
        }

        /// <summary>
        /// The <see cref="MLContext "/> which all the ML.NET operations happen.
        /// </summary>
        public MLContext MLContext
        {
            get { return _context ?? (_context = new MLContext()); }
            set { _context = value; }
        }
    }

    /// <summary>
    /// Configures the <see cref="MLOptions"/> type.
    /// </summary>
    /// <remarks>
    /// Note: This is run after all <see cref="IConfigureOptions{MLContextOptions}"/>.
    /// </remarks>
    internal class PostMLContextOptionsConfiguration : IPostConfigureOptions<MLOptions>
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
        public void PostConfigure(string name, MLOptions options)
        {
            options.MLContext.Log += Log;
        }

        private void Log(object sender, LoggingEventArgs e)
        {
            _logger.LogTrace(e.Message);
        }
    }
}
