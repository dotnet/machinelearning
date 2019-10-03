// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;
using Xunit;

namespace Microsoft.Extensions.ML
{
    public class UriLoaderTests
    {
        [Fact]
        public void throw_until_started()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<UriModelLoader>(sp);
            Assert.Throws<InvalidOperationException>(() => loaderUnderTest.GetModel());
            Assert.Throws<InvalidOperationException>(() => loaderUnderTest.GetReloadToken());
        }

        [Fact]
        public void can_reload_model()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<UriLoaderMock>(sp);
            loaderUnderTest.Start(new Uri("http://microsoft.com"), TimeSpan.FromMilliseconds(1));

            var changed = false;
            var changeTokenRegistration = ChangeToken.OnChange(
                        () => loaderUnderTest.GetReloadToken(),
                        () => changed = true);
            Thread.Sleep(30);
            Assert.True(changed);
        }

        [Fact]
        public void no_reload_no_change()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<UriLoaderMock>(sp);

            loaderUnderTest.ETagMatches = (a,b) => true;

            loaderUnderTest.Start(new Uri("http://microsoft.com"), TimeSpan.FromMilliseconds(1));

            var changed = false;
            var changeTokenRegistration = ChangeToken.OnChange(
                        () => loaderUnderTest.GetReloadToken(),
                        () => changed = true);
            Thread.Sleep(30);
            Assert.False(changed);
        }
    }

    class UriLoaderMock : UriModelLoader
    {
        public Func<Uri, string, bool> ETagMatches { get; set; } = (_, __) => false;

        public UriLoaderMock(IOptions<MLOptions> contextOptions,
                         ILogger<UriModelLoader> logger) : base(contextOptions, logger)
        {
        }

        public override ITransformer GetModel()
        {
            return null;
        }

        internal override Task<bool> LoadModel()
        {
            return Task.FromResult(true);
        }

        internal override Task<bool> MatchEtag(Uri uri, string eTag)
        {
            return Task.FromResult(ETagMatches(uri, eTag));
        }
    }
}
