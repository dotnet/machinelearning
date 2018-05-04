// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public class TestLruCache
    {
        [Fact]
        public void EntryLruCache()
        {
            var cache = new LruCache<string, int>(2);
            var keys = cache.Keys.ToArray();
            Assert.Empty(keys);

            cache.Add("foo", 1);
            keys = cache.Keys.ToArray();
            Assert.Single(keys);
            Assert.Equal("foo", keys[0]);

            cache.Add("bar", 2);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("bar", keys[0]);
            Assert.Equal("foo", keys[1]);

            cache.Add("baz", 3);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("baz", keys[0]);
            Assert.Equal("bar", keys[1]);

            int val;
            bool success = cache.TryGetValue("bar", out val);
            Assert.True(success);
            Assert.Equal(2, val);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("bar", keys[0]);
            Assert.Equal("baz", keys[1]);

            success = cache.TryGetValue("bar", out val);
            Assert.True(success);
            Assert.Equal(2, val);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("bar", keys[0]);
            Assert.Equal("baz", keys[1]);

            success = cache.TryGetValue("baz", out val);
            Assert.True(success);
            Assert.Equal(3, val);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("baz", keys[0]);
            Assert.Equal("bar", keys[1]);

            success = cache.TryGetValue("foo", out val);
           Assert.False(success);
            keys = cache.Keys.ToArray();
            Assert.Equal(2, keys.Length);
            Assert.Equal("baz", keys[0]);
            Assert.Equal("bar", keys[1]);
        }
    }
}