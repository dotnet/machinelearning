// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public class TestHosts
    {
        [Fact]
        public void TestCancellation()
        {
            var env = new TlcEnvironment(seed: 42);
            for (int z = 0; z < 1000; z++)
            {
                var mainHost = env.Register("Main");
                var children = new ConcurrentDictionary<IHost, List<IHost>>();
                var hosts = new BlockingCollection<Tuple<IHost, int>>();
                hosts.Add(new Tuple<IHost, int>(mainHost.Register("1"), 1));
                hosts.Add(new Tuple<IHost, int>(mainHost.Register("2"), 1));
                hosts.Add(new Tuple<IHost, int>(mainHost.Register("3"), 1));
                hosts.Add(new Tuple<IHost, int>(mainHost.Register("4"), 1));
                hosts.Add(new Tuple<IHost, int>(mainHost.Register("5"), 1));

                int iterations = 100;
                Random rand = new Random();
                var addThread = new Thread(
                () =>
                {
                    for (int i = 0; i < iterations; i++)
                    {
                        var randHostTuple = hosts.ElementAt(rand.Next(hosts.Count - 1));
                        var newHost = randHostTuple.Item1.Register((randHostTuple.Item2 + 1).ToString());
                        hosts.Add(new Tuple<IHost, int>(newHost, randHostTuple.Item2 + 1));
                        if (!children.ContainsKey(randHostTuple.Item1))
                            children[randHostTuple.Item1] = new List<IHost>();
                        else
                            children[randHostTuple.Item1].Add(newHost);
                    }
                });
                addThread.Start();
                Queue<IHost> queue = new Queue<IHost>();
                for (int i = 0; i < 5; i++)
                {
                    IHost rootHost = null;
                    var index = 0;
                    do
                    {
                        index = rand.Next(hosts.Count);
                    } while (hosts.ElementAt(index).Item1.IsCancelled || hosts.ElementAt(index).Item2 < 3);
                    hosts.ElementAt(index).Item1.StopExecution();
                    rootHost = hosts.ElementAt(index).Item1;
                    queue.Enqueue(rootHost);
                }
                addThread.Join();
                while (queue.Count > 0)
                {
                    var currentHost = queue.Dequeue();
                    Assert.True(currentHost.IsCancelled);

                    if (children.ContainsKey(currentHost))
                        children[currentHost].ForEach(x => queue.Enqueue(x));
                }
            }
        }
    }
}
