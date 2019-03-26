// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public class TestHosts
    {
        [Fact]
        public void TestCancellation()
        {
            IHostEnvironment env = new MLContext(seed: 42);
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
                    } while ((hosts.ElementAt(index).Item1 as ICancelable).IsCanceled || hosts.ElementAt(index).Item2 < 3);
                    (hosts.ElementAt(index).Item1 as ICancelable).CancelExecution();
                    rootHost = hosts.ElementAt(index).Item1;
                    queue.Enqueue(rootHost);
                }
                addThread.Join();
                while (queue.Count > 0)
                {
                    var currentHost = queue.Dequeue();
                    Assert.True((currentHost as ICancelable).IsCanceled);

                    if (children.ContainsKey(currentHost))
                        children[currentHost].ForEach(x => queue.Enqueue(x));
                }
            }
        }

        [Fact]
        public void TestCancellationApi()
        {
            IHostEnvironment env = new MLContext(seed: 42);
            var mainHost = env.Register("Main");
            var children = new ConcurrentDictionary<IHost, List<IHost>>();
            var hosts = new BlockingCollection<Tuple<IHost, int>>();
            hosts.Add(new Tuple<IHost, int>(mainHost.Register("1"), 1));
            hosts.Add(new Tuple<IHost, int>(mainHost.Register("2"), 1));
            hosts.Add(new Tuple<IHost, int>(mainHost.Register("3"), 1));
            hosts.Add(new Tuple<IHost, int>(mainHost.Register("4"), 1));
            hosts.Add(new Tuple<IHost, int>(mainHost.Register("5"), 1));

            for (int i = 0; i < 5; i++)
            {
                var tupple = hosts.ElementAt(i);
                var newHost = tupple.Item1.Register((tupple.Item2 + 1).ToString());
                hosts.Add(new Tuple<IHost, int>(newHost, tupple.Item2 + 1));
            }

            ((MLContext)env).CancelExecution();

            //Ensure all created hosts are cancelled.
            //5 parent and one child for each.
            Assert.Equal(10, hosts.Count);

            foreach (var host in hosts)
                Assert.True((host.Item1 as ICancelable).IsCanceled);
        }

        /// <summary>
        /// Tests that MLContext's Log event intercepts messages properly.
        /// </summary>
        [Fact]
        public void LogEventProcessesMessages()
        {
            var messages = new List<string>();

            var env = new MLContext();
            env.Log += (sender, e) => messages.Add(e.Message);

            // create a dummy text reader to trigger log messages
            env.Data.CreateTextLoader(new TextLoader.Options { Columns = new[] { new TextLoader.Column("TestColumn", DataKind.Single, 0) } });

            Assert.True(messages.Count > 0);
        }
    }
}
