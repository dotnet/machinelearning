// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    /// <summary>
    /// An ML.NET environment for local execution.
    /// </summary>
    internal sealed class LocalEnvironment : HostEnvironmentBase<LocalEnvironment>
    {
        private sealed class Channel : ChannelBase
        {
            public readonly Stopwatch Watch;
            public Channel(LocalEnvironment root, ChannelProviderBase parent, string shortName,
                Action<IMessageSource, ChannelMessage> dispatch)
                : base(root, parent, shortName, dispatch)
            {
                Watch = Stopwatch.StartNew();
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel started"));
            }

            private void ChannelFinished()
                => Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel finished. Elapsed {0:c}.", Watch.Elapsed));

            protected override void Dispose(bool disposing)
            {
                if(disposing)
                {
                    ChannelFinished();
                    Watch.Stop();

                    Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel disposed"));
                }

                base.Dispose(disposing);
            }
        }

        /// <summary>
        /// Create an ML.NET <see cref="IHostEnvironment"/> for local execution.
        /// </summary>
        /// <param name="seed">Random seed. Set to <c>null</c> for a non-deterministic environment.</param>
        public LocalEnvironment(int? seed = null)
            : base(RandomUtils.Create(seed), verbose: false)
        {
        }

        /// <summary>
        /// Add a custom listener to the messages of ML.NET components.
        /// </summary>
        public void AddListener(Action<IMessageSource, ChannelMessage> listener)
            => AddListener<ChannelMessage>(listener);

        /// <summary>
        /// Remove a previously added a custom listener.
        /// </summary>
        public void RemoveListener(Action<IMessageSource, ChannelMessage> listener)
            => RemoveListener<ChannelMessage>(listener);

        protected override IHost RegisterCore(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose);
        }

        protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is LocalEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is LocalEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }

        private sealed class Host : HostBase
        {
            public Host(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
                : base(source, shortName, parentFullName, rand, verbose)
            {
                IsCanceled = source.IsCanceled;
            }

            protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Channel(Root, parent, name, GetDispatchDelegate<ChannelMessage>());
            }

            protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
            }

            protected override IHost RegisterCore(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
            {
                return new Host(source, shortName, parentFullName, rand, verbose);
            }
        }
    }

}
