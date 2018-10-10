// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Data
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    /// <summary>
    /// An ML.NET environment for local execution.
    /// </summary>
    public sealed class LocalEnvironment : HostEnvironmentBase<LocalEnvironment>
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
                => Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel finished. Elapsed { 0:c }.", Watch.Elapsed));

            protected override void DisposeCore()
            {
                ChannelFinished();
                Watch.Stop();

                Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel disposed"));
                base.DisposeCore();
            }
        }

        /// <summary>
        /// Create an ML.NET <see cref="IHostEnvironment"/> for local execution.
        /// </summary>
        /// <param name="seed">Random seed. Set to <c>null</c> for a non-deterministic environment.</param>
        /// <param name="conc">Concurrency level. Set to 1 to run single-threaded. Set to 0 to pick automatically.</param>
        public LocalEnvironment(int? seed = null, int conc = 0)
            : base(RandomUtils.Create(seed), verbose: false, conc)
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

        protected override IFileHandle CreateTempFileCore(IHostEnvironment env, string suffix = null, string prefix = null)
            => base.CreateTempFileCore(env, suffix, "Local_" + prefix);

        protected override IHost RegisterCore(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose, conc);
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
            public Host(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
                : base(source, shortName, parentFullName, rand, verbose, conc)
            {
                IsCancelled = source.IsCancelled;
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

            protected override IHost RegisterCore(HostEnvironmentBase<LocalEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
            {
                return new Host(source, shortName, parentFullName, rand, verbose, conc);
            }
        }
    }

}
