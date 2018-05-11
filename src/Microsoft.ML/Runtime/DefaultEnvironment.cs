using Microsoft.ML.Runtime.Data;
using System;

namespace Microsoft.ML.Runtime
{
    public sealed class DefaultEnvironment : HostEnvironmentBase<DefaultEnvironment>
    {
        public DefaultEnvironment(int? seed = null, int conc = 0)
         : this(RandomUtils.Create(seed), true, conc)
        {
        }

        public DefaultEnvironment(IRandom rand, bool verbose, int conc, string shortName = null, string parentFullName = null) : base(rand, verbose, conc, shortName, parentFullName)
        {
            EnsureDispatcher<ChannelMessage>();
            AddListener<ChannelMessage>(OnMessageRecieved);
        }

        void OnMessageRecieved(IMessageSource sender, ChannelMessage msg)
        {
            ChannelMessageEventArgs eventArgs = new ChannelMessageEventArgs() { Message = msg };
            MessageRecieved?.Invoke(this, eventArgs);
        }

        public event EventHandler<ChannelMessageEventArgs> MessageRecieved;
        public class ChannelMessageEventArgs : EventArgs
        {
            public ChannelMessage Message { get; set; }
        }

        private sealed class Channel : ChannelBase
        {
            public Channel(DefaultEnvironment master, ChannelProviderBase parent, string shortName, Action<IMessageSource, ChannelMessage> dispatch)
                : base(master, parent, shortName, dispatch)
            {
            }
        }

        private sealed class Host : HostBase
        {
            public new bool IsCancelled => Root.IsCancelled;

            public Host(HostEnvironmentBase<DefaultEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
                : base(source, shortName, parentFullName, rand, verbose, conc)
            {
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

            protected override IHost RegisterCore(HostEnvironmentBase<DefaultEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
            {
                return new Host(source, shortName, parentFullName, rand, verbose, conc);
            }
        }

        protected override IHost RegisterCore(HostEnvironmentBase<DefaultEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
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
            Contracts.Assert(parent is DefaultEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is DefaultEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }
    }

}
