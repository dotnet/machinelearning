// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Base class for channel providers. This is a common base class for<see cref="HostEnvironmentBase{THostEnvironmentBase}"/>.
    /// The ParentFullName, ShortName, and FullName may be null or empty.
    /// </summary>
    [BestFriend]
    internal abstract class ChannelProviderBase : IExceptionContext
    {
        /// <summary>
        /// Data keys that are attached to the exception thrown via the exception context.
        /// </summary>
        public static class ExceptionContextKeys
        {
            public const string ThrowingComponent = "Throwing component";
            public const string ParentComponent = "Parent component";
            public const string Phase = "Phase";
        }

        public string ShortName { get; }
        public string ParentFullName { get; }
        public string FullName { get; }
        public bool Verbose { get; }

        /// <summary>
        /// The channel depth, NOT host env depth.
        /// </summary>
        public abstract int Depth { get; }

        /// <summary>
        /// ExceptionContext description.
        /// </summary>
        public virtual string ContextDescription => FullName;

        protected ChannelProviderBase(string shortName, string parentFullName, bool verbose)
        {
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertValueOrNull(shortName);

            ParentFullName = string.IsNullOrEmpty(parentFullName) ? null : parentFullName;
            ShortName = string.IsNullOrEmpty(shortName) ? null : shortName;
            FullName = GenerateFullName();
            Verbose = verbose;
        }

        /// <summary>
        /// Override this method to change the way full names are constructed.
        /// </summary>
        protected virtual string GenerateFullName()
        {
            if (string.IsNullOrEmpty(ParentFullName))
                return ShortName;
            return string.Format("{0}; {1}", ParentFullName, ShortName);
        }

        public virtual TException Process<TException>(TException ex)
            where TException : Exception
        {
            if (ex != null)
            {
                ex.Data[ExceptionContextKeys.ThrowingComponent] = ShortName;
                ex.Data[ExceptionContextKeys.ParentComponent] = ParentFullName;
                Contracts.Mark(ex);
            }
            return ex;
        }
    }

    /// <summary>
    /// Message source (a channel) that generated the message being dispatched.
    /// </summary>
    [BestFriend]
    internal interface IMessageSource
    {
        string ShortName { get; }
        string FullName { get; }
        bool Verbose { get; }
    }

    /// <summary>
    /// A basic host environment suited for many environments.
    /// This also supports modifying the concurrency factor, provides the ability to subscribe to pipes via the
    /// AddListener/RemoveListener methods, and exposes the <see cref="ProgressReporting.ProgressTracker"/> to
    /// query progress.
    /// </summary>
    [BestFriend]
    internal abstract class HostEnvironmentBase<TEnv> : ChannelProviderBase, IHostEnvironment, IChannelProvider, ICancelable
        where TEnv : HostEnvironmentBase<TEnv>
    {
        void ICancelable.CancelExecution()
        {
            lock (_cancelLock)
            {
                foreach (var child in _children)
                    if (child.TryGetTarget(out IHost host))
                        if (host is ICancelable cancelableHost)
                            cancelableHost.CancelExecution();

                _children.Clear();
                IsCanceled = true;
            }
        }

        /// <summary>
        /// Base class for hosts. Classes derived from  <see cref="HostEnvironmentBase{THostEnvironmentBase}"/> may choose
        /// to provide their own host class that derives from this class.
        /// This encapsulates the random number generator and name information.
        /// </summary>
        public abstract class HostBase : HostEnvironmentBase<TEnv>, IHost
        {
            public override int Depth { get; }

            public Random Rand => _rand;

            public HostBase(HostEnvironmentBase<TEnv> source, string shortName, string parentFullName, Random rand, bool verbose)
                : base(source, rand, verbose, shortName, parentFullName)
            {
                Depth = source.Depth + 1;
            }

            public new IHost Register(string name, int? seed = null, bool? verbose = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                IHost host;
                lock (_cancelLock)
                {
                    Random rand = (seed.HasValue) ? RandomUtils.Create(seed.Value) : RandomUtils.Create(_rand);
                    host = RegisterCore(this, name, Master?.FullName, rand, verbose ?? Verbose);
                    if (!IsCanceled)
                        _children.Add(new WeakReference<IHost>(host));
                }
                return host;
            }
        }

        /// <summary>
        /// Base class for implementing <see cref="IPipe{TMessage}"/>. Deriving classes can optionally override
        /// the Done() and the DisposeCore() methods. If no overrides are needed, the sealed class
        /// <see cref="Pipe{TMessage}"/> may be used.
        /// </summary>
        protected abstract class PipeBase<TMessage> : ChannelProviderBase, IPipe<TMessage>, IMessageSource
        {
            public override int Depth { get; }

            // The delegate to call to dispatch messages.
            protected readonly Action<IMessageSource, TMessage> Dispatch;

            public readonly ChannelProviderBase Parent;

            private bool _disposed;

            protected PipeBase(ChannelProviderBase parent, string shortName,
                Action<IMessageSource, TMessage> dispatch)
                : base(shortName, parent.FullName, parent.Verbose)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(dispatch);
                Parent = parent;
                Depth = parent.Depth + 1;
                Dispatch = dispatch;
            }

            public void Dispose()
            {
                if (!_disposed)
                {
                    Dispose(true);
                    _disposed = true;
                }
            }

            protected virtual void Dispose(bool disposing)
            {
            }

            public void Send(TMessage msg)
            {
                Dispatch(this, msg);
            }

            public override TException Process<TException>(TException ex)
            {
                if (ex != null)
                {
                    ex.Data[ExceptionContextKeys.ThrowingComponent] = Parent.ShortName;
                    ex.Data[ExceptionContextKeys.ParentComponent] = Parent.ParentFullName;
                    ex.Data[ExceptionContextKeys.Phase] = ShortName;
                    Contracts.Mark(ex);
                }
                return ex;
            }
        }

        /// <summary>
        /// A base class for <see cref="IChannel"/> implementations. A message is dispatched as a
        /// <see cref="ChannelMessage"/>. Deriving classes can optionally override the Done() and the
        /// DisposeCore() methods.
        /// </summary>
        protected abstract class ChannelBase : PipeBase<ChannelMessage>, IChannel
        {
            protected readonly TEnv Root;
            protected ChannelBase(TEnv root, ChannelProviderBase parent, string shortName,
                Action<IMessageSource, ChannelMessage> dispatch)
                : base(parent, shortName, dispatch)
            {
                Root = root;
            }

            public void Trace(MessageSensitivity sensitivity, string msg)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, sensitivity, msg));
            }

            public void Trace(MessageSensitivity sensitivity, string fmt, params object[] args)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, sensitivity, fmt, args));
            }

            public void Error(MessageSensitivity sensitivity, string msg)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Error, sensitivity, msg));
            }

            public void Error(MessageSensitivity sensitivity, string fmt, params object[] args)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Error, sensitivity, fmt, args));
            }

            public void Warning(MessageSensitivity sensitivity, string msg)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Warning, sensitivity, msg));
            }

            public void Warning(MessageSensitivity sensitivity, string fmt, params object[] args)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Warning, sensitivity, fmt, args));
            }

            public void Info(MessageSensitivity sensitivity, string msg)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Info, sensitivity, msg));
            }

            public void Info(MessageSensitivity sensitivity, string fmt, params object[] args)
            {
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Info, sensitivity, fmt, args));
            }
        }

        /// <summary>
        /// An optional implementation of <see cref="IPipe{TMessage}"/>.
        /// </summary>
        protected sealed class Pipe<TMessage> : PipeBase<TMessage>
        {
            public Pipe(ChannelProviderBase parent, string shortName,
                Action<IMessageSource, TMessage> dispatch) :
                base(parent, shortName, dispatch)
            {
            }
        }

        /// <summary>
        /// Base class for <see cref="Dispatcher{TMessage}"/>. The master host environment has a
        /// map from <see cref="System.Type"/> to <see cref="Dispatcher"/>.
        /// </summary>
        protected abstract class Dispatcher
        {
        }

        /// <summary>
        /// Strongly typed dispatcher class.
        /// </summary>
        protected sealed class Dispatcher<TMessage> : Dispatcher
        {
            /// <summary>
            /// This field is actually used as a <see cref="MulticastDelegate"/>, which holds the listener actions
            /// for all listeners that are currently subscribed. The action itself is an immutable object, so every time
            /// any listener subscribes or unsubscribes, the field is replaced with a modified version of the delegate.
            ///
            /// The field can be null, if no listener is currently subscribed.
            /// </summary>
            private volatile Action<IMessageSource, TMessage> _listenerAction;

            /// <summary>
            /// The dispatch delegate invokes the current dispatching action (wchch calls all current listeners).
            /// </summary>
            private readonly Action<IMessageSource, TMessage> _dispatch;

            public Dispatcher()
            {
                _dispatch = DispatchCore;
            }

            public Action<IMessageSource, TMessage> Dispatch { get { return _dispatch; } }

            private void DispatchCore(IMessageSource sender, TMessage message)
            {
                _listenerAction?.Invoke(sender, message);
            }

            public void AddListener(Action<IMessageSource, TMessage> listenerFunc)
            {
                lock (_dispatch)
                    _listenerAction += listenerFunc;
            }

            public void RemoveListener(Action<IMessageSource, TMessage> listenerFunc)
            {
                lock (_dispatch)
                    _listenerAction -= listenerFunc;
            }
        }

        protected readonly TEnv Root;
        // This is non-null iff this environment was a fork of another. Disposing a fork
        // doesn't free temp files. That is handled when the master is disposed.
        protected readonly HostEnvironmentBase<TEnv> Master;

        // Protect _cancellation logic.
        private readonly object _cancelLock;

        // The random number generator for this host.
        private readonly Random _rand;
        // A dictionary mapping the type of message to the Dispatcher that gets the strongly typed dispatch delegate.
        protected readonly ConcurrentDictionary<Type, Dispatcher> ListenerDict;

        protected readonly ProgressReporting.ProgressTracker ProgressTracker;

        public ComponentCatalog ComponentCatalog { get; }

        public override int Depth => 0;

        public bool IsCanceled { get; protected set; }

        // We don't have dispose mechanism for hosts, so to let GC collect children hosts we make them WeakReference.
        private readonly List<WeakReference<IHost>> _children;

        /// <summary>
        ///  The main constructor.
        /// </summary>
        protected HostEnvironmentBase(Random rand, bool verbose,
            string shortName = null, string parentFullName = null)
            : base(shortName, parentFullName, verbose)
        {
            Contracts.CheckValueOrNull(rand);
            _rand = rand ?? RandomUtils.Create();
            ListenerDict = new ConcurrentDictionary<Type, Dispatcher>();
            ProgressTracker = new ProgressReporting.ProgressTracker(this);
            _cancelLock = new object();
            Root = this as TEnv;
            ComponentCatalog = new ComponentCatalog();
            _children = new List<WeakReference<IHost>>();
        }

        /// <summary>
        /// This constructor is for forking.
        /// </summary>
        protected HostEnvironmentBase(HostEnvironmentBase<TEnv> source, Random rand, bool verbose,
            string shortName = null, string parentFullName = null)
            : base(shortName, parentFullName, verbose)
        {
            Contracts.CheckValue(source, nameof(source));
            Contracts.CheckValueOrNull(rand);
            _rand = rand ?? RandomUtils.Create();
            _cancelLock = new object();

            // This fork shares some stuff with the master.
            Master = source;
            Root = source.Root;
            ListenerDict = source.ListenerDict;
            ProgressTracker = source.ProgressTracker;
            ComponentCatalog = source.ComponentCatalog;
            _children = new List<WeakReference<IHost>>();
        }

        public IHost Register(string name, int? seed = null, bool? verbose = null)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            IHost host;
            lock (_cancelLock)
            {
                Random rand = (seed.HasValue) ? RandomUtils.Create(seed.Value) : RandomUtils.Create(_rand);
                host = RegisterCore(this, name, Master?.FullName, rand, verbose ?? Verbose);
                _children.Add(new WeakReference<IHost>(host));
            }
            return host;
        }

        protected abstract IHost RegisterCore(HostEnvironmentBase<TEnv> source, string shortName,
            string parentFullName, Random rand, bool verbose);

        public IProgressChannel StartProgressChannel(string name)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            return StartProgressChannelCore(null, name);
        }

        protected virtual IProgressChannel StartProgressChannelCore(HostBase host, string name)
        {
            Contracts.AssertNonEmpty(name);
            Contracts.AssertValueOrNull(host);
            return new ProgressReporting.ProgressChannel(this, ProgressTracker, name);
        }

        private void DispatchMessageCore<TMessage>(
            Action<IMessageSource, TMessage> listenerAction, IMessageSource channel, TMessage message)
        {
            Contracts.AssertValueOrNull(listenerAction);
            Contracts.AssertValue(channel);
            listenerAction?.Invoke(channel, message);
        }

        protected Action<IMessageSource, TMessage> GetDispatchDelegate<TMessage>()
        {
            var dispatcher = EnsureDispatcher<TMessage>();
            return dispatcher.Dispatch;
        }

        /// <summary>
        /// This method is called when a channel is created and when a listener is registered.
        /// This method is not invoked on every message.
        /// </summary>
        protected Dispatcher<TMessage> EnsureDispatcher<TMessage>()
        {
            if (!ListenerDict.TryGetValue(typeof(TMessage), out Dispatcher dispatcher)
                && !ListenerDict.TryAdd(typeof(TMessage), dispatcher = new Dispatcher<TMessage>()))
            {
                // TryAdd can only fail if some other thread won a race against us and inserted its own dispatcher into the dictionary.
                // Defer to that winning item.
                dispatcher = ListenerDict[typeof(TMessage)];
            }

            Contracts.Assert(dispatcher is Dispatcher<TMessage>);
            return (Dispatcher<TMessage>)dispatcher;
        }

        public IChannel Start(string name)
        {
            return CreateCommChannel(this, name);
        }

        public IPipe<TMessage> StartPipe<TMessage>(string name)
        {
            return CreatePipe<TMessage>(this, name);
        }

        protected abstract IChannel CreateCommChannel(ChannelProviderBase parent, string name);

        protected abstract IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name);

        public void AddListener<TMessage>(Action<IMessageSource, TMessage> listenerFunc)
        {
            Contracts.CheckValue(listenerFunc, nameof(listenerFunc));
            var dispatcher = EnsureDispatcher<TMessage>();
            dispatcher.AddListener(listenerFunc);
        }

        public void RemoveListener<TMessage>(Action<IMessageSource, TMessage> listenerFunc)
        {
            Contracts.CheckValue(listenerFunc, nameof(listenerFunc));
            if (!ListenerDict.TryGetValue(typeof(TMessage), out Dispatcher dispatcher))
                return;
            var typedDispatcher = dispatcher as Dispatcher<TMessage>;
            Contracts.AssertValue(typedDispatcher);
            typedDispatcher.RemoveListener(listenerFunc);
        }

        public override TException Process<TException>(TException ex)
        {
            Contracts.AssertValueOrNull(ex);
            if (ex != null)
            {
                ex.Data[ExceptionContextKeys.ThrowingComponent] = "Environment";
                Contracts.Mark(ex);
            }
            return ex;
        }

        public override string ContextDescription => "HostEnvironment";

        /// <summary>
        /// Line endings in message may not be normalized, this method provides normalized printing.
        /// </summary>
        /// <param name="writer">The text writer to write to.</param>
        /// <param name="message">The message, which if it contains newlines will be normalized.</param>
        /// <param name="removeLastNewLine">If false, then two newlines will be printed at the end,
        /// making messages be bracketed by blank lines. If true then only the single newline at the
        /// end of a message is printed.</param>
        /// <param name="prefix">A prefix that will be written to every line, except the first line.
        /// If <paramref name="message"/> contains no newlines then this prefix will not be
        /// written at all. This prefix is not written to the newline written if
        /// <paramref name="removeLastNewLine"/> is false.</param>
        public virtual void PrintMessageNormalized(TextWriter writer, string message, bool removeLastNewLine, string prefix = null)
        {
            int ichMin = 0;
            int ichLim = 0;
            for (; ; )
            {
                ichLim = ichMin;
                while (ichLim < message.Length && message[ichLim] != '\r' && message[ichLim] != '\n')
                    ichLim++;

                if (ichLim == message.Length)
                    break;

                if (prefix != null && ichMin > 0)
                    writer.Write(prefix);
                if (ichMin == ichLim)
                    writer.WriteLine();
                else
                    writer.WriteLine(message.Substring(ichMin, ichLim - ichMin));

                ichMin = ichLim + 1;
                if (ichMin < message.Length && message[ichLim] == '\r' && message[ichMin] == '\n')
                    ichMin++;
            }

            Contracts.Assert(ichMin <= ichLim);
            if (ichMin < ichLim)
            {
                if (prefix != null && ichMin > 0)
                    writer.Write(prefix);
                writer.WriteLine(message.Substring(ichMin, ichLim - ichMin));
            }
            else if (!removeLastNewLine)
                writer.WriteLine();
        }
    }
}
