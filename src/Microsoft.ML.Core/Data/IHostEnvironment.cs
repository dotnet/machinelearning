// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// A channel provider can create new channels and generic information pipes.
    /// </summary>
    public interface IChannelProvider : IExceptionContext
    {
        /// <summary>
        /// Start a standard message channel.
        /// </summary>
        IChannel Start(string name);

        /// <summary>
        /// Start a generic information pipe.
        /// </summary>
        IPipe<TMessage> StartPipe<TMessage>(string name);
    }

    /// <summary>
    /// The host environment interface creates hosts for components. Note that the methods of
    /// this interface should be called from the main thread for the environment. To get an environment
    /// to service another thread, call Fork and pass the return result to that thread.
    /// </summary>
    public interface IHostEnvironment : IChannelProvider, IProgressChannelProvider
    {
        /// <summary>
        /// Create a host with the given registration name.
        /// </summary>
        IHost Register(string name, int? seed = null, bool? verbose = null, int? conc = null);

        /// <summary>
        /// How much concurrency the component should use. A value of 1 means
        /// single-threaded. Higher values generally mean number of threads. Less
        /// than 1 means whatever the component views as ideal.
        /// </summary>
        int ConcurrencyFactor { get; }

        /// <summary>
        /// Flag which indicate should we stop any code execution in this host.
        /// </summary>
        bool IsCancelled { get; }

        /// <summary>
        /// The catalog of loadable components (<see cref="LoadableClassAttribute"/>) that are available in this host.
        /// </summary>
        ComponentCatalog ComponentCatalog { get; }

        /// <summary>
        /// Return a file handle for an input "file".
        /// </summary>
        IFileHandle OpenInputFile(string path);

        /// <summary>
        /// Create an output "file" and return a handle to it.
        /// </summary>
        IFileHandle CreateOutputFile(string path);

        /// <summary>
        /// Create a temporary "file" and return a handle to it. Generally temp files are expected to be
        /// written to exactly once, and then can be read multiple times.
        /// Note that IFileHandle derives from IDisposable. Clients may dispose the IFileHandle when it is
        /// no longer needed, but they are not required to. The host environment should track all temp file
        /// handles and ensure that they are disposed properly when the environment is "shut down".
        ///
        /// The suffix and prefix are optional. A common use for suffix is to specify an extension, eg, ".txt".
        /// The use of suffix and prefix, including whether they have any affect, is up to the host enviroment.
        /// </summary>
        IFileHandle CreateTempFile(string suffix = null, string prefix = null);
    }

    /// <summary>
    /// A host is coupled to a component and provides random number generation and concurrency guidance.
    /// Note that the random number generation, like the host environment methods, should be accessed only
    /// from the main thread for the component.
    /// </summary>
    public interface IHost : IHostEnvironment
    {
        /// <summary>
        /// The random number generator issued to this component. Note that random number
        /// generators are NOT thread safe.
        /// </summary>
        IRandom Rand { get; }

        /// <summary>
        /// Signal to stop exection in this host and all its children.
        /// </summary>
        void StopExecution();
    }

    /// <summary>
    /// A generic information pipe. Note that pipes are disposable. Generally, Done should
    /// be called before disposing to signal a normal shut-down of the pipe, as opposed
    /// to an aborted completion.
    /// </summary>
    public interface IPipe<TMessage> : IExceptionContext, IDisposable
    {
        /// <summary>
        /// The caller relinquishes ownership of the <paramref name="msg"/> object.
        /// </summary>
        void Send(TMessage msg);
    }

    /// <summary>
    /// The kinds of standard channel messages.
    /// Note: These values should never be changed. We can add new kinds, but don't change these values.
    /// Other code bases, including native code for other projects depends on these values.
    /// </summary>
    public enum ChannelMessageKind
    {
        Trace = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }

    /// <summary>
    /// A flag that can be attached to a message or exception to indicate that
    /// it has a certain class of sensitive data. By default, messages should be
    /// specified as being of unknown sensitivity, which is to say, every
    /// sensitivity flag is turned on, corresponding to <see cref="Unknown"/>.
    /// Messages that are totally safe should be marked as <see cref="None"/>.
    /// However, if, say, one prints out data from a file (for example, this might
    /// be done when expressing parse errors), it should be flagged in that case
    /// with <see cref="UserData"/>.
    /// </summary>
    [Flags]
    public enum MessageSensitivity
    {
        /// <summary>
        /// For non-sensitive data.
        /// </summary>
        None = 0,

        /// <summary>
        /// For messages that may contain user-data from data files.
        /// </summary>
        UserData = 0x1,

        /// <summary>
        /// For messages that contain information like column names from datasets.
        /// Note that, despite being part of the schema, metadata should be treated
        /// as user data, since it is often derived from user data. Note also that
        /// types, despite being part of the schema, are not considered "sensitive"
        /// as such, in the same way that column names might be.
        /// </summary>
        Schema = 0x2,

        // REVIEW: Other potentially sensitive things might include
        // stack traces in certain environments.

        /// <summary>
        /// The default value, unknown, is treated as if everything is sensitive.
        /// </summary>
        Unknown = ~None,

        /// <summary>
        /// An alias for <see cref="Unknown"/>, so it is functionally the same, except
        /// semantically it communicates the idea that we want all bits set.
        /// </summary>
        All = Unknown,
    }

    /// <summary>
    /// A channel message.
    /// </summary>
    public struct ChannelMessage
    {
        public readonly ChannelMessageKind Kind;
        public readonly MessageSensitivity Sensitivity;
        private readonly string _message;
        private readonly object[] _args;

        /// <summary>
        /// Line endings may not be normalized.
        /// </summary>
        public string Message => _args != null ? string.Format(_message, _args) : _message;

        public ChannelMessage(ChannelMessageKind kind, MessageSensitivity sensitivity, string message)
        {
            Contracts.CheckNonEmpty(message, nameof(message));
            Kind = kind;
            Sensitivity = sensitivity;
            _message = message;
            _args = null;
        }

        public ChannelMessage(ChannelMessageKind kind, MessageSensitivity sensitivity, string fmt, params object[] args)
        {
            Contracts.CheckNonEmpty(fmt, nameof(fmt));
            Contracts.CheckNonEmpty(args, nameof(args));
            Kind = kind;
            Sensitivity = sensitivity;
            _message = fmt;
            _args = args;
        }
    }

    /// <summary>
    /// A standard communication channel.
    /// </summary>
    public interface IChannel : IPipe<ChannelMessage>
    {
        void Trace(MessageSensitivity sensitivity, string fmt);
        void Trace(MessageSensitivity sensitivity, string fmt, params object[] args);
        void Error(MessageSensitivity sensitivity, string fmt);
        void Error(MessageSensitivity sensitivity, string fmt, params object[] args);
        void Warning(MessageSensitivity sensitivity, string fmt);
        void Warning(MessageSensitivity sensitivity, string fmt, params object[] args);
        void Info(MessageSensitivity sensitivity, string fmt);
        void Info(MessageSensitivity sensitivity, string fmt, params object[] args);
    }

    /// <summary>
    /// General utility extension methods for objects in the "host" universe, i.e.,
    /// <see cref="IHostEnvironment"/>, <see cref="IHost"/>, and <see cref="IChannel"/>
    /// that do not belong in more specific areas, for example, <see cref="Contracts"/> or
    /// component creation.
    /// </summary>
    public static class HostExtensions
    {
        public static T Apply<T>(this IHost host, string channelName, Func<IChannel, T> func)
        {
            T t;
            using (var ch = host.Start(channelName))
            {
                t = func(ch);
            }
            return t;
        }

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Trace(MessageSensitivity, string)"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Trace(this IChannel ch, string fmt)
            => ch.Trace(MessageSensitivity.Unknown, fmt);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Trace(MessageSensitivity, string, object[])"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Trace(this IChannel ch, string fmt, params object[] args)
            => ch.Trace(MessageSensitivity.Unknown, fmt);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Error(MessageSensitivity, string)"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Error(this IChannel ch, string fmt)
            => ch.Error(MessageSensitivity.Unknown, fmt);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Error(MessageSensitivity, string, object[])"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Error(this IChannel ch, string fmt, params object[] args)
            => ch.Error(MessageSensitivity.Unknown, fmt, args);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Warning(MessageSensitivity, string)"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Warning(this IChannel ch, string fmt)
            => ch.Warning(MessageSensitivity.Unknown, fmt);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Warning(MessageSensitivity, string, object[])"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Warning(this IChannel ch, string fmt, params object[] args)
            => ch.Warning(MessageSensitivity.Unknown, fmt, args);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Info(MessageSensitivity, string)"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Info(this IChannel ch, string fmt)
            => ch.Info(MessageSensitivity.Unknown, fmt);

        /// <summary>
        /// Convenience variant of <see cref="IChannel.Info(MessageSensitivity, string, object[])"/>
        /// setting <see cref="MessageSensitivity.Unknown"/>.
        /// </summary>
        public static void Info(this IChannel ch, string fmt, params object[] args)
            => ch.Info(MessageSensitivity.Unknown, fmt, args);
    }
}
