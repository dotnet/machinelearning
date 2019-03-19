// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

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
    /// Utility class for IHostEnvironment
    /// </summary>
    [BestFriend]
    internal static class HostEnvironmentExtensions
    {
        /// <summary>
        /// Return a file handle for an input "file".
        /// </summary>
        public static IFileHandle OpenInputFile(this IHostEnvironment env, string path)
        {
            Contracts.AssertValue(env);
            Contracts.CheckNonWhiteSpace(path, nameof(path));
            return new SimpleFileHandle(env, path, needsWrite: false, autoDelete: false);
        }

        /// <summary>
        /// Create an output "file" and return a handle to it.
        /// </summary>
        public static IFileHandle CreateOutputFile(this IHostEnvironment env, string path)
        {
            Contracts.AssertValue(env);
            Contracts.CheckNonWhiteSpace(path, nameof(path));
            return new SimpleFileHandle(env, path, needsWrite: true, autoDelete: false);
        }
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
        IHost Register(string name, int? seed = null, bool? verbose = null);

        /// <summary>
        /// The catalog of loadable components (<see cref="LoadableClassAttribute"/>) that are available in this host.
        /// </summary>
        ComponentCatalog ComponentCatalog { get; }
    }

    [BestFriend]
    internal interface ICancelable
    {
        /// <summary>
        /// Signal to stop exection in all the hosts.
        /// </summary>
        void CancelExecution();

        /// <summary>
        /// Flag which indicates host execution has been stopped.
        /// </summary>
        bool IsCanceled { get; }
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
        Random Rand { get; }
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
        /// Note that, despite being part of the schema, annotations should be treated
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
    public readonly struct ChannelMessage
    {
        public readonly ChannelMessageKind Kind;
        public readonly MessageSensitivity Sensitivity;
        private readonly string _message;
        private readonly object[] _args;

        /// <summary>
        /// Line endings may not be normalized.
        /// </summary>
        public string Message => _args != null ? string.Format(_message, _args) : _message;

        [BestFriend]
        internal ChannelMessage(ChannelMessageKind kind, MessageSensitivity sensitivity, string message)
        {
            Contracts.CheckNonEmpty(message, nameof(message));
            Kind = kind;
            Sensitivity = sensitivity;
            _message = message;
            _args = null;
        }

        [BestFriend]
        internal ChannelMessage(ChannelMessageKind kind, MessageSensitivity sensitivity, string fmt, params object[] args)
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
    [BestFriend]
    internal static class HostExtensions
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
            => ch.Trace(MessageSensitivity.Unknown, fmt, args);

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
