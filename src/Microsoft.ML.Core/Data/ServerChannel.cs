// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Instances of this class are used to set up a bundle of named delegates. These
    /// delegates are registered through <see cref="Register{TRet}"/> and its overloads.
    /// Once all registrations are done, <see cref="Publish"/> is called and a message
    /// of type <see cref="Bundle"/> is sent through the input channel
    /// provider. The intended use case is that any information surfaced through these
    /// delegates will be published in some fashion, with the target scenario being
    /// that the library will publish some sort of restful API.
    /// </summary>
    public sealed class ServerChannel : ServerChannel.IPendingBundleNotification, IDisposable
    {
        // See ServerChannel.md for a more elaborate discussion of high level usage and design.
        private readonly IChannelProvider _chp;
        private readonly string _identifier;

        // This holds the running collection of named delegates, if any. The dictionary itself
        // is lazily initialized only when a listener
        private Dictionary<string, Delegate> _toPublish;
        private Action<Bundle> _onPublish;
        private Bundle _published;
        private bool _disposed;

        /// <summary>
        /// Returns either this object, or <c>null</c> if there are no listeners on this server
        /// channel. This can be used in conjunction with the <c>?.</c> operator to have more
        /// performant though more robust calls to <see cref="Register{TRet}"/> and
        /// <see cref="Publish"/>.
        /// </summary>
        private ServerChannel ThisIfActiveOrNull => _toPublish == null ? null : this;

        private ServerChannel(IChannelProvider provider, string idenfier)
        {
            Contracts.AssertValue(provider);
            _chp = provider;
            _chp.AssertNonWhiteSpace(idenfier);
            _identifier = idenfier;
        }

        /// <summary>
        /// Starts a new server channel.
        /// </summary>
        /// <param name="provider">The channel provider, on which to send
        /// the notification that a server is being constructed</param>
        /// <param name="identifier">A semi-unique identifier for this
        /// "bundle" that is being constructed</param>
        /// <returns>The constructed server channel, or <c>null</c> if there
        /// was no listeners for server channels registered on <paramref name="provider"/></returns>
        public static ServerChannel Start(IChannelProvider provider, string identifier)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckNonWhiteSpace(identifier, nameof(identifier));
            using (var pipe = provider.StartPipe<IPendingBundleNotification>("Server"))
            {
                var sc = new ServerChannel(provider, identifier);
                pipe.Send(sc);
                return sc.ThisIfActiveOrNull;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _published?.Done();
            }
        }

        private void RegisterCore(string name, Delegate func)
        {
            _chp.CheckNonEmpty(name, nameof(name));
            _chp.CheckValue(func, nameof(func));
            _chp.Check(_published == null, "Cannot expose more interfaces once a server channel has been published");
            _chp.AssertValue(_toPublish);

            _toPublish.Add(name, func);
        }

        public void Register<TRet>(string name, Func<TRet> func)
        {
            if (_toPublish != null)
                RegisterCore(name, func);
        }

        public void Register<T1, TRet>(string name, Func<T1, TRet> func)
        {
            if (_toPublish != null)
                RegisterCore(name, func);
        }

        public void Register<T1, T2, TRet>(string name, Func<T1, T2, TRet> func)
        {
            if (_toPublish != null)
                RegisterCore(name, func);
        }

        public void Register<T1, T2, T3, TRet>(string name, Func<T1, T2, T3, TRet> func)
        {
            if (_toPublish != null)
                RegisterCore(name, func);
        }

        /// <summary>
        /// Finalizes all registrations of delegates, and pipes the bundle of objects
        /// in a <see cref="Bundle"/> up through the pipe to be consumed by any
        /// listeners.
        /// </summary>
        public void Publish()
        {
            _chp.Assert((_toPublish == null) == (_onPublish == null));
            if (_toPublish == null)
                return;
            _chp.Check(_published == null, "Cannot republish once a server channel has been published");
            _published = new Bundle(this);
            _onPublish(_published);
        }

        public void Acknowledge(Action<Bundle> toDo)
        {
            _chp.CheckValue(toDo, nameof(toDo));
            _chp.Assert((_onPublish == null) == (_toPublish == null));
            if (_toPublish == null)
                _toPublish = new Dictionary<string, Delegate>();
            _onPublish += toDo;
            _chp.AssertValue(_onPublish);
        }

        /// <summary>
        /// Entry point factory for creating <see cref="IServer"/> instances.
        /// </summary>
        [TlcModule.ComponentKind("Server")]
        public interface IServerFactory : IArgsComponent, IComponentFactory<IChannel, IServer>
        {
            new IServer CreateComponent(IHostEnvironment env, IChannel ch);
        }

        /// <summary>
        /// Classes that want to publish the bundles from server channels in some fashion should implement
        /// this interface. The intended simple use case is that this will be some form of in-process web
        /// server, and then when disposed, they should stop themselves.
        ///
        /// Note that the primary communication with the server from the client code's perspective is not
        /// through method calls on this interface, but rather communication through an
        /// <see cref="IPipe{IPendingBundleNotification}"/> that the server will listen to throughout its
        /// lifetime.
        /// </summary>
        public interface IServer : IDisposable
        {
            /// <summary>
            /// This should return the base address where the server is. If this server is not actually
            /// serving content at any URL, this property should be null.
            /// </summary>
            Uri BaseAddress { get; }
        }

        /// <summary>
        /// Creates what might be considered a good "default" server factory, if possible,
        /// or <c>null</c> if no good default was possible. A <c>null</c> value could be returned,
        /// for example, if a user opted to remove all implementations of <see cref="IServer"/> and
        /// the associated <see cref="IServerFactory"/> for security reasons.
        /// </summary>
        public static IServerFactory CreateDefaultServerFactoryOrNull(IExceptionContext ectx)
        {
            Contracts.CheckValue(ectx, nameof(ectx));
            // REVIEW: There should be a better way. There currently isn't,
            // but there should be. This is pretty horrifying, but it is preferable to
            // the alternative of having core components depend on an actual server
            // implementation, since we want those to be removable because of security
            // concerns in certain environments (since not everyone will be wild about
            // web servers popping up everywhere).
            var cat = ModuleCatalog.CreateInstance(ectx);
            ModuleCatalog.ComponentInfo component;
            if (!cat.TryFindComponent(typeof(IServerFactory), "mini", out component))
                return null;
            IServerFactory factory = (IServerFactory)Activator.CreateInstance(component.ArgumentType);
            var field = factory.GetType().GetField("Port");
            if (field?.FieldType != typeof(int))
                return null;
            field.SetValue(factory, 12345);
            return factory;
        }

        /// <summary>
        /// When a <see cref="ServerChannel"/> is created, the creation method will send an implementation
        /// is a notification sent through an <see cref="IPipe{IPendingBundleNotification}"/>, to indicate that
        /// a <see cref="Bundle"/> may be pending soon. Listeners that want to receive the bundle to
        /// expose it, e.g., a web service, should register this interest by passing in an action to be called.
        /// If no listener registers interest, the server channel that sent the notification will act
        /// differently by, say, acting as a no-op w.r.t. client calls to it.
        /// </summary>
        public interface IPendingBundleNotification
        {
            /// <summary>
            /// Any publisher of the named delegates will call this method, upon receiving an instance
            /// of this object through the pipe. This method serves two purposes: firstly it detects
            /// whether anyone is even interested in publishing anything at all, so that we can just
            /// ignore any input delegates in the case where no one is listening (which, we must expect,
            /// is the majority of scenarios). The second is that it provides an action to call, once
            /// all publishing is complete, and <see cref="Publish"/> has been called by the client code.
            /// </summary>
            /// <param name="toDo">The callback to perform when all named delegates have been registered,
            /// and <see cref="Publish"/> is called.</param>
            void Acknowledge(Action<Bundle> toDo);
        }

        /// <summary>
        /// The final bundle of published named delegates that a listener can serve.
        /// </summary>
        public sealed class Bundle
        {
            /// <summary>
            /// This contains a name to delegate mappings. The delegates contained herein are gauranteed to be
            /// some variety of <see cref="Func{TResult}"/>, <see cref="Func{T1, TResult}"/>,
            /// <see cref="Func{T1, T2, TResult}"/>, etc.
            /// </summary>
            public readonly IReadOnlyDictionary<string, Delegate> NameToFuncs;

            /// <summary>
            /// This should be a more-or-less unique identifier for the type of API this bundle is producing.
            /// Its intended use is that it will form part of the URL for the RESTful API, so to the extent that
            /// it contains multiple tokens they must be slash delimited.
            /// </summary>
            public readonly string Identifier;

            internal Action Done;

            internal Bundle(ServerChannel sch)
            {
                Contracts.AssertValue(sch);

                NameToFuncs = sch._toPublish;
                Identifier = sch._identifier;
            }

            public void AddDoneAction(Action onDone)
            {
                Done += onDone;
            }
        }
    }

    public static class ServerChannelUtilities
    {
        /// <summary>
        /// Convenience method for <see cref="ServerChannel.Start"/> that looks more idiomatic to typical
        /// channel creation methods on <see cref="IChannelProvider"/>.
        /// </summary>
        /// <param name="provider">The channel provider.</param>
        /// <param name="identifier">This is an identifier of the "type" of bundle that is being published,
        /// and should form a path with forward-slash '/' delimiters.</param>
        /// <returns>The newly created server channel, or <c>null</c> if there was no listener for
        /// server channels on <paramref name="provider"/>.</returns>
        public static ServerChannel StartServerChannel(this IChannelProvider provider, string identifier)
        {
            Contracts.CheckValue(provider, nameof(provider));
            Contracts.CheckNonWhiteSpace(identifier, nameof(identifier));
            return ServerChannel.Start(provider, identifier);
        }
    }
}
