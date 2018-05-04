// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Model
{
    public sealed partial class ModelLoadContext : IDisposable
    {
        public const string ModelStreamName = "Model.key";
        internal const string NameBinary = "Model.bin";

        /// <summary>
        /// Return whether this context contains a directory and stream for a sub-model with
        /// the indicated name. This does not attempt to load the sub-model.
        /// </summary>
        public bool ContainsModel(string name)
        {
            if (!InRepository)
                return false;
            if (string.IsNullOrEmpty(name))
                return false;

            var dir = Path.Combine(Directory ?? "", name);
            var ent = Repository.OpenEntryOrNull(dir, ModelStreamName);
            if (ent != null)
            {
                ent.Dispose();
                return true;
            }

            if ((ent = Repository.OpenEntryOrNull(dir, NameBinary)) != null)
            {
                ent.Dispose();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Load an optional object from the repository directory.
        /// Returns false iff no stream was found for the object, iff result is set to null.
        /// Throws if loading fails for any other reason.
        /// </summary>
        public static bool LoadModelOrNull<TRes, TSig>(IHostEnvironment env, out TRes result, RepositoryReader rep, string dir, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(rep, nameof(rep));
            var ent = rep.OpenEntryOrNull(dir, ModelStreamName);
            if (ent != null)
            {
                using (ent)
                {
                    // Provide the repository, entry, and directory name to the loadable class ctor.
                    env.Assert(ent.Stream.Position == 0);
                    LoadModel<TRes, TSig>(env, out result, rep, ent, dir, extra);
                    return true;
                }
            }

            if ((ent = rep.OpenEntryOrNull(dir, NameBinary)) != null)
            {
                using (ent)
                {
                    env.Assert(ent.Stream.Position == 0);
                    LoadModel<TRes, TSig>(env, out result, ent.Stream, extra);
                    return true;
                }
            }

            result = null;
            return false;
        }

        /// <summary>
        /// Load an object from the repository directory.
        /// </summary>
        public static void LoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, RepositoryReader rep, string dir, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(rep, nameof(rep));
            if (!LoadModelOrNull<TRes, TSig>(env, out result, rep, dir, extra))
                throw env.ExceptDecode("Corrupt model file");
            env.AssertValue(result);
        }

        /// <summary>
        /// Load a sub model from the given sub directory if it exists. This requires InRepository to be true.
        /// Returns false iff no stream was found for the object, iff result is set to null.
        /// Throws if loading fails for any other reason.
        /// </summary>
        public bool LoadModelOrNull<TRes, TSig>(IHostEnvironment env, out TRes result, string name, params object[] extra)
            where TRes : class
        {
            _ectx.CheckValue(env, nameof(env));
            _ectx.Check(InRepository, "Can't load a sub-model when reading from a single stream");
            return LoadModelOrNull<TRes, TSig>(env, out result, Repository, Path.Combine(Directory ?? "", name), extra);
        }

        /// <summary>
        /// Load a sub model from the given sub directory. This requires InRepository to be true.
        /// </summary>
        public void LoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, string name, params object[] extra)
            where TRes : class
        {
            _ectx.CheckValue(env, nameof(env));
            if (!LoadModelOrNull<TRes, TSig>(env, out result, name, extra))
                throw _ectx.ExceptDecode("Corrupt model file");
            _ectx.AssertValue(result);
        }

        /// <summary>
        /// Try to load from the given repository entry using the default loader(s) specified in the header.
        /// Returns false iff the default loader(s) could not be bound to a compatible loadable class.
        /// </summary>
        private static bool TryLoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, RepositoryReader rep, Repository.Entry ent, string dir, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(rep, nameof(rep));
            long fp = ent.Stream.Position;
            using (var ctx = new ModelLoadContext(rep, ent, dir))
            {
                env.Assert(fp == ctx.FpMin);
                if (ctx.TryLoadModelCore<TRes, TSig>(env, out result, extra))
                    return true;
            }

            // TryLoadModelCore should rewind on failure.
            Contracts.Assert(fp == ent.Stream.Position);
            return false;
        }

        /// <summary>
        /// Load from the given repository entry using the default loader(s) specified in the header.
        /// </summary>
        public static void LoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, RepositoryReader rep, Repository.Entry ent, string dir, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(rep, nameof(rep));
            if (!TryLoadModel<TRes, TSig>(env, out result, rep, ent, dir, extra))
                throw env.ExceptDecode("Couldn't load model: '{0}'", dir);
        }

        /// <summary>
        /// Try to load from the given stream (non-Repository).
        /// Returns false iff the default loader(s) could not be bound to a compatible loadable class.
        /// </summary>
        public static bool TryLoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, Stream stream, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true))
                return TryLoadModel<TRes, TSig>(env, out result, reader, extra);
        }

        /// <summary>
        /// Load from the given stream (non-Repository) using the default loader(s) specified in the header.
        /// </summary>
        public static void LoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, Stream stream, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            if (!TryLoadModel<TRes, TSig>(env, out result, stream, extra))
                throw Contracts.ExceptDecode("Couldn't load model");
        }

        /// <summary>
        /// Try to load from the given reader (non-Repository).
        /// Returns false iff the default loader(s) could not be bound to a compatible loadable class.
        /// </summary>
        public static bool TryLoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, BinaryReader reader, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            long fp = reader.BaseStream.Position;
            using (var ctx = new ModelLoadContext(reader))
            {
                Contracts.Assert(fp == ctx.FpMin);
                return ctx.TryLoadModelCore<TRes, TSig>(env, out result, extra);
            }
        }

        /// <summary>
        /// Load from the given reader (non-Repository) using the default loader(s) specified in the header.
        /// </summary>
        public static void LoadModel<TRes, TSig>(IHostEnvironment env, out TRes result, BinaryReader reader, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            if (!TryLoadModel<TRes, TSig>(env, out result, reader, extra))
                throw Contracts.ExceptDecode("Couldn't load model");
        }

        /// <summary>
        /// Tries to load.
        /// Returns false iff the default loader(s) could not be bound to a compatible loadable class.
        /// </summary>
        private bool TryLoadModelCore<TRes, TSig>(IHostEnvironment env, out TRes result, params object[] extra)
            where TRes : class
        {
            _ectx.AssertValue(env, "env");
            _ectx.Assert(Reader.BaseStream.Position == FpMin + Header.FpModel);

            var args = ConcatArgsRev(extra, this);

            object tmp;
            string sig = ModelHeader.GetLoaderSig(ref Header);
            if (!string.IsNullOrWhiteSpace(sig) &&
                ComponentCatalog.TryCreateInstance<object, TSig>(env, out tmp, sig, "", args))
            {
                result = tmp as TRes;
                if (result != null)
                {
                    Done();
                    return true;
                }
                // REVIEW: Should this fall through?
            }
            _ectx.Assert(Reader.BaseStream.Position == FpMin + Header.FpModel);

            string sigAlt = ModelHeader.GetLoaderSigAlt(ref Header);
            if (!string.IsNullOrWhiteSpace(sigAlt) &&
                ComponentCatalog.TryCreateInstance<object, TSig>(env, out tmp, sigAlt, "", args))
            {
                result = tmp as TRes;
                if (result != null)
                {
                    Done();
                    return true;
                }
                // REVIEW: Should this fall through?
            }
            _ectx.Assert(Reader.BaseStream.Position == FpMin + Header.FpModel);

            Reader.BaseStream.Position = FpMin;
            result = null;
            return false;
        }

        private static object[] ConcatArgsRev(object[] args2, params object[] args1)
        {
            Contracts.AssertNonEmpty(args1);
            return Utils.Concat(args1, args2);
        }

        /// <summary>
        /// Try to load a sub model from the given sub directory. This requires InRepository to be true.
        /// </summary>
        public bool TryProcessSubModel(string dir, Action<ModelLoadContext> action)
        {
            _ectx.Check(InRepository, "Can't Load a sub-model when reading from a single stream");
            _ectx.CheckNonEmpty(dir, nameof(dir));
            _ectx.CheckValue(action, nameof(action));

            string path = Path.Combine(Directory, dir);
            var ent = Repository.OpenEntryOrNull(path, ModelStreamName);
            if (ent == null)
                return false;

            using (ent)
            {
                // Provide the repository, entry, and directory name to the loadable class ctor.
                _ectx.Assert(ent.Stream.Position == 0);
                using (var ctx = new ModelLoadContext(Repository, ent, path))
                    action(ctx);
            }
            return true;
        }

        /// <summary>
        /// Try to load a binary stream from the current directory. This requires InRepository to be true.
        /// </summary>
        public bool TryLoadBinaryStream(string name, Action<BinaryReader> action)
        {
            _ectx.Check(InRepository, "Can't Load a sub-model when reading from a single stream");
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValue(action, nameof(action));

            var ent = Repository.OpenEntryOrNull(Directory, name);
            if (ent == null)
                return false;

            using (ent)
            using (var reader = new BinaryReader(ent.Stream, Encoding.UTF8, leaveOpen: true))
            {
                action(reader);
            }
            return true;
        }

        /// <summary>
        /// Try to load a text stream from the current directory. This requires InRepository to be true.
        /// </summary>
        public bool TryLoadTextStream(string name, Action<TextReader> action)
        {
            _ectx.Check(InRepository, "Can't Load a sub-model when reading from a single stream");
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValue(action, nameof(action));

            var ent = Repository.OpenEntryOrNull(Directory, name);
            if (ent == null)
                return false;

            using (ent)
            using (var reader = new StreamReader(ent.Stream))
            {
                action(reader);
            }
            return true;
        }
    }
}