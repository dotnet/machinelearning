// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Model
{
    public sealed partial class ModelSaveContext : IDisposable
    {
        /// <summary>
        /// Save a sub model to the given sub directory. This requires InRepository to be true.
        /// </summary>
        public void SaveModel<T>(T value, string name)
            where T : class
        {
            _ectx.Check(InRepository, "Can't save a sub-model when writing to a single stream");
            SaveModel(Repository, value, Path.Combine(Directory ?? "", name));
        }

        /// <summary>
        /// Save the object by calling TrySaveModel then falling back to .net serialization.
        /// </summary>
        public static void SaveModel<T>(RepositoryWriter rep, T value, string path)
            where T : class
        {
            if (value == null)
                return;

            var sm = value as ICanSaveModel;
            if (sm != null)
            {
                using (var ctx = new ModelSaveContext(rep, path, ModelLoadContext.ModelStreamName))
                {
                    sm.Save(ctx);
                    ctx.Done();
                }
                return;
            }

            var sb = value as ICanSaveInBinaryFormat;
            if (sb != null)
            {
                using (var ent = rep.CreateEntry(path, ModelLoadContext.NameBinary))
                using (var writer = new BinaryWriter(ent.Stream, Encoding.UTF8, leaveOpen: true))
                {
                    sb.SaveAsBinary(writer);
                }
                return;
            }
        }

        /// <summary>
        /// Save to a single-stream by invoking the given action.
        /// </summary>
        public static void Save(BinaryWriter writer, Action<ModelSaveContext> fn)
        {
            Contracts.CheckValue(writer, nameof(writer));
            Contracts.CheckValue(fn, nameof(fn));

            using (var ctx = new ModelSaveContext(writer))
            {
                fn(ctx);
                ctx.Done();
            }
        }

        /// <summary>
        /// Save to the given sub directory by invoking the given action. This requires InRepository to be true.
        /// </summary>
        public void SaveSubModel(string dir, Action<ModelSaveContext> fn)
        {
            _ectx.Check(InRepository, "Can't save a sub-model when writing to a single stream");
            _ectx.CheckNonEmpty(dir, nameof(dir));
            _ectx.CheckValue(fn, nameof(fn));

            using (var ctx = new ModelSaveContext(Repository, Path.Combine(Directory ?? "", dir), ModelLoadContext.ModelStreamName))
            {
                fn(ctx);
                ctx.Done();
            }
        }
    }
}