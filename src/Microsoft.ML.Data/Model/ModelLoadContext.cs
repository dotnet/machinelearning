// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Model
{
    /// <summary>
    /// This is a convenience context object for loading models from a repository, for
    /// implementors of ICanSaveModel. It is not mandated but designed to reduce the
    /// amount of boiler plate code. It can also be used when loading from a single stream,
    /// for implementors of ICanSaveInBinaryFormat.
    /// </summary>
    public sealed partial class ModelLoadContext : IDisposable
    {
        /// <summary>
        /// When in repository mode, this is the repository we're reading from. It is null when
        /// in single-stream mode.
        /// </summary>
        public readonly RepositoryReader Repository;

        /// <summary>
        /// When in repository mode, this is the direcory we're reading from. Null means the root
        /// of the repository. It is always null in single-stream mode.
        /// </summary>
        public readonly string Directory;

        /// <summary>
        /// The main stream reader.
        /// </summary>
        public readonly BinaryReader Reader;

        /// <summary>
        /// The strings loaded from the main stream's string table.
        /// </summary>
        public readonly string[] Strings;

        /// <summary>
        /// The name of the assembly that the loader lives in.
        /// </summary>
        /// <remarks>
        /// This may be null or empty if one was never written to the model, or is an older model version.
        /// </remarks>
        public readonly string LoaderAssemblyName;

        /// <summary>
        /// The main stream's model header.
        /// </summary>
        public ModelHeader Header;

        /// <summary>
        /// The min file position of the main stream.
        /// </summary>
        public readonly long FpMin;

        /// <summary>
        /// Exception context provided by Repository (can be null).
        /// </summary>
        private readonly IExceptionContext _ectx;

        /// <summary>
        /// Returns whether this context is in repository mode (true) or single-stream mode (false).
        /// </summary>
        public bool InRepository { get { return Repository != null; } }

        /// <summary>
        /// Create a ModelLoadContext supporting loading from a repository, for implementors of ICanSaveModel.
        /// </summary>
        public ModelLoadContext(RepositoryReader rep, Repository.Entry ent, string dir)
        {
            Contracts.CheckValue(rep, nameof(rep));
            Repository = rep;
            _ectx = rep.ExceptionContext;

            _ectx.CheckValue(ent, nameof(ent));
            _ectx.CheckValueOrNull(dir);

            Directory = dir;

            Reader = new BinaryReader(ent.Stream, Encoding.UTF8, leaveOpen: true);
            try
            {
                ModelHeader.BeginRead(out FpMin, out Header, out Strings, out LoaderAssemblyName, Reader);
            }
            catch
            {
                Reader.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Create a ModelLoadContext supporting loading from a single-stream, for implementors of ICanSaveInBinaryFormat.
        /// </summary>
        public ModelLoadContext(BinaryReader reader, IExceptionContext ectx = null)
        {
            Contracts.AssertValueOrNull(ectx);
            _ectx = ectx;
            _ectx.CheckValue(reader, nameof(reader));

            Repository = null;
            Directory = null;
            Reader = reader;
            ModelHeader.BeginRead(out FpMin, out Header, out Strings, out LoaderAssemblyName, Reader);
        }

        public void CheckAtModel()
        {
            _ectx.Check(Reader.BaseStream.Position == FpMin + Header.FpModel);
        }

        public void CheckAtModel(VersionInfo ver)
        {
            _ectx.Check(Reader.BaseStream.Position == FpMin + Header.FpModel);
            ModelHeader.CheckVersionInfo(ref Header, ver);
        }

        /// <summary>
        /// Performs version checks.
        /// </summary>
        public void CheckVersionInfo(VersionInfo ver)
        {
            ModelHeader.CheckVersionInfo(ref Header, ver);
        }

        /// <summary>
        /// Reads an integer from the load context's reader, and returns the associated string,
        /// or null (encoded as -1).
        /// </summary>
        public string LoadStringOrNull()
        {
            int id = Reader.ReadInt32();
            // Note that -1 means null. Empty strings are in the string table.
            _ectx.CheckDecode(-1 <= id && id < Utils.Size(Strings));
            if (id >= 0)
                return Strings[id];
            return null;
        }

        /// <summary>
        /// Reads an integer from the load context's reader, and returns the associated string.
        /// </summary>
        public string LoadString()
        {
            int id = Reader.ReadInt32();
            Contracts.CheckDecode(0 <= id && id < Utils.Size(Strings));
            return Strings[id];
        }

        /// <summary>
        /// Reads an integer from the load context's reader, and returns the associated string.
        /// Throws if the string is empty or null.
        /// </summary>
        public string LoadNonEmptyString()
        {
            int id = Reader.ReadInt32();
            _ectx.CheckDecode(0 <= id && id < Utils.Size(Strings));
            var str = Strings[id];
            _ectx.CheckDecode(str.Length > 0);
            return str;
        }

        /// <summary>
        /// Commit the load operation. This completes reading of the main stream. When in repository
        /// mode, it disposes the Reader (but not the repository).
        /// </summary>
        public void Done()
        {
            ModelHeader.EndRead(FpMin, ref Header, Reader);
            Dispose();
        }

        /// <summary>
        /// When in repository mode, this disposes the Reader (but no the repository).
        /// </summary>
        public void Dispose()
        {
            // When in single-stream mode, we don't own the Reader.
            if (InRepository)
                Reader.Dispose();
        }
    }
}
