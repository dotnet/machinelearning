// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Model
{
    /// <summary>
    /// Signature for a repository based model loader. This is the dual of ICanSaveModel.
    /// </summary>
    public delegate void SignatureLoadModel(ModelLoadContext ctx);

    /// <summary>
    /// For saving a model into a repository.
    /// </summary>
    public interface ICanSaveModel
    {
        void Save(ModelSaveContext ctx);
    }

    /// <summary>
    /// For saving to a single stream.
    /// </summary>
    public interface ICanSaveInBinaryFormat
    {
        void SaveAsBinary(BinaryWriter writer);
    }

    /// <summary>
    /// Abstraction around a ZipArchive or other hierarchical storage.
    /// </summary>
    public abstract class Repository : IDisposable
    {
        public sealed class Entry : IDisposable
        {
            // The parent repository.
            private Repository _rep;

            /// <summary>
            /// The relative path of this entry.
            /// /// </summary>
            public string Path { get; }

            /// <summary>
            /// The stream for this entry. This is either a memory stream or a file stream in
            /// the temporary directory. In either case, it is seekable and NOT the actual
            /// archive stream.
            /// </summary>
            public Stream Stream { get; }

            internal Entry(Repository rep, string path, Stream stream)
            {
                _rep = rep;
                Path = path;
                Stream = stream;
            }

            public void Dispose()
            {
                if (_rep != null)
                {
                    // Tell the repository that we're disposed. Note that the repository "owns" the stream
                    // so is in charge of closing it.
                    _rep.OnDispose(this);
                    _rep = null;
                }
            }
        }

        // These are the open entries that may contain streams into our _dirTemp.
        private List<Entry> _open;

        private bool _disposed;

        private readonly IExceptionContext _ectx;

        // This is a temporary directory that we create. It is essentially treated like an un-managed resource,
        // hence the need for the complete dispose pattern. Note that it is optional - if we use memory
        // streams for everything, we don't need it. This ability is needed for Scope or other environments
        // where access to the file system is restricted.
        protected readonly string DirTemp;

        // Maps from relative path to the corresponding absolute path in the temp directory.
        // This is populated as we decompress streams in the archive, so we don't de-compress
        // more than once.
        // REVIEW: Should we garbage collect to some degree? Currently we don't delete any
        // of these temp files until the repository is disposed.
        protected readonly Dictionary<string, string> PathMap;

        /// <summary>
        /// Exception context.
        /// </summary>
        public IExceptionContext ExceptionContext => _ectx;

        protected bool Disposed => _disposed;

        internal Repository(bool needDir, IExceptionContext ectx)
        {
            Contracts.AssertValueOrNull(ectx);
            _ectx = ectx;

            PathMap = new Dictionary<string, string>();
            _open = new List<Entry>();
            if (needDir)
            {
                DirTemp = GetTempPath();
                Directory.CreateDirectory(DirTemp);
            }
            else
                GC.SuppressFinalize(this);
        }

        // REVIEW: This should use host environment functionality.
        private static string GetTempPath()
        {
            Guid guid = Guid.NewGuid();
            return Path.GetFullPath(Path.Combine(Path.GetTempPath(), "TLC_" + guid.ToString()));
        }

        ~Repository()
        {
            if (!Disposed)
                Dispose(false);
        }

        public void Dispose()
        {
            if (!Disposed)
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            _ectx.Assert(!Disposed);

            // Close all temp files.
            try
            {
                DisposeAllEntries();
            }
            catch
            {
                _ectx.Assert(false, "Closing entries should not throw!");
            }

            // Delete the temp directory.
            if (DirTemp != null)
            {
                try
                {
                    Directory.Delete(DirTemp, true);
                }
                catch
                {
                }
            }

            _disposed = true;
        }

        /// <summary>
        /// Force all open entries to be disposed.
        /// </summary>
        protected void DisposeAllEntries()
        {
            while (_open.Count > 0)
            {
                var ent = _open[_open.Count - 1];
                ent.Dispose();
            }
        }

        /// <summary>
        /// Remove the entry from _open. Note that under normal access patterns, entries are LIFO,
        /// so we search from the end of _open.
        /// </summary>
        protected void RemoveEntry(Entry ent)
        {
            // Note that under normal access patterns, entries are LIFO, so we search from the end of _open.
            for (int i = _open.Count; --i >= 0;)
            {
                if (_open[i] == ent)
                {
                    _open.RemoveAt(i);
                    return;
                }
            }
            _ectx.Assert(false, "Why wasn't the entry found?");
        }

        /// <summary>
        /// The entry is being disposed. Note that overrides should always call RemoveEntry, in addition to whatever
        /// they need to do with the corresponding stream.
        /// </summary>
        protected abstract void OnDispose(Entry ent);

        /// <summary>
        /// When considering entries inside one of our model archives, we want to ensure that we
        /// use a consistent directory separator. Zip archives are stored as flat lists of entries.
        /// When we load those entries into our look-up dictionary, we normalize them to always use
        /// backward slashes.
        /// </summary>
        protected static string NormalizeForArchiveEntry(string path) => path?.Replace('/', '\\');

        /// <summary>
        /// When building paths to our local file system, we want to force both forward and backward slashes
        /// to the system directory separator character. We do this for cases where we either used Windows-specific
        /// path building logic, or concatenated filesystem paths with zip archive entries on Linux. 
        /// </summary>
        private static string NormalizeForFileSystem(string path) =>
            path?.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);

        /// <summary>
        /// Constructs both the relative path to the entry and the absolute path of a corresponding
        /// temporary file. If createDir is true, makes sure the directory exists within the temp directory.
        /// </summary>
        protected void GetPath(out string pathEnt, out string pathTemp, string dir, string name, bool createDir)
        {
            _ectx.Assert(!Disposed);
            _ectx.CheckValueOrNull(dir);
            _ectx.CheckParam(dir == null || !dir.Contains(".."), nameof(dir));
            _ectx.CheckParam(!string.IsNullOrWhiteSpace(name), nameof(name));
            _ectx.CheckParam(!name.Contains(".."), nameof(name));

            // The gymnastics below are meant to deal with bad invocations including absolute paths, etc.
            // That's why we go through it even if _dirTemp is null.
            string root = Path.GetFullPath(DirTemp ?? @"x:\dummy");
            string entityPath = Path.Combine(root, dir ?? "", name);
            entityPath = Path.GetFullPath(entityPath);
            string tempPath = Path.Combine(root, PathMap.Count.ToString());
            tempPath = Path.GetFullPath(tempPath);

            string parent = Path.GetDirectoryName(entityPath);
            _ectx.Check(parent != null);
            _ectx.Check(parent.StartsWith(root));

            int ichSplit = root.Length;
            _ectx.Check(entityPath.Length > ichSplit && entityPath[ichSplit] == Path.DirectorySeparatorChar);

            if (createDir && DirTemp != null && parent.Length > ichSplit)
                Directory.CreateDirectory(parent);

            // Get the relative path portion. This is the archive entry name.
            pathEnt = entityPath.Substring(ichSplit + 1);
            _ectx.Check(Utils.Size(pathEnt) > 0);
            _ectx.Check(entityPath == Path.Combine(root, pathEnt));

            // Set pathTemp to non-null iff _dirTemp is non-null.
            pathTemp = DirTemp != null ? tempPath : null;

            pathEnt = NormalizeForArchiveEntry(pathEnt);
            pathTemp = NormalizeForFileSystem(pathTemp);
        }

        protected Entry AddEntry(string pathEnt, Stream stream)
        {
            _ectx.Assert(!Disposed);
            _ectx.AssertValue(stream);

            var ent = new Entry(this, pathEnt, stream);
            _open.Add(ent);
            return ent;
        }
    }

    public sealed class RepositoryWriter : Repository
    {
        private ZipArchive _archive;
        private Queue<KeyValuePair<string, Stream>> _closed;

        public static RepositoryWriter CreateNew(Stream stream, IExceptionContext ectx = null, bool useFileSystem = true)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(stream, nameof(stream));
            var rep = new RepositoryWriter(stream, ectx, useFileSystem);
            using (var ent = rep.CreateEntry(ModelFileUtils.DirTrainingInfo, "Version.txt"))
            using (var writer = Utils.OpenWriter(ent.Stream))
                writer.WriteLine(typeof(RepositoryWriter).Assembly.GetName().Version);
            return rep;
        }

        private RepositoryWriter(Stream stream, IExceptionContext ectx, bool useFileSystem = true)
            : base(useFileSystem, ectx)
        {
            _archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true);
            _closed = new Queue<KeyValuePair<string, Stream>>();
        }

        public Entry CreateEntry(string name)
        {
            return CreateEntry(null, name);
        }

        public Entry CreateEntry(string dir, string name)
        {
            ExceptionContext.Check(!Disposed);

            Flush();

            string pathEnt;
            string pathTemp;
            GetPath(out pathEnt, out pathTemp, dir, name, true);
            if (PathMap.ContainsKey(pathEnt))
                throw ExceptionContext.ExceptParam(nameof(name), "Duplicate entry: '{0}'", pathEnt);
            else
                PathMap.Add(pathEnt, pathTemp);

            Stream stream;
            if (pathTemp != null)
                stream = new FileStream(pathTemp, FileMode.CreateNew);
            else
                stream = new MemoryStream();

            return AddEntry(pathEnt, stream);
        }

        // The entry is being disposed. Note that this isn't supposed to throw, so we simply queue
        // the stream so it can be written to the archive when it IS legal to throw.
        protected override void OnDispose(Entry ent)
        {
            ExceptionContext.AssertValue(ent);
            RemoveEntry(ent);

            if (_closed != null)
                _closed.Enqueue(new KeyValuePair<string, Stream>(ent.Path, ent.Stream));
            else
                ent.Stream.CloseEx();
        }

        protected override void Dispose(bool disposing)
        {
            ExceptionContext.Assert(!Disposed);

            if (_closed != null)
            {
                while (_closed.Count > 0)
                {
                    var kvp = _closed.Dequeue();
                    kvp.Value.CloseEx();
                }
                _closed = null;
            }

            if (_archive != null)
            {
                try
                {
                    _archive.Dispose();
                }
                catch
                {
                }
                _archive = null;
            }

            // Close all the streams.
            base.Dispose(disposing);
        }

        // Write "closed" entries to the archive.
        private void Flush()
        {
            ExceptionContext.Assert(!Disposed);
            ExceptionContext.AssertValue(_closed);
            ExceptionContext.AssertValue(_archive);

            while (_closed.Count > 0)
            {
                string path = null;
                var kvp = _closed.Dequeue();
                using (var src = kvp.Value)
                {
                    var fs = src as FileStream;
                    if (fs != null)
                        path = fs.Name;

                    var ae = _archive.CreateEntry(kvp.Key);
                    using (var dst = ae.Open())
                    {
                        src.Position = 0;
                        src.CopyTo(dst);
                    }
                }

                if (!string.IsNullOrEmpty(path))
                    File.Delete(path);
            }
        }

        /// <summary>
        /// Commit the writing of the repository. This signals successful completion of the write.
        /// </summary>
        public void Commit()
        {
            ExceptionContext.Check(!Disposed);
            ExceptionContext.AssertValue(_closed);

            DisposeAllEntries();
            Flush();
            Dispose(true);
        }
    }

    public sealed class RepositoryReader : Repository
    {
        private ZipArchive _archive;

        // Maps from a normalized path to the entry in the _archive. This is needed since
        // a zip might use / or \ for directory separation.
        private Dictionary<string, ZipArchiveEntry> _entries;

        public static RepositoryReader Open(Stream stream, IExceptionContext ectx = null, bool useFileSystem = true)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(stream, nameof(stream));
            return new RepositoryReader(stream, ectx, useFileSystem);
        }

        private RepositoryReader(Stream stream, IExceptionContext ectx, bool useFileSystem)
            : base(useFileSystem, ectx)
        {
            try
            {
                _archive = new ZipArchive(stream, ZipArchiveMode.Read, true);
            }
            catch (Exception ex)
            {
                throw ExceptionContext.ExceptDecode(ex, "Failed to open a zip archive");
            }

            _entries = new Dictionary<string, ZipArchiveEntry>();
            foreach (var entry in _archive.Entries)
            {
                var path = NormalizeForArchiveEntry(entry.FullName);
                _entries[path] = entry;
            }
        }

        public Entry OpenEntry(string name)
        {
            return OpenEntry(null, name);
        }

        public Entry OpenEntry(string dir, string name)
        {
            var ent = OpenEntryOrNull(dir, name);
            if (ent != null)
                return ent;

            string pathEnt;
            string pathTemp;
            GetPath(out pathEnt, out pathTemp, dir, name, false);
            throw ExceptionContext.Except("Repository doesn't contain entry {0}", pathEnt);
        }

        public Entry OpenEntryOrNull(string name)
        {
            return OpenEntryOrNull(null, name);
        }

        public Entry OpenEntryOrNull(string dir, string name)
        {
            ExceptionContext.Check(!Disposed);

            string pathEnt;
            string pathTemp;
            GetPath(out pathEnt, out pathTemp, dir, name, false);

            ZipArchiveEntry entry;
            Stream stream;
            string pathAbs;
            string pathLower = pathEnt.ToLowerInvariant();
            if (PathMap.TryGetValue(pathLower, out pathAbs))
                stream = new FileStream(pathAbs, FileMode.Open, FileAccess.Read);
            else if (!_entries.TryGetValue(pathEnt, out entry))
                return null;
            else if (pathTemp != null)
            {
                // Extract to a temporary file.
                Directory.CreateDirectory(Path.GetDirectoryName(pathTemp));
                entry.ExtractToFile(pathTemp);
                PathMap.Add(pathLower, pathTemp);
                stream = new FileStream(pathTemp, FileMode.Open, FileAccess.Read);
            }
            else
            {
                // Extract to a memory stream.
                ExceptionContext.CheckDecode(entry.Length < int.MaxValue, "Repository stream too large to read into memory");
                stream = new MemoryStream((int)entry.Length);
                using (var src = entry.Open())
                    src.CopyTo(stream);
                stream.Position = 0;
            }

            return AddEntry(pathEnt, stream);
        }

        protected override void OnDispose(Entry ent)
        {
            ExceptionContext.AssertValue(ent);
            RemoveEntry(ent);
            ent.Stream.CloseEx();
        }
    }
}
