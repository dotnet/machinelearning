// owner: rragno

#define LZMA_PLAIN
#define UNBUFFERED
//#define GZIP_UNBUFFERED

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;

// TODO:
//   * Unbuffered writing
//   * temporary file support (~ prefix?)
//   * Store end emit Cosmos tools
//   * Copy compression utils from a share
//   * convert local UNC path to local path
//   * reset and restart internal gzip as 4GB occurs
//   * check encoding
//   * Concatenation support for copy, wildcards
//   * support compressed extensions in wildcard expansion
//   * command piping support
//   * C# execution support
//   * XPRESS? (LZO / UCL?)
//   * QuickLZ
//
//   * make MultiStream RAID-like
//   * parchive-like parity
//   * IFilter
//   * Compression support in Cosmos, HTTP
//   * Positioned StreamReader fixes
//   * Allow "_" as capital " "
//   * filter stream
//   * detect encoding for HTTP
//   * Unicode for console
//   * Fix zip support (possibly add zip.exe and unzip.exe as tools)
//   * Complete archive directory support
//        - Directory listings (especially for directories)
//        - wildcard pattern integration
//        - Directory removal
//        - Directory moving
//   * Pluggable decompression
//   * built-in zip and gzip handling with J#
//   * convert named streams to compressed archives
//   * clipboard binary support
//   * symbolic streams (temporaries?)
//   * Support single file/URL for Cosmos config
//   * Support programatic cosmos defaults
//   * Support default cosmos path from settings or ini
//   * I/O completion ports?
//   * Examples

//   * T, Y, Concat stream composition (+ streamnames?)
//   * Generalize provider interface!
//   * shares, shortcuts, hardlinks
//   * sparse files, NTFS compression, encryption
//   * named pipes
//   * shared memory
//   * SharedDictionary support for shared memory or such
//   * make TextReaders Enumerable
//   * (check gzip size hack)
//   * (defragmentation)

//// GZip/Compression TODO: ***
////  - Support LN tag at the beginning
////  - Enable easy adding of tag to end of existing 4GB+ gzip files
////  - Enable seekable gzip files
////  - Enable opening the file and piping (potentially with the same Stream used to get the length)
////  - Enable appending by adding a new gzip file

//// Unbuffered I/O TODO: ***
////  - investigate using FileStream with undocumented NoBuffering flag

//// Other TODO: ***
////  - Make BoundedStream avoid using .Position!

namespace Microsoft.ML.Runtime.Internal.IO
{
    #region IOUtil
    /// <summary>
    /// Utility functionality for handling paths and other I/O issues.
    /// </summary>
    public static class IOUtil
    {
        #region DLLimport
        /// <summary>
        /// Imports the Win32 APIs used by the library.
        /// </summary>
        internal class Win32
        {
            #region Constants

            public static readonly IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);
            public static readonly IntPtr NULL_HANDLE = IntPtr.Zero;

            [Flags]
            public enum FileAccess : uint
            {
                QUERY = 0x00000000,
                GENERIC_READ = 0x80000000,
                GENERIC_WRITE = 0x40000000,
                GENERIC_EXECUTE = 0x10000000,
                GENERIC_ALL = 0x10000000,
            }

            [Flags]
            public enum FileShare : uint
            {
                FILE_SHARE_READ = 1,
                FILE_SHARE_WRITE = 2,
                FILE_SHARE_DELETE = 4,
                FILE_SHARE_READ_AND_WRITE = 1 | 2,
            }

            public enum CreationDisposition : uint
            {
                CREATE_NEW = 1,
                CREATE_ALWAYS = 2,
                OPEN_EXISTING = 3,
                OPEN_ALWAYS = 4,
                TRUNCATE_EXISTING = 5,
            }

            [Flags]
            public enum FileFlag : uint
            {
                FILE_FLAG_WRITE_THROUGH = 0x80000000,
                FILE_FLAG_OVERLAPPED = 0x40000000,
                FILE_FLAG_NO_BUFFERING = 0x20000000,
                FILE_FLAG_RANDOM_ACCESS = 0x10000000,
                FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000,
                FILE_FLAG_DELETE_ON_CLOSE = 0x04000000,
                FILE_FLAG_BACKUP_SEMANTICS = 0x02000000,
                FILE_FLAG_POSIX_SEMANTICS = 0x01000000,
                FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000,
                FILE_FLAG_OPEN_NO_RECALL = 0x00100000,
                FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000,
            }

            [Flags]
            public enum FileAttributes : uint
            {
                FILE_ATTRIBUTE_READONLY = 0x00000001,
                FILE_ATTRIBUTE_HIDDEN = 0x00000002,
                FILE_ATTRIBUTE_SYSTEM = 0x00000004,
                FILE_ATTRIBUTE_DIRECTORY = 0x00000010,
                FILE_ATTRIBUTE_ARCHIVE = 0x00000020,
                FILE_ATTRIBUTE_DEVICE = 0x00000040,
                FILE_ATTRIBUTE_NORMAL = 0x00000080,
                FILE_ATTRIBUTE_TEMPORARY = 0x00000100,
                FILE_ATTRIBUTE_SPARSE_FILE = 0x00000200,
                FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400,
                FILE_ATTRIBUTE_COMPRESSED = 0x00000800,
                FILE_ATTRIBUTE_OFFLINE = 0x00001000,
                FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x00002000,
                FILE_ATTRIBUTE_ENCRYPTED = 0x00004000,
            }

            [Flags]
            public enum FileFlagsAndAttributes : uint
            {
                FILE_FLAG_WRITE_THROUGH = 0x80000000,
                FILE_FLAG_OVERLAPPED = 0x40000000,
                FILE_FLAG_NO_BUFFERING = 0x20000000,
                FILE_FLAG_RANDOM_ACCESS = 0x10000000,
                FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000,
                FILE_FLAG_DELETE_ON_CLOSE = 0x04000000,
                FILE_FLAG_BACKUP_SEMANTICS = 0x02000000,
                FILE_FLAG_POSIX_SEMANTICS = 0x01000000,
                FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000,
                FILE_FLAG_OPEN_NO_RECALL = 0x00100000,
                FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000,

                FILE_ATTRIBUTE_READONLY = 0x00000001,
                FILE_ATTRIBUTE_HIDDEN = 0x00000002,
                FILE_ATTRIBUTE_SYSTEM = 0x00000004,
                FILE_ATTRIBUTE_DIRECTORY = 0x00000010,
                FILE_ATTRIBUTE_ARCHIVE = 0x00000020,
                FILE_ATTRIBUTE_DEVICE = 0x00000040,
                FILE_ATTRIBUTE_NORMAL = 0x00000080,
                FILE_ATTRIBUTE_TEMPORARY = 0x00000100,
                FILE_ATTRIBUTE_SPARSE_FILE = 0x00000200,
                FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400,
                FILE_ATTRIBUTE_COMPRESSED = 0x00000800,
                FILE_ATTRIBUTE_OFFLINE = 0x00001000,
                FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x00002000,
                FILE_ATTRIBUTE_ENCRYPTED = 0x00004000,
            }

            public enum SeekOrigin : uint
            {
                FILE_BEGIN = 0,
                FILE_CURRENT = 1,
                FILE_END = 2,
            }

            [Flags]
            public enum AllocationType : uint
            {
                MEM_COMMIT = 0x1000,
                MEM_RESERVE = 0x2000,
                MEM_RESET = 0x80000,

                //MEM_PRIVATE = 0x20000,
                //MEM_MAPPED = 0x40000,

                /// Can be combined with the types above:
                /// <summary>
                /// This can be combined with the allocation type.
                /// </summary>
                MEM_TOP_DOWN = 0x100000,
                /// <summary>
                /// This can be combined with the allocation type.
                /// </summary>
                MEM_LARGE_PAGES = 0x20000000,
                //MEM_WRITE_WATCH = 0x200000,
                //MEM_PHYSICAL = 0x400000,
                //MEM_4MB_PAGES = 0x80000000,
            }

            public enum FreeType : uint
            {
                MEM_DECOMMIT = 0x4000,
                MEM_RELEASE = 0x8000,
                MEM_FREE = 0x10000,
            }

            [Flags]
            public enum Protect : uint
            {
                PAGE_NONE = 0x00000000,
                PAGE_NOACCESS = 0x00000001,
                PAGE_READONLY = 0x00000002,
                PAGE_READWRITE = 0x00000004,
                PAGE_WRITECOPY = 0x00000008,
                PAGE_EXECUTE = 0x00000010,
                PAGE_EXECUTE_READ = 0x00000020,
                PAGE_EXECUTE_READWRITE = 0x00000040,
                PAGE_EXECUTE_WRITECOPY = 0x00000080,
                PAGE_GUARD = 0x00000100,
                PAGE_NOCACHE = 0x00000200,
                PAGE_WRITECOMBINE = 0x00000400,
            }

            /// <summary>
            /// Access for the mapped file.
            /// </summary>
            public enum MapAccess : uint
            {
                FILE_MAP_COPY = 0x00000001,
                FILE_MAP_WRITE = 0x00000002,
                FILE_MAP_READ = 0x00000004,
                FILE_MAP_ALL_ACCESS = 0x0000001f,
            }

            #endregion

            #region Structures

            public struct IO_STATUS_BLOCK
            {
                public static readonly IO_STATUS_BLOCK NullBlock = new IO_STATUS_BLOCK(true);

                // are these still uint on x64? ***
                // If this is the wrong size, the results are disastrous...
                public uint /*ulong*/ /*IntPtr*/ /*NTSTATUS*/ Status;
                public uint /*ulong*/ /*IntPtr*/ /*ULONG_PTR*/ Information;

                private IO_STATUS_BLOCK(bool d)
                {
                    Status = 0; //IntPtr.Zero;
                    Information = 0; //IntPtr.Zero;
                }
            }

            public enum FILE_INFORMATION_CLASS : int
            {
                FileDirectoryInformation = 1,
                FileFullDirectoryInformation,
                FileBothDirectoryInformation,
                FileBasicInformation,
                FileStandardInformation,
                FileInternalInformation,
                FileEaInformation,
                FileAccessInformation,
                FileNameInformation,
                FileRenameInformation,
                FileLinkInformation,
                FileNamesInformation,
                FileDispositionInformation,
                FilePositionInformation,
                FileFullEaInformation,
                FileModeInformation,
                FileAlignmentInformation,
                FileAllInformation,
                FileAllocationInformation,
                FileEndOfFileInformation,
                FileAlternateNameInformation,
                FileStreamInformation,
                FilePipeInformation,
                FilePipeLocalInformation,
                FilePipeRemoteInformation,
                FileMailslotQueryInformation,
                FileMailslotSetInformation,
                FileCompressionInformation,
                FileCopyOnWriteInformation,
                FileCompletionInformation,
                FileMoveClusterInformation,
                FileQuotaInformation,
                FileReparsePointInformation,
                FileNetworkOpenInformation,
                FileObjectIdInformation,
                FileTrackingInformation,
                FileOleDirectoryInformation,
                FileContentIndexInformation,
                FileInheritContentIndexInformation,
                FileOleInformation,
                FileMaximumInformation
            };

            private struct FILE_ALLOCATION_INFORMATION
            {
                // is this too fragile?? ***
                public long /*LARGE_INTEGER*/ AllocationSize;
                public FILE_ALLOCATION_INFORMATION(long length)
                {
                    AllocationSize = length;
                }
            }

            #endregion

            public static bool NT_SUCCESS(uint status)
            {
                return (status <= 0x3FFFFFFF);
            }

            [DllImport("NTDLL", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern uint NtQueryInformationFile(
                IntPtr handle,
                ref IO_STATUS_BLOCK ioStatusBlock,
                byte[] fileInformation,
                uint length,
                FILE_INFORMATION_CLASS fileInformationClass);

            [DllImport("kernel32.dll", SetLastError = true)]
            public static extern IntPtr GetStdHandle(
                int nStdHandle);

            // this seems rather fragile.
            [DllImport("NTDLL", ExactSpelling = true, SetLastError = true)]
            public static extern int NtSetInformationFile(
                IntPtr   /*HANDLE*/    fileHandle,
                ref IO_STATUS_BLOCK /*PIO_STATUS_BLOCK*/ ioStatusBlock,
                ref long /*FILE_ALLOCATION_INFORMATION*/ /*IntPtr*/   /*PVOID*/     fileInformation,
                uint     /*ULONG*/     length,
                FILE_INFORMATION_CLASS fileInformationClass);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern IntPtr CreateFile(
                string fileName,
                FileAccess desiredAccess,
                FileShare shareMode,
                IntPtr securityAttributes,
                CreationDisposition creationDisposition,
                FileFlagsAndAttributes flagsAndAttributes,
                IntPtr templateFile);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool CloseHandle(
                IntPtr hObject);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool SetFilePointerEx(
                IntPtr handle,
                long offset,
                out long /*IntPtr*/ newPosition,
                SeekOrigin seekOrigin);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool GetFileSizeEx(
                IntPtr handle,
                out long newPosition);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool DeleteFile(
                string fileName);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern IntPtr VirtualAlloc(
                IntPtr lpAddress,
                IntPtr dwSize,
                AllocationType flAllocationType,
                Protect flProtect);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool VirtualFree(
                IntPtr lpAddress,
                IntPtr dwSize,
                FreeType dwFreeType);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool FlushFileBuffers(
                IntPtr handle);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool SetEndOfFile(
                IntPtr handle);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool ReadFile(
                IntPtr hFile,
                byte[] /*IntPtr*/ lpBuffer,
                int nNumberOfBytesToRead,
                out int nNumberOfBytesRead,
                //    ref OVERLAPPED lpOverlapped);
                IntPtr lpOverlapped);
            public static bool ReadFile(
                IntPtr hFile,
                byte[] /*IntPtr*/ lpBuffer,
                int nNumberOfBytesToRead,
                out int nNumberOfBytesRead)
            {
                return ReadFile(hFile, lpBuffer, nNumberOfBytesToRead, out nNumberOfBytesRead, IntPtr.Zero);
            }

            // there is likely a better way to do this...
            public class Raw
            {
                private Raw()
                {
                }

                [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
                public static extern bool ReadFile(
                    IntPtr hFile,
                    IntPtr lpBuffer,
                    int nNumberOfBytesToRead,
                    out int nNumberOfBytesRead,
                    //    ref OVERLAPPED lpOverlapped);
                    IntPtr lpOverlapped);
                public static bool ReadFile(
                    IntPtr hFile,
                    IntPtr lpBuffer,
                    int nNumberOfBytesToRead,
                    out int nNumberOfBytesRead)
                {
                    return ReadFile(hFile, lpBuffer, nNumberOfBytesToRead, out nNumberOfBytesRead, IntPtr.Zero);
                }
            }

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool WriteFile(
                IntPtr hFile,
                byte[] /*IntPtr*/ lpBuffer,
                int nNumberOfBytesToWrite,
                out int nNumberOfBytesWritten,
                //    ref OVERLAPPED lpOverlapped);
                IntPtr lpOverlapped);
            public static bool WriteFile(
                IntPtr hFile,
                byte[] /*IntPtr*/ lpBuffer,
                int nNumberOfBytesToWrite,
                out int nNumberOfBytesWritten)
            {
                return WriteFile(hFile, lpBuffer, nNumberOfBytesToWrite, out nNumberOfBytesWritten, IntPtr.Zero);
            }

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto, BestFitMapping = false)]
            public static extern bool GetDiskFreeSpace(
                string path,
                out uint sectorsPerCluster,
                out uint bytesPerSector,
                out uint numberOfFreeClusters,
                out uint totalNumberOfClusters);

            [DllImport("KERNEL32")]
            public static extern bool GetDiskFreeSpaceEx(
                string lpDirectoryName,
                out UInt64 lpFreeBytesAvailable,
                out UInt64 lpTotalNumberOfBytes,
                out UInt64 lpTotalNumberOfFreeBytes);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto)]
            public static extern IntPtr CreateFileMapping(
                IntPtr hFile,
                IntPtr lpAttributes,
                Protect flProtect,
                int dwMaximumSizeLow,
                int dwMaximumSizeHigh,
                string lpName);

            [DllImport("KERNEL32", SetLastError = true)]
            public static extern bool FlushViewOfFile(
                IntPtr lpBaseAddress,
                int dwNumBytesToFlush);

            [DllImport("KERNEL32", SetLastError = true)]
            public static extern IntPtr MapViewOfFile(
                IntPtr hFileMappingObject,
                int dwDesiredAccess,
                int dwFileOffsetHigh,
                int dwFileOffsetLow,
                int dwNumBytesToMap);

            [DllImport("KERNEL32", SetLastError = true, CharSet = CharSet.Auto)]
            public static extern IntPtr OpenFileMapping(
                int dwDesiredAccess,
                bool bInheritHandle,
                string lpName);

            [DllImport("KERNEL32", SetLastError = true)]
            public static extern bool UnmapViewOfFile(
                IntPtr lpBaseAddress);

            #region Unused
            private Win32()
            {
            }

            #endregion
        }
        #endregion DLLimport

#if TLCFULLBUILD
        #region Disk Info

        /// <summary>
        /// Get the amount of space free on a volume.
        /// </summary>
        /// <param name="path">The drive, share, directory, or file located on the volume</param>
        /// <returns>the free space on the given volume</returns>
        public static long DiskFree(string path)
        {
            // Could support cosmos... ***
            if (path == null || path.Length == 0)
                return 0;
            path = Directory.GetDirectoryRoot(path);
            ulong freeBytesAvailable;
            ulong totalNumberOfBytes;
            ulong totalNumberOfFreeBytes;
            if (!Win32.GetDiskFreeSpaceEx(path, out freeBytesAvailable, out totalNumberOfBytes, out totalNumberOfFreeBytes))
            {
                freeBytesAvailable = 0;
            }
            return (long)freeBytesAvailable;
        }

        #endregion

        #region Path Manipulation

        internal static readonly char[] pathSeparators = new char[] { '/', '\\' };

        /// <summary>
        /// Combine a base path with another partial path
        /// </summary>
        /// <param name="basePath">the base path to start in</param>
        /// <param name="subPath">the path to combine with the base path</param>
        /// <returns>the combined path</returns>
        /// <remarks>
        /// If subPath is an absolute path, it will be returned. If either path is
        /// empty or null, the other path will be returned.
        /// </remarks>
        public static string PathCombine(string basePath, string subPath)
        {
            if (basePath == null)
                return subPath;
            if (subPath == null)
                return basePath;
            if (basePath.Length == 0)
                return subPath;
            if (subPath.Length == 0)
                return basePath;

            string nameLower = subPath.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(subPath))
            {
                return subPath;
            }
            if (ZStreamIn.IsConsoleStream(subPath))
            {
                return subPath;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(subPath))
            {
                return subPath;
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                return subPath;
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                return subPath;
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:"))
            {
                return subPath;
            }
            if (nameLower.StartsWith("filelist:"))
            {
                return subPath;
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                return subPath;
            }

            if (subPath[0] == '/' || subPath[0] == '\\')
            {
                if (subPath.Length == 1 || (subPath.Length == 2 && subPath[1] == '.'))
                {
                    return PathRoot(basePath);
                }
                if (subPath[1] == '/' || subPath[1] == '\\')
                {
                    return subPath;
                }
                basePath = PathRoot(basePath);
                subPath = subPath.Substring(1);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(subPath))
            {
                return subPath;
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(subPath))
            {
                return subPath;
            }

            // examine basePath

            if (InternalStoreUtility.IsInternalStore(basePath))
            {
                // is this best?
                return basePath.Substring(0, basePath.IndexOf(':') + 1) +
                    PathCombine(InternalStoreUtility.StorePath(basePath), subPath);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(basePath))
            {
                // is this best? ***
                if (!basePath.EndsWith("/"))
                    basePath = basePath + "/";
                return basePath + subPath;
            }

            // check for Multistream:
            if (string.Compare(basePath, 0, "multi:", 0, "multi:".Length, true) == 0)
            {
                // is this best? ***
                return basePath.Substring(0, "multi:".Length) +
                    PathCombine(basePath.Substring("multi:".Length), subPath);
            }
            if (string.Compare(basePath, 0, "filelist:", 0, "filelist:".Length, true) == 0)
            {
                // is this best? ***
                return basePath.Substring(0, "filelist:".Length) +
                    PathCombine(basePath.Substring("filelist:".Length), subPath);
            }

            return ReducePath(Path.Combine(basePath, subPath));
        }

        /// <summary>
        /// Find the root element of a path.
        /// </summary>
        /// <param name="path">the path to analyze</param>
        /// <returns>the root element of the path, or the empty string if none exists</returns>
        /// <remarks>
        /// <p>
        /// This is only based on the text of the path string.
        /// </p>
        /// <p>
        /// The path returned will be canonical, so it can be compared for equality with
        /// other roots.
        /// </p>
        /// </remarks>
        public static string PathRoot(string path)
        {
            if (path == null || path.Length == 0)
                return "";
            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                //return "";
                // should it be this? The length then won't match... ***
                return ZStreamIn.NullStreamName;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                //return "";
                // should it be this? The length then won't match... ***
                return ZStreamIn.ConsoleStreamName;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                // should this be empty, or "clip:" ?
                return "clip:";
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                if (nameLower.Length < "cosmos://".Length)
                    return "cosmos:";
                int s = nameLower.IndexOfAny(pathSeparators, "cosmos://".Length);
                if (s < 0)
                {
                    return nameLower;
                }
                s = nameLower.IndexOfAny(pathSeparators, s + 1);
                if (s < 0)
                {
                    return nameLower;
                }
                return nameLower.Substring(0, s);
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                // what is right here?
                if (path.Length < "cockpit://aa".Length)
                    return path;
                int s = path.IndexOfAny(pathSeparators, "cockpit://a".Length);
                if (s < 0)
                {
                    return path;
                }
                return path.Substring(0, s);
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:") || nameLower.StartsWith("filelist:"))
            {
                // what is right here? ***
                if (nameLower.Length < "multi:a".Length)
                    return path;
                int s = path.IndexOfAny(pathSeparators, "multi:a".Length);
                if (s < 0)
                {
                    return path;
                }
                return path.Substring(0, s);
            }
            if (nameLower.StartsWith("filelist:"))
            {
                // what is right here? ***
                if (nameLower.Length < "filelist:a".Length)
                    return path;
                int s = path.IndexOfAny(pathSeparators, "filelist:a".Length);
                if (s < 0)
                {
                    return path;
                }
                return path.Substring(0, s);
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                // what is right here?
                int s = path.IndexOfAny(pathSeparators, "http://a".Length);
                if (s < 0)
                {
                    return path;
                }
                return path.Substring(0, s);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.PathRoot(path);
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return InternalStoreUtility.PathRoot(path);
                //return "store:" + PathRoot(InternalStoreUtility.StorePath(path));
            }

            if (nameLower.IndexOfAny(pathSeparators) < 0)
            {
                if (nameLower.Length == 2 && nameLower[1] == ':')
                {
                    //return nameLower + "\\";
                    return nameLower + "/";
                }
                return "";
            }

            if (path[0] == ':')
                return "";
            try
            {
                return Path.GetPathRoot(path).ToLower().Replace('\\', '/');
            }
            catch
            {
                return "";
            }
        }

        /// <summary>
        /// Find the parent of a given file or directory, based on the path.
        /// </summary>
        /// <param name="path">the original path to start from</param>
        /// <returns>the path to the parent directory, or null if none exists</returns>
        /// <remarks>
        /// The parent is determined based only on the given path information.
        /// Additionally, a file in the current directory will return ".", but
        /// this may be misleading if the path is misinterpretted. The current
        /// directory, ".", will return "..", but that might not be a valid directory.
        /// </remarks>
        public static string PathParent(string path)
        {
            if (path == null || path.Length == 0)
                return null;

            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ? null : ZStreamIn.NullStreamName;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ? null : ZStreamIn.ConsoleStreamName;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 || string.Compare(path, "clip://", true) == 0 ?
                    null : "clip:";
            }

            // check for Cosmos:
            if (nameLower.StartsWith("cosmos://"))
            {
                if (nameLower.Length < "cosmos://a/b/c".Length)
                    return null;
                int s = nameLower.IndexOfAny(pathSeparators, "cosmos://".Length);
                if (s < 0)
                {
                    return null;
                }
                s = nameLower.IndexOfAny(pathSeparators, s + 1);
                if (s < 0)
                {
                    return null;
                }
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                if (nameLower.Length < "cockpit://a/b".Length)
                    return null;
                int s = nameLower.IndexOfAny(pathSeparators, "cockpit://a".Length);
                if (s < 0)
                {
                    return null;
                }
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:"))
            {
                // what is right here? ***
                if (nameLower.Length < "multi:a/b".Length)
                    return null;
                int s = nameLower.IndexOfAny(pathSeparators, "multi:a/b".Length);
                if (s < 0)
                {
                    return null;
                }
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }
            if (nameLower.StartsWith("filelist:"))
            {
                // what is right here? ***
                if (nameLower.Length < "filelist:a/b".Length)
                    return null;
                int s = nameLower.IndexOfAny(pathSeparators, "filelist:a/b".Length);
                if (s < 0)
                {
                    return null;
                }
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                if (nameLower.Length < "http://a/b".Length)
                    return null;
                int s = nameLower.IndexOfAny(pathSeparators, "http://a".Length);
                if (s < 0)
                {
                    return null;
                }
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                string s = SqlTextReader.PathRoot(path);
                if (s == null || s.Length == 0 || s.Length <= path.Length - 1)
                    return null;
                return s;
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                // should this just always return null?
                //return null;
                return path.Substring(0, path.IndexOf(':') + 1) + PathParent(InternalStoreUtility.StorePath(path));
            }

            if (nameLower == "." || nameLower == "./" || nameLower == ".\\")
                return "..";

            // filename with no path separators:
            if (nameLower.IndexOfAny(pathSeparators) < 0)
            {
                // check for a drive specification:
                if (nameLower.Length == 2 && nameLower[1] == ':')
                {
                    return null;
                }
                // a simple filename, in the current directory?
                return ".";
            }
            // drive root:
            if (nameLower.Length == 1)
                return null;

            // path with separators:
            // UNC:
            if ((nameLower[0] == '/' || nameLower[0] == '\\') &&
                (nameLower[1] == '/' || nameLower[1] == '\\'))
            {
                if (nameLower.Length <= 3)
                    return null;
                // machine names do not count as directories!!
                // \\machine\share\path
                int s = nameLower.IndexOfAny(pathSeparators, 2);
                if (s < 0 || s == nameLower.Length - 1)
                    return null;
                s = nameLower.IndexOfAny(pathSeparators, s + 1);
                if (s < 0 || s == nameLower.Length - 1)
                    return null;
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                if (e <= s)
                    return null;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                return path.Substring(0, e + 1);
            }

            // simple file path:
            {
                int e = nameLower.Length - 1;
                if (nameLower[e] == '/' || nameLower[e] == '\\')
                    e--;
                e = nameLower.LastIndexOfAny(pathSeparators, e);
                if (e < 0)
                    return ".";
                return path.Substring(0, e + 1);
            }
        }

        /// <summary>
        /// Gets the full path of the file or directory in a standard form.
        /// </summary>
        /// <param name="path">the original path</param>
        /// <returns>the full path, with a standard casing and presentation</returns>
        /// <remarks>
        /// <p>
        /// A trailing slash will be preserved. In general, the canonical path for a directory
        /// includes the trailing slash, but it would be expensive to check for existance
        /// and insert the slash when it is omitted (and incorrect, in the case of directory
        /// paths for directories that do not yet exist).
        /// </p>
        /// <p>
        /// <see cref="PathsEqual"/> ignores a trailing slash, to account for this effect.
        /// </p>
        /// </remarks>
        public static string GetCanonicalPath(string path)
        {
            if (path == null || path.Length == 0)
                return "";
            path = path.Trim();
            if (path.Length == 0)
                return "";
            path = path.Replace('\\', '/');
            // remove repeated slashes?
            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return ZStreamIn.NullStreamName;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return ZStreamIn.ConsoleStreamName;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return "clip:";
            }

            // check for cosmos:
            if (string.Compare(path, 0, "cosmos://", 0, "cosmos://".Length, true) == 0)
            {
                string root = IOUtil.PathRoot(path);
                path = root.ToLower() +
                    path.Substring(root.Length).Replace("//", "/");

                return ReducePath(path);
            }

            // check for Cockpit:
            if (string.Compare(path, 0, "cockpit:", 0, "cockpit:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                string root = IOUtil.PathRoot(path);
                path = "cockpit:" + root.Substring("cockpit:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/').ToLower();
                return ReducePath(path);
            }

            // check for Multistream:
            if (string.Compare(path, 0, "multi:", 0, "multi:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                string root = IOUtil.PathRoot(path);
                path = "multi:" + root.Substring("multi:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/').ToLower();
                return ReducePath(path);
            }
            if (string.Compare(path, 0, "filelist:", 0, "filelist:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                string root = IOUtil.PathRoot(path);
                path = "multi:" + root.Substring("filelist:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/').ToLower();
                return ReducePath(path);
            }

            // check for HTTP:
            if (string.Compare(path, 0, "http://", 0, "http://".Length, true) == 0 ||
                string.Compare(path, 0, "https://", 0, "https://".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                string root = IOUtil.PathRoot(path);
                if (root.EndsWith(":80"))
                {
                    path = root.Substring(0, root.Length - 3).ToLower() +
                        path.Substring(root.Length);
                }
                else
                {
                    path = root.ToLower() +
                        path.Substring(root.Length);
                }
                return ReducePath(path);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.GetCanonicalPath(path);
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return InternalStoreUtility.GetCanonicalPath(path);
            }

            try
            {
                return ReducePath(Path.GetFullPath(path).Replace('\\', '/').ToLower());
            }
            catch
            {
                path = path.ToLower();
                if (path.StartsWith("//"))
                {
                    path = "/" + path.Substring(1).Replace("//", "/");
                }
                else
                {
                    path = path.Replace("//", "/");
                }
                return ReducePath(path);
            }
        }

        private static string ReducePath(string path)
        {
            try
            {
                if (path == null)
                    return null;
                if (path.Length == 0 || path == ".")
                    return path;
                bool noBack = (path.IndexOf('/') < 0) && (path.IndexOf('\\') >= 0);
                path = path.Replace('\\', '/');
                path = path.Replace("/./", "/");
                if (path.StartsWith("./"))
                    path = path.Substring(2);
                if (path.EndsWith("/."))
                    path = path.Substring(0, path.Length - 1);
                if (path.Length == 0 || path == ".")
                    return path;
                int i = 0;
                while ((i = path.IndexOf("/../", i)) >= 0)
                {
                    if (i == 0)
                    {
                        // malformed... skip.
                        i += 4;
                    }
                    else
                    {
                        int s = path.LastIndexOf('/', i - 1);
                        if (s < 0)
                            s = -1;
                        path = path.Substring(0, s + 1) + path.Substring(i + 4);
                        i = s;
                    }
                    if (i < 0 || i >= path.Length)
                        break;
                }
                if (path.Length > 3 && path.EndsWith("/.."))
                {
                    int s = path.LastIndexOf('/', path.Length - 4);
                    if (s < 0)
                    {
                        path = ".";
                    }
                    else
                    {
                        path = path.Substring(0, s + 1);
                    }
                }
                if (path.Length == 0)
                    path = ".";
                if (noBack)
                    path = path.Replace('/', '\\');
                return path;
            }
            catch
            {
                return path;
            }
        }

        /// <summary>
        /// Determine whether two paths refer to the same location.
        /// </summary>
        /// <param name="path1">the first path to compare</param>
        /// <param name="path2">the second path to compare</param>
        /// <returns>true if the paths refer to the same entity; false otherwise</returns>
        /// <remarks>
        /// This is equivalent to the expression
        /// <code>(GetCanonicalPath(path1).TrimEnd('/').Equals(GetCanonicalPath(path2).TrimEnd('/')))</code>.
        /// It will not find all paths that lead to the same location (for example, a
        /// UNC share path may refer to a local file system path, or a hardlink or junction
        /// may exist).
        /// </remarks>
        public static bool PathsEqual(string path1, string path2)
        {
            if (path1 == null || path2 == null)
                return path1 == path2;
            string path1C = GetCanonicalPath(path1);
            string path2C = GetCanonicalPath(path2);
            if (path1C.Length != path2C.Length)
            {
                if (path2C.Length < path1C.Length)
                {
                    string tmp = path1C;
                    path1C = path2C;
                    path2C = tmp;
                }
                if (path2C.Length - path1C.Length == 1 &&
                    (path2C[path2C.Length - 1] == '/' || path2C[path2C.Length - 1] == '\\'))
                {
                    return string.CompareOrdinal(path2C, 0, path1C, 0, path1C.Length) == 0;
                }
                return false;
            }
            return (path1C.Equals(path2C));
        }

        /// <summary>
        /// Gets the given path with the correct case.
        /// </summary>
        /// <param name="path">the original path</param>
        /// <returns>the path with the recorded casing</returns>
        /// <remarks>
        /// <p>
        /// For case-sensitive file systems, such as Cosmos, this will fall back
        /// to a case-insensitive match.
        /// </p>
        /// </remarks>
        public static string PathCorrectCase(string path)
        {
            if (path == null || path.Length == 0)
                return path;
            path = path.Trim();
            if (path.Length == 0)
                return "";
            string pathOrig = path;
            path = path.Replace('\\', '/');
            // remove repeated slashes?
            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return nameLower;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return nameLower;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return nameLower;
            }

            string root = IOUtil.PathRoot(path);

            // check for Cockpit:
            if (string.Compare(path, 0, "cockpit:", 0, "cockpit:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                path = "cockpit:" + root.Substring("cockpit:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/');
                return path;
            }

            // check for MultiStream:
            if (string.Compare(path, 0, "multi:", 0, "multi:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                path = "multi:" + root.Substring("multi:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/');
                return path;
            }
            if (string.Compare(path, 0, "filelist:", 0, "filelist:".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                path = "multi:" + root.Substring("filelist:".Length).Replace('\\', '/').ToUpper() +
                    path.Substring(root.Length).Replace('\\', '/');
                return path;
            }

            // check for HTTP:
            if (string.Compare(path, 0, "http://", 0, "http://".Length, true) == 0 ||
                string.Compare(path, 0, "https://", 0, "https://".Length, true) == 0)
            {
                // should we trim a trailing slash in this case??
                if (root.EndsWith(":80"))
                {
                    path = root.Substring(0, root.Length - 3).ToLower() +
                        path.Substring(root.Length);
                }
                else
                {
                    path = root.ToLower() +
                        path.Substring(root.Length);
                }
                return path;
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.GetCanonicalPath(path);
            }

            StringBuilder res = new StringBuilder(path.Length);
            int pos = 0;

            // cosmos and normal file system:
            res.Append(root.ToLower());
            if ((root.EndsWith("\\") || root.EndsWith("/")) &&
                pathOrig.Length >= root.Length)
            {
                res[root.Length - 1] = pathOrig[root.Length - 1];
            }
            pos = root.Length;
            while (pos < path.Length)
            {
                int next = path.IndexOf('/', pos);
                if (next < 0)
                    next = path.Length;
                if (next == pos)
                {
                    res.Append(pathOrig[pos]);
                    pos++;
                    continue;
                }
                string[] dirs = DirectoryEntries(pathOrig.Substring(0, pos));
                if (dirs == null || dirs.Length == 0)
                {
                    //Console.WriteLine("No entries found: " + res.ToString());
                    res.Append(pathOrig.Substring(pos));
                    break;
                }
                string found = null;
                for (int i = 0; i < dirs.Length; i++)
                {
                    int len = dirs[i].Length;
                    if (dirs[i][dirs[i].Length - 1] == '/')
                        len--;
                    int dstart = dirs[i].LastIndexOfAny(pathSeparators, len - 1);
                    if (dstart < 0)
                    {
                        dstart = 0;
                    }
                    else
                    {
                        dstart++;
                        len = len - dstart;
                    }
                    if (len == next - pos &&
                        string.Compare(path, pos, dirs[i], dstart, next - pos, true) == 0)
                    {
                        found = dirs[i].Substring(dstart, next - pos);
                        if (string.Compare(path, pos, dirs[i], dstart, next - pos, false) == 0)
                        {
                            break;
                        }
                    }
                    else
                    {
                        //Console.WriteLine("  non-match: " + dirs[i] + " : " +
                        //    dirs[i].Substring(dstart, next - pos) +
                        //    " != " + path.Substring(pos, next - pos));
                    }
                }
                if (found == null)
                {
                    //Console.WriteLine("No matching entries found: " + res.ToString() + "  " + path.Substring(pos, next - pos));
                    res.Append(pathOrig.Substring(pos));
                    break;
                }

                res.Append(found);
                if (next < pathOrig.Length)
                {
                    res.Append(pathOrig[next]);
                }
                pos = next + 1;
            }

            if (IsFileSystemPath(res.ToString()))
            {
                res.Replace('/', '\\');
            }
            return res.ToString();
        }

        /// <summary>
        /// Get the filename, without any path or protocol information.
        /// </summary>
        /// <param name="path">the path or stream name</param>
        /// <returns>the name of the file, or null if the name does not exist, or the empty
        /// string if the path is a directory</returns>
        /// <remarks>
        /// This may fail if the path is a directory, since it operates only on the string. In
        /// that case, it could return the name of the last directory, if the path does not end on a
        /// directory seperator character. However, it will attempt to
        /// return an empty string in the case of directories.
        /// </remarks>
        public static string GetFileName(string path)
        {
            if (path == null || path.Length == 0)
                return null;

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                //return null;
                //return path;
                return ZStreamIn.NullStreamName;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                //return path;
                return ZStreamIn.ConsoleStreamName;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                //return null;
                //return path;
                return "clip:";
            }

            string pathLower = path.ToLower();
            // check for Cosmos:
            if (pathLower.StartsWith("cosmos:"))
            {
                if (path[path.Length - 1] == '/' || path[path.Length - 1] == '\\')
                {
                    // a directory...
                    return "";
                }
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "cosmos://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for Cockpit:
            if (pathLower.StartsWith("cockpit:"))
            {
                if (path[path.Length - 1] == '/' || path[path.Length - 1] == '\\')
                {
                    // a directory...
                    return "";
                }
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "cockpit://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for Multistream:
            if (pathLower.StartsWith("multi:"))
            {
                if (path[path.Length - 1] == '/' || path[path.Length - 1] == '\\')
                {
                    // a directory...
                    return "";
                }
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "multi:a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }
            if (pathLower.StartsWith("filelist:"))
            {
                if (path[path.Length - 1] == '/' || path[path.Length - 1] == '\\')
                {
                    // a directory...
                    return "";
                }
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "filelist:a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for HTTP:
            if (pathLower.StartsWith("http:") || pathLower.StartsWith("https:"))
            {
                if (path[path.Length - 1] == '/' || path[path.Length - 1] == '\\')
                {
                    // a directory...
                    return "";
                }
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "http://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.GetFileName(path);
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                // maybe not right? ***
                // return:
                //  - whole path?
                //  - inner path of directory?
                //  - directory name alone?    <
                // only return it here if explicitly repeated??
                path = path.TrimEnd(pathSeparators);
                int e = path.LastIndexOfAny(pathSeparators);
                if (e < 0)
                {
                    e = path.IndexOf(':');
                }
                if (e < 0)
                    return null;
                return path.Substring(e + 1).TrimEnd(pathSeparators);
            }

            // plain file:
            try
            {
                return Path.GetFileName(path);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get the filename or directory name, without any path or protocol information.
        /// </summary>
        /// <param name="path">the file or directory name</param>
        /// <returns>the name of the file or directory, or null if the name does not exist</returns>
        /// <remarks>
        /// Unlike <see cref="GetFileName"/>, this method attempts to return the name
        /// for directories as well. This still operates only on the path string.
        /// </remarks>
        public static string GetName(string path)
        {
            if (path == null || path.Length == 0)
                return null;

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                //return null;
                //return path;
                return ZStreamIn.NullStreamName;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                //return path;
                return ZStreamIn.ConsoleStreamName;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                //return null;
                //return path;
                return "clip:";
            }

            path = path.TrimEnd(pathSeparators);
            string pathLower = path.ToLower();
            // check for Cosmos:
            if (pathLower.StartsWith("cosmos:"))
            {
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "cosmos://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for Cockpit:
            if (pathLower.StartsWith("cockpit:"))
            {
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "cockpit://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for Multistream:
            if (pathLower.StartsWith("multi:"))
            {
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "multi:a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }
            if (pathLower.StartsWith("filelist:"))
            {
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "filelist:a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for HTTP:
            if (pathLower.StartsWith("http:") || pathLower.StartsWith("https:"))
            {
                int lastSep = path.LastIndexOfAny(pathSeparators);
                if (lastSep < "http://a".Length)
                {
                    return "";
                }
                return path.Substring(lastSep + 1);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.GetName(path);
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                // maybe not right? ***
                // return:
                //  - whole path?
                //  - inner path of directory?
                //  - directory name alone?    <
                int e = path.LastIndexOfAny(pathSeparators);
                if (e < 0)
                {
                    e = path.IndexOf(':');
                }
                if (e < 0)
                    return null;
                return path.Substring(e + 1).TrimEnd(pathSeparators);
            }

            // plain file:
            try
            {
                return Path.GetFileName(path);
            }
            catch
            {
                return null;
            }
        }

        private static bool PathAncestor(string ancestor, string path2)
        {
            if (ancestor == null || ancestor.Length == 0 || path2 == null || path2.Length == 0)
            {
                return false;
            }
            ancestor = ancestor.TrimEnd(pathSeparators);
            path2 = path2.TrimEnd(pathSeparators);
            if (ancestor.Length > path2.Length)
                return false;
            ancestor = ancestor.Replace('\\', '/');
            path2 = path2.Replace('\\', '/');
            if (string.Compare(ancestor, 0, path2, 0, ancestor.Length, !ancestor.StartsWith("cosmos:")) != 0)
            {
                return false;
            }
            if (ancestor.Length == path2.Length || path2[ancestor.Length] == '/')
            {
                return true;
            }
            return false;
        }

        /// <summary>
        /// Determine if a path refers to a true file system path, rather than
        /// a stream name.
        /// </summary>
        /// <param name="path">the path to test</param>
        /// <returns>true if path is a file system file or directory, false otherwise</returns>
        /// <remarks>
        /// <p>
        /// The result is based on the path string alone; it does not reflect whether the
        /// given item actually exists.
        /// </p>
        /// <p>
        /// For compressed files, a result of true merely means that the path refers
        /// to a compressed file on a file system. Reading or writing the file without
        /// InternalStreams will still reslt in different behavior, since the raw file will be
        /// read. Similarly, a named stream will not work with standard .NET file
        /// operations.
        /// </p>
        /// </remarks>
        private static bool IsFileSystemPath(string path)
        {
            if (path == null || path.Length == 0)
                return false;
            if (path[path.Length - 1] == '$')
                return false;
            if (path[0] == '/' || path[0] == '\\')
                return true;
            if (path.Length == 2 && path[1] == ':')
                return true;
            if (path.Length > 2 && path[1] == ':' &&
                (path[2] == '/' || path[2] == '\\'))
                return true;

            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return false;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return false;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return false;
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                return false;
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                return false;
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:"))
            {
                return false;
            }
            if (nameLower.StartsWith("filelist:"))
            {
                return false;
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                return false;
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return false;
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return false;
            }

            return true;
        }

        private static readonly char[] _wildChars = new char[] { '*', '?' };
        private static readonly char[] _wildPlusChars = new char[] { '*', '?', '+' };

        /// <summary>
        /// Expand an extended wildcard pattern into a set of file paths.
        /// </summary>
        /// <param name="pattern">the pattern to expand</param>
        /// <returns>the set of file paths matching the pattern</returns>
        /// <remarks>
        /// The wildcard pattern accepts the standard "*" and "?" placeholders.
        /// "..." also refers to a recursive search over subdirectories.
        /// "+" can also be used to make a union of several filenames or patterns.
        /// In addition to filenames, HTTP URLs, <c>nul</c>, <c>null</c>, <c>$</c>,
        /// <c>-</c>, and Cosmos stream names are all recognized as elements.
        /// Names of files that do not exist will be excluded.
        /// </remarks>
        public static string[] ExpandWildcards(string pattern)
        {
            if (pattern == null || (pattern.IndexOfAny(_wildPlusChars) < 0 && pattern.IndexOf("...") < 0))
            {
                if (FileExists(pattern))
                {
                    return new string[] { pattern };
                }
                else
                {
                    return new string[0];
                }
            }
            List<string> matchList = new List<string>();
            bool disjoint = false;
            int filePatternCount = 0;
            string[] patterns;
            // try to avoid bad splitting?
            if (string.Compare(pattern, 0, "http:", 0, 5, true) == 0 ||
                string.Compare(pattern, 0, "https:", 0, 6, true) == 0)
            {
                patterns = new string[] { pattern };
            }
            else
            {
                patterns = pattern.Split('+');
            }
            foreach (string pat in patterns)
            {
                // hard-code in special types??
                if (pat.Length == 0)
                    continue;
                string patLower = pat.ToLower();
                if (pat == "$" || pat == "-" ||
                    patLower.StartsWith("cosmos:") ||
                    patLower.StartsWith("cockpit:") ||
                    patLower.StartsWith("http:") || patLower.StartsWith("https:") ||
                    InternalStoreUtility.IsInternalStore(pat) ||
                    SqlTextReader.IsSqlTextReader(pat) ||
                    patLower == "nul" || patLower == "null")
                {
                    // hack for at least some wildcards in cosmos, cockpit:
                    if (patLower.StartsWith("cosmos:") &&
                        (pat.IndexOfAny(_wildChars) >= 0 || pat.IndexOf("...") >= 0))
                    {
                        matchList.AddRange(Cosmos.DirectoryEntries(pat, true, false, false));
                    }
                    else if (patLower.StartsWith("cockpit:") &&
                        (pat.EndsWith("/*") || pat.EndsWith("\\*")))
                    {
                        matchList.AddRange(DirectoryFiles(pat.Substring(0, pat.Length - 1)));
                    }
                    else if ((patLower.StartsWith("http:") || patLower.StartsWith("https:")) &&
                             pat.IndexOf("*") > pat.LastIndexOf("/"))
                    {
                        //Is this a wildcard blob path? get all the blobs under this directory
                        int index = pat.LastIndexOf("/");
                        string filter = pat.Substring(index + 1);
                        AzureStorageIO azureIO = new AzureStorageIO();
                        List<string> blobsInPath = azureIO.ListBlobsInPath(pat.Remove(index), filter);
                        matchList.AddRange(blobsInPath);
                    }
                    else
                    {
                        if (FileExists(pat))
                            matchList.Add(pat);
                    }
                }
                else if (patLower.StartsWith("multi:") || patLower.StartsWith("filelist:"))
                {
                    // add in multistream completions??
                    if (patLower.StartsWith("multi:"))
                    {
                        patLower = patLower.Substring("multi:".Length);
                    }
                    else
                    {
                        patLower = patLower.Substring("filelist:".Length);
                    }
                    string[] mList = ExpandWildcards(patLower);
                    for (int i = 0; i < mList.Length; i++)
                    {
                        if (mList[i].Length == 0)
                            continue;
                        matchList.Add("multi:" + mList[i]);
                    }
                }
                else
                {
                    filePatternCount++;
                    int prepatternCount = matchList.Count;
                    if (pat.IndexOfAny(_wildChars) >= 0 || pat.IndexOf("...") >= 0)
                    {
                        // compressed extensions are not automatically used! ***
                        int recursiveIndex = pat.IndexOf("...");
                        if (recursiveIndex >= 0)
                        {
                            string left = pat.Substring(0, recursiveIndex);
                            string right = pat.Substring(recursiveIndex + 3);
                            right = right.TrimStart('\\', '/');
                            if (right.Length == 0)
                                right = "*";
                            string path = left;
                            bool pathEmpty = (path == null || path.Length == 0);
                            if (pathEmpty)
                                path = ".";
                            Stack dirsLeft = new Stack();
                            dirsLeft.Push(path);
                            while (dirsLeft.Count != 0)
                            {
                                string dir = (string)dirsLeft.Pop();

                                // watch for lack of access:
                                try
                                {
                                    // this is actually incorrect, for 3-char extensions: ***
                                    string[] files = Directory.GetFiles(dir, right);
                                    if (pathEmpty)
                                    {
                                        for (int i = 0; i < files.Length; i++)
                                        {
                                            if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                                files[i] = files[i].Substring(2);
                                        }
                                    }
                                    matchList.AddRange(files);

                                    string[] subs = Directory.GetDirectories(dir);
                                    for (int s = subs.Length - 1; s >= 0; s--)
                                    {
                                        dirsLeft.Push(subs[s]);
                                    }
                                }
                                catch
                                {
                                    // ignore
                                }
                            }
                        }
                        else
                        {
                            try
                            {
                                string path = Path.GetDirectoryName(pat);
                                bool pathEmpty = !(pat.StartsWith("./") || pat.StartsWith(".\\"));
                                if (path == null || path.Length == 0)
                                    path = ".";
                                // watch for lack of access:
                                try
                                {
                                    string[] files = Directory.GetFiles(path, Path.GetFileName(pat));
                                    if (pathEmpty)
                                    {
                                        for (int i = 0; i < files.Length; i++)
                                        {
                                            if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                                files[i] = files[i].Substring(2);
                                        }
                                    }
                                    matchList.AddRange(files);
                                }
                                catch
                                {
                                    // ignore
                                }
                            }
                            catch
                            {
                                // ignore bad path?
                            }
                        }
                    }
                    else
                    {
                        // what to do?? Filter to only those that exist?? ***
                        if (!FileExists(pat))
                            continue;
                        matchList.Add(pat);
                    }
                    if (filePatternCount > 1 && matchList.Count != prepatternCount)
                    {
                        disjoint = true;
                    }
                }
            }
            if (disjoint || true)
            {
                // remove duplicates, very inefficiently - but it is simple, preserves
                // the order, uses no additional memory, and is case-insensitive...:
                for (int i = 0; i < matchList.Count - 1; i++)
                {
                    for (int j = i + 1; j < matchList.Count; j++)
                    {
                        bool caseInsensitive = !(string.Compare((string)matchList[i], 0, "cosmos:", 0, "cosmos:".Length, true) == 0);
                        {
                            if (string.Compare((string)matchList[i], (string)matchList[j], caseInsensitive) == 0)
                            {
                                matchList.RemoveAt(j);
                                j--;
                            }
                        }
                    }
                }
            }
            return matchList.ToArray();
        }

        #endregion

        #region Directory Operations

        /// <summary>
        /// Determine if a directory or archive exists.
        /// </summary>
        /// <param name="path">the directory to look for</param>
        /// <returns>true if the directory exists, false otherwise</returns>
        /// <remarks>
        /// This will only detect directories on file paths, Cosmos, and
        /// compressed archives.
        /// </remarks>
        public static bool DirectoryExists(string path)
        {
            if (path == null || path.Length == 0)
                return false;
            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                // allow self-directory:
                return path.IndexOfAny(pathSeparators) < 0;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                // allow self-directory:
                return path.IndexOfAny(pathSeparators) < 0;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ||
                    string.Compare(path, "clip://", true) == 0;
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                return Cosmos.DirectoryExists(path);
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                // what is right here?
                string parent = PathParent(path);
                if (parent == null)
                {
                    // assume it exists??
                    return true;
                }
                string name = GetName(path);
                string[] dirs = DirectoryEntries(parent, false, true);
                for (int i = 0; i < dirs.Length; i++)
                {
                    string dir = GetName(dirs[i]).Trim(pathSeparators);
                    if (string.Compare(name, dir, true) == 0)
                    {
                        return true;
                    }
                }
                return false;
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:"))
            {
                return false;
            }
            if (nameLower.StartsWith("filelist:"))
            {
                return false;
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                // what is right here?
                return false;  //HttpStream.Exists(path);
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.ExistsDatabase(path);
            }

            // InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return DirectoryExists(InternalStoreUtility.StorePath(path));
            }

            if (Directory.Exists(path))
                return true;

            if ((path.StartsWith("\\\\") || path.StartsWith("//")) &&
                !Directory.Exists(Path.GetPathRoot(path)))
            {
                return false;
            }

            // check for compressed archive:
            for (int i = 0; i < ZStreamIn.decompressionArchiveExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionArchiveExtensions[i];
                if ((nameLower.EndsWith(ext) && File.Exists(path)) ||
                    File.Exists(path + ext))
                {
                    return true;
                }
            }

            // check for compressed archives in path:
            // only one path segment is allowed to be an archive...
            // normalize path:
            if (path[path.Length - 1] != '\\')
                path = path + "\\";
            path = path.Replace('/', '\\');
            bool isUnc = path.StartsWith("\\\\");
            while (path.IndexOf("\\\\") >= 0)
            {
                path = path.Replace("\\\\", "\\");
            }
            if (isUnc)
                path = "\\" + path;
            nameLower = path.ToLower();

            string archPath = null;
            string inArch = null;
            // should non-archives be considered as one-file directories??
            for (int i = 0; i < ZStreamIn.decompressionArchiveExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionArchiveExtensions[i];
                int seg = nameLower.IndexOf(ext + "\\");
                if (seg > 0)
                {
                    archPath = path.Substring(0, seg + ext.Length);
                    if (File.Exists(archPath))
                    {
                        inArch = path.Substring(seg + ext.Length);
                        break;
                    }
                    archPath = null;
                }
            }
            if (archPath == null)
            {
#if !QUICK_ARCHIVE_SEARCH
                // add in extension to each segment...
                string[] segs = path.Split('\\');
                for (int i = 0; i < segs.Length; i++)
                {
                    if (segs[i].Length == 0)
                        continue;
                    string partial = string.Join("\\", segs, 0, i + 1);
                    if (partial.Length == 2 && partial[1] == ':')
                        continue;
                    if (Directory.Exists(partial))
                        continue;
                    for (int c = 0; c < ZStreamIn.decompressionArchiveExtensions.Length; c++)
                    {
                        string ext = ZStreamIn.decompressionArchiveExtensions[c];
                        if (File.Exists(partial + ext))
                        {
                            archPath = partial + ext;
                            inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1);
                            break;
                        }
                    }
                    // quit when parent will not exist
                    break;
#else
                // add in extension to last segment...
                string[] segs = path.TrimEnd('\\').Split('\\');
                for (int i = 0; i < segs.Length; i++)
                {
                    if (segs[i].Length == 0)  continue;
                    string partial = string.Join("\\", segs, 0, segs.Length - 1);
                    for (int c = 0; c < ZStreamIn.decompressionArchiveExtensions.Length; c++)
                    {
                        string ext = ZStreamIn.decompressionArchiveExtensions[c];
                        if (File.Exists(partial + ext))
                        {
                            archPath = partial + ext;
                            inArch = segs[segs.Length - 1];
                            break;
                        }
                    }
                    if (archPath != null)  break;
                    // quit when parent will not exist
                    if (!Directory.Exists(partial))  break;
#endif
                }
            }
            if (archPath != null)
            {
                // check for path in archive.
                inArch = inArch.Trim('\\');
                if (inArch.Length == 0)
                    return true;
                // Must check directories...
                if (Z7zDecodeStream.Exists(archPath, inArch + "/*"))
                    return true;
                if (!Z7zDecodeStream.Exists7z && Path.GetExtension(archPath).ToLower() == ".rar")
                {
                    if (RarDecodeStream.Exists(archPath, inArch + "/*"))
                        return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Create a directory or archive.
        /// </summary>
        /// <param name="path">the directory or archive name to create</param>
        /// <remarks>
        /// <p>
        /// Archives will be created empty.
        /// </p>
        /// <p>
        /// When a directory already exists, this method will silently do nothing.
        /// </p>
        /// </remarks>
        /// <exception cref="IOException">The directory cannot be created.</exception>
        public static void CreateDirectory(string path)
        {
            if (path == null || path.Length == 0)
                throw new IOException("Cannot create directory for empty path");
            string nameLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return;
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return;
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return;
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                Cosmos.CreateDirectory(path);
                return;
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                throw new IOException("Cannot create directory for Cockpit");
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:") || nameLower.StartsWith("filelist:"))
            {
                throw new IOException("Cannot create directory for Multistream");
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                throw new IOException("Cannot create directory for HTTP");
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                throw new IOException("Cannot create directory for SQL");
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                throw new IOException("Cannot create directory for InternalStore");
            }

            if (Directory.Exists(path))
                return;

            // check for compressed archive:
            for (int i = 0; i < ZStreamOut.compressionArchiveExtensions.Length; i++)
            {
                string ext = ZStreamOut.compressionArchiveExtensions[i];
                if ((nameLower.EndsWith(ext)))
                {
                    if (!File.Exists(path))
                    {
                        try
                        {
                            //Console.WriteLine("Opening: " + path);
                            using (Stream s = ZStreamOut.Open(path))
                            {
                                s.Close();
                                if (s is CmdStream && ((CmdStream)s).ExitCode != 0)
                                {
                                    throw new IOException("Failed to create archive");
                                }
                            }
                            try
                            {
                                Delete(path + "/" + Path.GetFileNameWithoutExtension(path));
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine(ex);
                            throw new IOException("Failed to create archive");
                        }
                    }
                    return;
                }
            }

            try
            {
                Directory.CreateDirectory(path);
            }
            catch
            {
                throw new IOException("Failed to create directory");
            }
        }

        /// <summary>
        /// Get the paths to files within a directory.
        /// </summary>
        /// <param name="path">the directory to look in</param>
        /// <returns>the set of paths for the files in that directory</returns>
        /// <remarks>
        /// This will silently return the empty list if there are any problems.
        /// </remarks>
        public static string[] DirectoryFiles(string path)
        {
            return DirectoryEntries(path, true, false);
        }

        /// <summary>
        /// Get the paths to directories within a directory.
        /// </summary>
        /// <param name="path">the directory to look in</param>
        /// <returns>the set of paths for the directories in that directory</returns>
        /// <remarks>
        /// This will silently return the empty list if there are any problems.
        /// </remarks>
        public static string[] DirectoryDirectories(string path)
        {
            return DirectoryEntries(path, false, true);
        }

        /// <summary>
        /// Get the paths to files and directories within a directory.
        /// </summary>
        /// <param name="path">the directory to look in</param>
        /// <returns>the set of paths for the files and directories in that directory</returns>
        /// <remarks>
        /// <p>
        /// Directories will be distinguished by ending with "/".
        /// </p>
        /// <p>
        /// This will silently return the empty list if there are any problems.
        /// </p>
        /// </remarks>
        public static string[] DirectoryEntries(string path)
        {
            return DirectoryEntries(path, true, true);
        }

        private static string[] DirectoryEntries(string path, bool allowFile, bool allowDirectory)
        {

            if (path == null || path.Length == 0)
                return new string[0];
            if (!allowFile && !allowDirectory)
                return new string[0];
            if (path[path.Length - 1] != '\\' && path[path.Length - 1] != '/')
            {
                path = path + "/";
            }

            string pathLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ?
                    new string[] { ZStreamIn.NullStreamName } : new string[0];
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ?
                    new string[] { ZStreamIn.ConsoleStreamName } : new string[0];
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ||
                    string.Compare(path.TrimEnd(pathSeparators), "clip:", true) == 0 ?
                    new string[] { "clip:" } : new string[0];
            }

            // check for cosmos:
            if (pathLower.StartsWith("cosmos:"))
            {
                return Cosmos.DirectoryEntries(path, allowFile, allowDirectory, false);
            }

            // check for Cockpit:
            if (pathLower.StartsWith("cockpit:"))
            {
                StreamInfo[] entries = DirectoryEntriesInfo(path, allowFile, allowDirectory);
                if (entries == null)
                    return null;
                string[] res = new string[entries.Length];
                for (int i = 0; i < res.Length; i++)
                {
                    res[i] = entries[i].Path;
                }
                return res;
            }

            // check for Multistream:
            if (pathLower.StartsWith("multi:") || pathLower.StartsWith("filelist:"))
            {
                return new string[0];
            }

            // check for HTTP:
            if (pathLower.StartsWith("http://") || pathLower.StartsWith("https://"))
            {
                return new string[0];
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                return SqlTextReader.DatabaseTablePaths(path);
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return InternalStoreUtility.DirectoryEntries(path);
            }

            if (Directory.Exists(path))
            {
                try
                {
                    if (!allowDirectory)
                    {
                        return Directory.GetFiles(path);
                    }
                    else if (!allowFile)
                    {
                        string[] resDirs = Directory.GetDirectories(path);
                        for (int i = 0; i < resDirs.Length; i++)
                        {
                            resDirs[i] = resDirs[i] + "/";
                        }
                        return resDirs;
                    }
                    else
                    {
                        string[] res = Directory.GetFiles(path);
                        string[] resDirs = Directory.GetDirectories(path);
                        if (resDirs.Length != 0)
                        {
                            string[] resFiles = res;
                            res = new string[res.Length + resDirs.Length];
                            for (int i = 0; i < resDirs.Length; i++)
                            {
                                res[i] = resDirs[i] + "/";
                            }
                            if (resFiles.Length != 0)
                            {
                                Array.Copy(resFiles, 0, res, resDirs.Length, resFiles.Length);
                            }
                        }
                        return res;
                    }
                }
                catch
                {
                    return new string[0];
                }
            }

            if ((path.StartsWith("\\\\") || path.StartsWith("//")) &&
                !Directory.Exists(Path.GetPathRoot(path)))
            {
                return new string[0];
            }

            // check for compressed forms:
            string pathBase = pathLower.Substring(0, path.Length - 1);
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionExtensions[i];
                if (pathBase.EndsWith(ext) && File.Exists(pathBase))
                {
                    string[] res = Z7zDecodeStream.DirectoryEntries(pathBase, null, allowFile, allowDirectory);
                    if (res.Length == 0 && !Z7zDecodeStream.Exists7z && ext == ".rar")
                    {
                        res = RarDecodeStream.DirectoryEntries(pathBase, null, allowFile, allowDirectory);
                    }
                    return res;
                }
                else
                {
                    string pathBaseExt = pathBase + ext;
                    if (File.Exists(pathBaseExt))
                    {
                        string[] res = Z7zDecodeStream.DirectoryEntries(pathBaseExt, null, allowFile, allowDirectory);
                        if (res.Length == 0 && !Z7zDecodeStream.Exists7z && ext == ".rar")
                        {
                            res = RarDecodeStream.DirectoryEntries(pathBaseExt, null, allowFile, allowDirectory);
                        }
                        return res;
                    }
                }
            }

            // check for compressed archive as directory segment:
            // check for compressed archives in path:
            // only one path segment is allowed to be an archive...
            // normalize path:
            string zfileName = path.Replace('/', '\\');
            bool isUnc = zfileName.StartsWith("\\\\");
            while (zfileName.IndexOf("\\\\") >= 0)
            {
                zfileName = zfileName.Replace("\\\\", "\\");
            }
            if (isUnc)
                zfileName = "\\" + zfileName;
            string znameLower = zfileName.ToLower();

            string archPath = null;
            string inArch = null;
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionExtensions[i];
                int seg = znameLower.IndexOf(ext + "\\");
                if (seg > 0)
                {
                    archPath = zfileName.Substring(0, seg + ext.Length);
                    if (File.Exists(archPath))
                    {
                        inArch = zfileName.Substring(seg + ext.Length).Trim('/', '\\');
                        break;
                    }
                    archPath = null;
                }
            }
            if (archPath == null)
            {
                // add in extension to each segment...
                string[] segs = zfileName.Split('\\');
                for (int i = 0; i < segs.Length; i++)
                {
                    if (segs[i].Length == 0)
                        continue;
                    string partial = string.Join("\\", segs, 0, i + 1);
                    if (partial.Length == 2 && partial[1] == ':')
                        continue;
                    if (Directory.Exists(partial))
                        continue;
                    for (int c = 0; c < ZStreamIn.decompressionExtensions.Length; c++)
                    {
                        string ext = ZStreamIn.decompressionExtensions[c];
                        if (File.Exists(partial + ext))
                        {
                            archPath = partial + ext;
                            inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1).Trim('/', '\\');
                            break;
                        }
                    }
                    // quit when parent will not exist
                    break;
                }
            }
            if (archPath != null)
            {
                // check for path in archive.
                inArch = inArch.Trim('\\');
                if (inArch.Length == 0)
                    return new string[0];
                string[] res = Z7zDecodeStream.DirectoryEntries(archPath, inArch + "\\*", allowFile, allowDirectory);
                if (res.Length == 0 && !Z7zDecodeStream.Exists7z && Path.GetExtension(archPath).ToLower() == ".rar")
                {
                    res = RarDecodeStream.DirectoryEntries(archPath, inArch + "\\*", allowFile, allowDirectory);
                }
                return res;
            }

            return new string[0];
        }

        /// <summary>
        /// Get the <see cref="StreamInfo"/> objects for files within a directory.
        /// </summary>
        /// <param name="path">the directory to look in</param>
        /// <returns>the set of <see cref="StreamInfo"/> objects for the files in that directory</returns>
        /// <remarks>
        /// This will silently return the empty list if there are any problems.
        /// </remarks>
        public static StreamInfo[] DirectoryFilesInfo(string path)
        {
            return DirectoryEntriesInfo(path, true, false);
        }

        /// <summary>
        /// Get the <see cref="StreamInfo"/> objects for files and directories within a directory.
        /// </summary>
        /// <param name="path">the directory to look in</param>
        /// <returns>the set of <see cref="StreamInfo"/> objects for the files and directories in that directory</returns>
        /// <remarks>
        /// <p>
        /// This will silently return the empty list if there are any problems.
        /// </p>
        /// </remarks>
        public static StreamInfo[] DirectoryEntriesInfo(string path)
        {
            return DirectoryEntriesInfo(path, true, true);
        }

        private static StreamInfo[] DirectoryEntriesInfo(string path, bool allowFile, bool allowDirectory)
        {
            // is it worth keeping this distinct from DirectoryEntries for efficiency?? ****

            if (path == null || path.Length == 0)
                return new StreamInfo[0];
            if (!allowFile && !allowDirectory)
                return new StreamInfo[0];
            if (path[path.Length - 1] != '\\' && path[path.Length - 1] != '/')
            {
                path = path + (path.IndexOf('/') < 0 ? "\\" : "/");
            }

            string pathLower = path.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ?
                    new StreamInfo[] { new StreamInfo(ZStreamIn.NullStreamName, 0, DateTime.MinValue) } : new StreamInfo[0];
            }
            if (ZStreamIn.IsConsoleStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ?
                    new StreamInfo[] { new StreamInfo(ZStreamIn.ConsoleStreamName, 0, DateTime.MinValue) } : new StreamInfo[0];
            }

            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(path))
            {
                return path.IndexOfAny(pathSeparators) < 0 ||
                    string.Compare(path.TrimEnd(pathSeparators), "clip:", true) == 0 ?
                    new StreamInfo[] { new StreamInfo("clip:", 0, DateTime.MinValue) } : new StreamInfo[0];
            }

            // check for cosmos:
            if (pathLower.StartsWith("cosmos:"))
            {
                return Cosmos.DirectoryEntriesInfo(path, allowFile, allowDirectory, false);
            }

            // check for Cockpit:
            if (pathLower.StartsWith("cockpit:"))
            {
                return CockpitDirectoryEntriesInfo(path, allowFile, allowDirectory);
            }

            // check for Multistream:
            if (pathLower.StartsWith("multi:") || pathLower.StartsWith("filelist:"))
            {
                return new StreamInfo[0];
            }

            // check for HTTP:
            if (pathLower.StartsWith("http://") || pathLower.StartsWith("https://"))
            {
                return new StreamInfo[0];
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(path))
            {
                if (allowFile)
                {
                    string[] files = SqlTextReader.DatabaseTablePaths(path);
                    StreamInfo[] res = new StreamInfo[files.Length];
                    for (int i = 0; i < res.Length; i++)
                    {
                        long len = 0;
                        DateTime lastMod = DateTime.MinValue;
                        res[i] = new StreamInfo(files[i], len, lastMod);
                    }
                    return res;
                }
                return new StreamInfo[0];
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(path))
            {
                return InternalStoreUtility.DirectoryEntriesInfo(path);
            }

            if (Directory.Exists(path))
            {
                try
                {
                    StreamInfo[] res = null;
                    if (allowFile)
                    {
                        string[] files = Directory.GetFiles(path);
                        res = new StreamInfo[files.Length];
                        for (int i = 0; i < res.Length; i++)
                        {
                            long len = 0;
                            DateTime lastMod = DateTime.MinValue;
                            try
                            {
                                FileInfo f = new FileInfo(files[i]);
                                len = f.Length;
                                lastMod = f.LastWriteTime;
                            }
                            catch
                            {
                                // ignore?
                            }
                            res[i] = new StreamInfo(files[i], len, lastMod);
                        }
                    }
                    if (!allowDirectory)
                        return res;

                    string[] dirs = Directory.GetDirectories(path);
                    if (res == null || res.Length == 0)
                    {
                        res = new StreamInfo[dirs.Length];
                    }
                    else
                    {
                        StreamInfo[] resOld = res;
                        res = new StreamInfo[resOld.Length + dirs.Length];
                        Array.Copy(resOld, 0, res, res.Length - resOld.Length, resOld.Length);
                    }
                    for (int i = 0; i < dirs.Length; i++)
                    {
                        long len = 0;
                        DateTime lastMod = DateTime.MinValue;
                        try
                        {
                            //DirectoryInfo f = new DirectoryInfo(dirs[i]);
                            //lastMod = f.LastWriteTime;
                            lastMod = Directory.GetLastWriteTime(dirs[i]);
                        }
                        catch
                        {
                            // ignore?
                        }
                        if (dirs[i].Length == 0)
                        {
                            dirs[i] = ".\\";
                        }
                        else
                        {
                            if (dirs[i][dirs[i].Length - 1] != '\\' &&
                                dirs[i][dirs[i].Length - 1] != '/')
                            {
                                dirs[i] = dirs[i] + "\\";
                            }
                        }
                        res[i] = new StreamInfo(dirs[i], len, lastMod);
                    }

                    return res;
                }
                catch
                {
                    return new StreamInfo[0];
                }
            }

            if ((path.StartsWith("\\\\") || path.StartsWith("//")) &&
                !Directory.Exists(Path.GetPathRoot(path)))
            {
                return new StreamInfo[0];
            }

            // check for compressed forms:
            string pathBase = pathLower.Substring(0, path.Length - 1);
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionExtensions[i];
                if (pathBase.EndsWith(ext) && File.Exists(pathBase))
                {
                    StreamInfo[] res = Z7zDecodeStream.DirectoryEntriesInfo(pathBase, null, allowFile, allowDirectory);
                    if (res.Length == 0 && !Z7zDecodeStream.Exists7z && ext == ".rar")
                    {
                        res = RarDecodeStream.DirectoryEntriesInfo(pathBase, null, allowFile, allowDirectory);
                    }
                    return res;
                }
                else
                {
                    string pathBaseExt = pathBase + ext;
                    if (File.Exists(pathBaseExt))
                    {
                        StreamInfo[] res = Z7zDecodeStream.DirectoryEntriesInfo(pathBaseExt, null, allowFile, allowDirectory);
                        if (res.Length == 0 && !Z7zDecodeStream.Exists7z && ext == ".rar")
                        {
                            res = RarDecodeStream.DirectoryEntriesInfo(pathBaseExt, null, allowFile, allowDirectory);
                        }
                        return res;
                    }
                }
            }

            // check for compressed archive as directory segment:
            // check for compressed archives in path:
            // only one path segment is allowed to be an archive...
            // normalize path:
            string zfileName = path.Replace('/', '\\');
            bool isUnc = zfileName.StartsWith("\\\\");
            while (zfileName.IndexOf("\\\\") >= 0)
            {
                zfileName = zfileName.Replace("\\\\", "\\");
            }
            if (isUnc)
                zfileName = "\\" + zfileName;
            string znameLower = zfileName.ToLower();

            string archPath = null;
            string inArch = null;
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionExtensions[i];
                int seg = znameLower.IndexOf(ext + "\\");
                if (seg > 0)
                {
                    archPath = zfileName.Substring(0, seg + ext.Length);
                    if (File.Exists(archPath))
                    {
                        inArch = zfileName.Substring(seg + ext.Length).Trim('/', '\\');
                        break;
                    }
                    archPath = null;
                }
            }
            if (archPath == null)
            {
                // add in extension to each segment...
                string[] segs = zfileName.Split('\\');
                for (int i = 0; i < segs.Length; i++)
                {
                    if (segs[i].Length == 0)
                        continue;
                    string partial = string.Join("\\", segs, 0, i + 1);
                    if (partial.Length == 2 && partial[1] == ':')
                        continue;
                    if (Directory.Exists(partial))
                        continue;
                    for (int c = 0; c < ZStreamIn.decompressionExtensions.Length; c++)
                    {
                        string ext = ZStreamIn.decompressionExtensions[c];
                        if (File.Exists(partial + ext))
                        {
                            archPath = partial + ext;
                            inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1).Trim('/', '\\');
                            break;
                        }
                    }
                    // quit when parent will not exist
                    break;
                }
            }
            if (archPath != null)
            {
                //Console.WriteLine(archPath + " :: " + inArch);
                // check for path in archive.
                inArch = inArch.Trim('\\');
                if (inArch.Length == 0)
                    return new StreamInfo[0];
                StreamInfo[] res = Z7zDecodeStream.DirectoryEntriesInfo(archPath, inArch + "\\*", allowFile, allowDirectory);
                if (res.Length == 0 && !Z7zDecodeStream.Exists7z && Path.GetExtension(archPath).ToLower() == ".rar")
                {
                    res = RarDecodeStream.DirectoryEntriesInfo(archPath, inArch + "\\*", allowFile, allowDirectory);
                }
                return res;
            }

            return new StreamInfo[0];
        }

        private static StreamInfo[] CockpitDirectoryEntriesInfo(string path, bool allowFile, bool allowDirectory)
        {
            List<StreamInfo> res = new List<StreamInfo>();
            string htmlDir;
            try
            {
                if (!path.EndsWith("/") && !path.EndsWith("\\"))
                {
                    path = path + '\\';
                }
                htmlDir = ReadFile(path);
            }
            catch
            {
                // throw exception?
                return new StreamInfo[0];
            }
            if (htmlDir == null || htmlDir.Length == 0)
                return new StreamInfo[0];
            htmlDir = htmlDir.Replace("<br>", "<br />");

            Hashtable filenames = new Hashtable();

#if !DISABLE_XML
            System.Xml.XmlTextReader xmlReader = null;
            try
            {
                // apparently, not IDisposable in .NET 1.1:
                //using (System.Xml.XmlTextReader xmlReader = new System.Xml.XmlTextReader(new StringReader(htmlDir)))
                xmlReader = new System.Xml.XmlTextReader(new StringReader(htmlDir));
                {
                    string filename = null;
                    long length = -1;
                    DateTime lastModified = DateTime.MinValue;
                    bool isDirectory = false;

                    // Parse node by node
                    while (xmlReader.Read())
                    {
                        if (xmlReader.IsStartElement("tr"))
                        {
                            filename = null;
                            length = -1;
                            lastModified = DateTime.MinValue;
                            isDirectory = false;
                        }
                        else if (xmlReader.IsStartElement("td"))
                        {
                            xmlReader.Read();
                            if (xmlReader.IsStartElement("a"))
                            {
                                // Store the anchor text value
                                xmlReader.Read();
                                filename = xmlReader.Value;
                            }
                            else
                            {
                                try
                                {
                                    string val = xmlReader.Value;
                                    if (string.Compare(val.Trim(), "&nbsp;", true) == 0)
                                    {
                                        isDirectory = true;
                                    }
                                    else
                                    {
                                        if (val.Length != 0)
                                        {
                                            if (char.IsDigit(val[0]))
                                            {
                                                // Attempt to convert to a long
                                                length = Convert.ToInt64(val);
                                            }
                                            else
                                            {
                                                // Attempt to convert to a timestamp
                                                lastModified = Convert.ToDateTime(val);
                                            }
                                        }
                                    }
                                }
                                catch
                                {
                                    // Skip this item
                                }
                            }
                        }
                        // When all the data has been captured, record the entry
                        if (filename != null)
                        {
                            if (length >= 0 && lastModified != DateTime.MinValue)
                            {
                                if (allowFile)
                                {
                                    string fullname = PathCombine(path, filename);
                                    res.Add(new StreamInfo(fullname, length, lastModified));
                                }
                                filename = null;
                                length = -1;
                                lastModified = DateTime.MinValue;
                                isDirectory = false;
                            }
                            else if (isDirectory)
                            {
                                if (allowDirectory)
                                {
                                    string line = filename;
                                    line = line.Replace('\t', ' ');
                                    int oldLen = -1;
                                    while (line.Length != oldLen)
                                    {
                                        oldLen = line.Length;
                                        line = line.Replace("  ", " ");
                                    }
                                    line = line.Trim();

                                    if (line != "Parent directory\\")
                                    {
                                        string fullname = PathCombine(path, filename);
                                        if (!fullname.EndsWith("\\") && !fullname.EndsWith("/"))
                                        {
                                            fullname = fullname + "\\";
                                        }
                                        res.Add(new StreamInfo(fullname, 0, DateTime.MinValue));
                                    }
                                }
                                filename = null;
                                length = -1;
                                lastModified = DateTime.MinValue;
                                isDirectory = false;
                            }
                        }
                    }
                }
            }
            catch
            {
                // throw exception?
                return new StreamInfo[0];
            }
            finally
            {
                try
                {
                    if (xmlReader != null)
                        xmlReader.Close();
                }
                catch
                {
                    // ignore
                }
            }
#endif
            return res.ToArray();
        }

        #endregion

        #region File Operations

        /// <summary>
        /// Determine if a file exists, including in forms with compression extensions, HTTP URLs, and
        /// Cosmos streams.
        /// </summary>
        /// <param name="fileName">the original filename</param>
        /// <returns>true if a file with a name of fileName or its compressed variations exists, false otherwise</returns>
        /// <remarks>
        /// This is useful for detecting whether a file can be opened by
        /// <c>ZStreamIn.Open()</c>, since it will look for compressed forms and other variations.
        /// </remarks>
        public static bool FileExists(string fileName)
        {
            if (fileName == null || fileName.Length == 0)
                return false;
            string nameLower = fileName.ToLower();

            // check for special names:
            if (ZStreamIn.IsNullStream(fileName))
            {
                return true;
            }
            if (ZStreamIn.IsConsoleStream(fileName))
            {
                return true;
            }

            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(fileName))
            {
                return ClipboardReadStream.FileExists();
            }

            // check for cosmos:
            if (nameLower.StartsWith("cosmos:"))
            {
                // no compression support, anyway...
                fileName = fileName.TrimEnd('$');
                return Cosmos.FileExists(fileName);
            }

            // check for Cockpit:
            if (nameLower.StartsWith("cockpit:"))
            {
                // what is right here?
                string parent = PathParent(fileName);
                if (parent == null)
                {
                    // assume it exists??
                    return true;
                }
                string name = GetName(fileName);
                string[] dirs = DirectoryEntries(parent, true, false);
                for (int i = 0; i < dirs.Length; i++)
                {
                    string dir = GetName(dirs[i]).Trim(pathSeparators);
                    if (string.Compare(name, dir, true) == 0)
                    {
                        return true;
                    }
                }
                return false;
            }

            // check for Multistream:
            if (nameLower.StartsWith("multi:"))
            {
                fileName = fileName.Substring("multi:".Length);
                return FileExists(fileName);
            }
            if (nameLower.StartsWith("filelist:"))
            {
                fileName = fileName.Substring("filelist:".Length);
                return FileExists(fileName);
            }

            // check for HTTP:
            if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
            {
                AzureStorageIO azureStorage = new AzureStorageIO();
                if (azureStorage.BlockBlobExistsByUri(fileName))
                {
                    return true;
                }
                return HttpStream.Exists(fileName);
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return SqlTextReader.Exists(fileName);
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(fileName))
            {
                return InternalStoreUtility.Exists(fileName);
            }
            // remove trailing "$"
            if (fileName[fileName.Length - 1] == '$')
            {
                fileName = fileName.Substring(0, fileName.Length - 1);
            }

            if (File.Exists(fileName))
                return true;

            // check for compressed forms:
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                if (File.Exists(fileName + ZStreamIn.decompressionExtensions[i]))
                {
                    return true;
                }
            }

            // check for named stream:
            // should this be based on file existance, first?
            int cIndex = fileName.LastIndexOf(':');
            if (cIndex > 0)
            {
                if (File.Exists(fileName.Substring(0, cIndex)))
                {
                    if (cIndex > 1 ||
                        fileName.IndexOfAny(pathSeparators, 2) < 0)
                    {
                        // named:
                        return (Array.IndexOf(IOUtil.GetNamedStreams(fileName.Substring(0, cIndex)),
                            fileName.Substring(cIndex + 1)) >= 0);
                    }
                }
            }

            // check for compressed archive as directory segment:
            // check for compressed archives in path:
            // only one path segment is allowed to be an archive...
            // normalize path:
            fileName = fileName.Replace('/', '\\');
            bool isUnc = fileName.StartsWith("\\\\");
            while (fileName.IndexOf("\\\\") >= 0)
            {
                fileName = fileName.Replace("\\\\", "\\");
            }
            if (isUnc)
                fileName = "\\" + fileName;
            nameLower = fileName.ToLower();

            string archPath = null;
            string inArch = null;
            for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
            {
                string ext = ZStreamIn.decompressionExtensions[i];
                int seg = nameLower.IndexOf(ext + "\\");
                if (seg > 0)
                {
                    archPath = fileName.Substring(0, seg + ext.Length);
                    if (File.Exists(archPath))
                    {
                        inArch = fileName.Substring(seg + ext.Length).Trim('/', '\\');
                        break;
                    }
                    archPath = null;
                }
            }
            if (archPath == null)
            {
                // add in extension to each segment...
                string[] segs = fileName.Split('\\');
                for (int i = 0; i < segs.Length; i++)
                {
                    if (segs[i].Length == 0)
                        continue;
                    string partial = string.Join("\\", segs, 0, i + 1);
                    if (partial.Length == 2 && partial[1] == ':')
                        continue;
                    if (Directory.Exists(partial))
                        continue;
                    for (int c = 0; c < ZStreamIn.decompressionExtensions.Length; c++)
                    {
                        string ext = ZStreamIn.decompressionExtensions[c];
                        if (File.Exists(partial + ext))
                        {
                            archPath = partial + ext;
                            inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1).Trim('/', '\\');
                            break;
                        }
                    }
                    // quit when parent will not exist
                    break;
                }
            }
            if (archPath != null)
            {
                //Console.WriteLine(archPath + " :: " + inArch);
                // check for path in archive.
                inArch = inArch.Trim('\\');
                if (inArch.Length == 0)
                    return false;
                if (Z7zDecodeStream.Exists(archPath, inArch))
                    return true;
                if (!Z7zDecodeStream.Exists7z && Path.GetExtension(archPath).ToLower() == ".rar")
                {
                    if (RarDecodeStream.Exists(archPath, inArch))
                        return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Find the full path for a command, if it exists in the environment path.
        /// </summary>
        /// <param name="cmd">the command to look for</param>
        /// <returns>the full path, or null if it is not found</returns>
        /// <remarks>
        /// <p>
        /// This will always look in the current PATH directories, as well as in
        /// the current directory. The current directory takes precedence.
        /// </p>
        /// <p>
        /// The command may have an extension included; it will also be tried
        /// with ".exe", ".bat", and ".cmd", in that order, in each path
        /// directory. This gives similar results to the standard command line.
        /// </p>
        /// </remarks>
        public static string FindInPath(string cmd)
        {
            return FindInPath(cmd, false);
        }
        /// <summary>
        /// Find the full path for a command, if it exists in the environment path.
        /// </summary>
        /// <param name="cmd">the command to look for</param>
        /// <param name="includeAssemblyDirectory">
        /// if true, also check the directory of the calling assembly.
        /// </param>
        /// <returns>the full path, or null if it is not found</returns>
        /// <remarks>
        /// <p>
        /// This will always look in the current PATH directories, as well as in
        /// the current directory. The current directory takes precedence.
        /// </p>
        /// <p>
        /// If <paramref name="includeAssemblyDirectory"/> is true, it takes
        /// precedence over any other directory.
        /// </p>
        /// <p>
        /// The command may have an extension included; it will also be tried
        /// with ".exe", ".cmd", and ".bat", in that order, in each path
        /// directory. This gives similar results to the standard command line.
        /// </p>
        /// </remarks>
        public static string FindInPath(string cmd, bool includeAssemblyDirectory)
        {
            if (cmd == null || cmd.Length == 0)
                return null;
            // caller's directory:
            if (includeAssemblyDirectory)
            {
                string dir = Path.GetDirectoryName(System.Reflection.Assembly.GetCallingAssembly().Location);
                string path = Path.Combine(dir, cmd);
                if (File.Exists(path))
                    return path;
                string extPath;
                extPath = path + ".exe";
                if (File.Exists(extPath))
                    return extPath;
                extPath = path + ".cmd";
                if (File.Exists(extPath))
                    return extPath;
                extPath = path + ".bat";
                if (File.Exists(extPath))
                    return extPath;
            }
            // current directory:
            {
                string dir = Environment.CurrentDirectory;
                string path = Path.Combine(dir, cmd);
                if (File.Exists(path))
                    return path;
                string extPath;
                extPath = path + ".exe";
                if (File.Exists(extPath))
                    return extPath;
                extPath = path + ".cmd";
                if (File.Exists(extPath))
                    return extPath;
                extPath = path + ".bat";
                if (File.Exists(extPath))
                    return extPath;
            }
            // PATH:
            {
                string[] paths = Environment.GetEnvironmentVariable("PATH").Trim(';').Split(';');
                for (int i = 0; i < paths.Length; i++)
                {
                    string dir = paths[i].Trim();
                    if (dir.Length == 0)
                        continue;
                    string path = Path.Combine(dir, cmd);
                    if (File.Exists(path))
                        return path;
                    string extPath;
                    extPath = path + ".exe";
                    if (File.Exists(extPath))
                        return extPath;
                    extPath = path + ".cmd";
                    if (File.Exists(extPath))
                        return extPath;
                    extPath = path + ".bat";
                    if (File.Exists(extPath))
                        return extPath;
                }
            }
            return null;
        }

        /// <summary>
        /// Delete a stream or directory, if it exists.
        /// </summary>
        /// <param name="fileName">the stream or directory to delete</param>
        /// <remarks>
        /// <p>
        /// This will silently do nothing if the file already does not exist.
        /// </p>
        /// <p>
        /// If used on a directory, the directory must already be empty.
        /// </p>
        /// <p>
        /// Note that this will not delete all compressed versions of the given filename,
        /// without explicitly using an overload of this method.
        /// </p>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="IOException"> An I/O error has occurred, or the file cannot be deleted.</exception>
        /// <exception cref="NotSupportedException">The stream does not support deleting.</exception>
        public static void Delete(string fileName)
        {
            Delete(fileName, false);
        }
        /// <summary>
        /// Delete a file or stream, if it exists.
        /// </summary>
        /// <param name="fileName">the file or stream to delete</param>
        /// <param name="recursive">if true, delete all files and subdirectories if
        /// fileName is a directory; otherwise, fileName must be empty if it is a directory</param>
        /// <remarks>
        /// <p>
        /// This will silently do nothing if the file already does not exist.
        /// </p>
        /// <p>
        /// When including compressed files, this respects the setting of <see cref="ZStreamIn.FallbackExtension"/>.
        /// </p>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="IOException"> An I/O error has occurred, or the file cannot be deleted.</exception>
        /// <exception cref="NotSupportedException">The stream does not support deleting.</exception>
        public static void Delete(string fileName, bool recursive)
        {
            Delete(fileName, recursive, false);
        }
        /// <summary>
        /// Delete a file or stream, if it exists.
        /// </summary>
        /// <param name="fileName">the file or stream to delete</param>
        /// <param name="recursive">if true, delete all files and subdirectories if
        /// fileName is a directory; otherwise, fileName must be empty if it is a directory</param>
        /// <param name="includeCompressedVersions">if true, also delete any compressed versions of the
        /// given name; otherwise, only delete the given name</param>
        /// <remarks>
        /// <p>
        /// This will silently do nothing if the file already does not exist.
        /// </p>
        /// <p>
        /// When including compressed files, this respects the setting of <see cref="ZStreamIn.FallbackExtension"/>.
        /// </p>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="IOException"> An I/O error has occurred, or the file cannot be deleted.</exception>
        /// <exception cref="NotSupportedException">The stream does not support deleting.</exception>
        public static void Delete(string fileName, bool recursive, bool includeCompressedVersions)
        {
            Contracts.CheckNonEmpty(fileName, nameof(fileName));

            // check for special names:
            string fileNameLower = fileName.ToLower();
            if (ZStreamIn.IsNullStream(fileName))
            {
                // just succeed...
                return;
            }
            if (ZStreamIn.IsConsoleStream(fileName))
            {
                throw new NotSupportedException("Console streams cannot be deleted.");
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(fileName))
            {
                throw new NotSupportedException("Clipboard streams cannot be deleted.");
            }

            // check for Cosmos:
            if (fileNameLower.StartsWith("cosmos:"))
            {
                Cosmos.Delete(fileName, recursive);
                // don't look for compressed versions?
                return;
            }

            // check for Cockpit:
            if (fileNameLower.StartsWith("cockpit:"))
            {
                throw new NotSupportedException("Cockpit streams cannot be deleted.");
            }

            // check for Multistream:
            if (fileNameLower.StartsWith("multi:"))
            {
                fileName = fileName.Substring("multi:".Length);
                Delete(fileName);
            }
            if (fileNameLower.StartsWith("filelist:"))
            {
                fileName = fileName.Substring("filelist:".Length);
                Delete(fileName);
            }

            //REVIEW: We can actually delete a blob, however, it is blocked by this code
            // check for HTTP:
            if (fileNameLower.StartsWith("http:") || fileNameLower.StartsWith("https:"))
            {
                throw new NotSupportedException("HTTP streams or blobs cannot be deleted.");
            }

            // check for SQL:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                throw new NotSupportedException("Cannot delete a SQL table.");
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(fileName))
            {
                throw new NotSupportedException("Cannot delete a InternalStore.");
            }

            // remove trailing "$"
            if (fileName[fileName.Length - 1] == '$')
            {
                fileName = fileName.Substring(0, fileName.Length - 1);
            }

            // check for named stream:
            // should this be based on file existance, first?
            int cIndex = fileName.LastIndexOf(':');
            if (cIndex > 0)
            {
                if (File.Exists(fileName.Substring(0, cIndex)))
                {
                    if (cIndex > 1 ||
                        fileName.IndexOfAny(pathSeparators, 2) < 0)
                    {
                        // named:
                        NamedStream.Delete(fileName);
                        return;
                    }
                }
            }

            bool deleted = false;
            // plain file?:
            // (handle the mess of exceptions that are normally generated)
            try
            {
                if (Directory.Exists(fileName))
                {
                    Directory.Delete(fileName, recursive);
                    deleted = true;
                }
                else if (File.Exists(fileName))
                {
                    File.Delete(fileName);
                    deleted = true;
                }
            }
            catch (UnauthorizedAccessException ex)
            {
                throw new IOException("Cannot delete: " + fileName, ex);
            }
            catch (PathTooLongException ex)
            {
                throw new ArgumentException("Path is too long: " + fileName, "fileName", ex);
            }
            catch (DirectoryNotFoundException ex)
            {
                throw new IOException("Directory does not exist: " + fileName, ex);
            }
            catch (NotSupportedException ex)
            {
                throw new IOException("Cannot delete: " + fileName, ex);
            }

            if (!deleted)
            {
                // check for compressed archive as directory segment:
                // only one path segment is allowed to be an archive...
                // normalize path:
                //if (fileName[fileName.Length - 1] != '\\')  fileName = fileName + "\\";
                string zfileName = fileName.Replace('/', '\\');
                bool isUnc = zfileName.StartsWith("\\\\");
                while (zfileName.IndexOf("\\\\") >= 0)
                {
                    zfileName = zfileName.Replace("\\\\", "\\");
                }
                if (isUnc)
                    zfileName = "\\" + zfileName;
                string zfileNameLower = zfileName.ToLower();

                string archPath = null;
                string inArch = null;
                // should this be archives only?
                for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
                {
                    string ext = ZStreamIn.decompressionExtensions[i];
                    int seg = zfileNameLower.IndexOf(ext + "\\");
                    if (seg > 0)
                    {
                        archPath = zfileName.Substring(0, seg + ext.Length);
                        if (File.Exists(archPath))
                        {
                            inArch = zfileName.Substring(seg + ext.Length).Trim('\\');
                            break;
                        }
                        archPath = null;
                    }
                }
                if (archPath == null)
                {
                    // add in extension to each segment...
                    string[] segs = zfileName.Split('\\');
                    for (int i = 0; i < segs.Length; i++)
                    {
                        if (segs[i].Length == 0)
                            continue;
                        string partial = string.Join("\\", segs, 0, i + 1);
                        if (partial.Length == 2 && partial[1] == ':')
                            continue;
                        if (Directory.Exists(partial))
                            continue;
                        // should this be archives only?
                        for (int c = 0; c < ZStreamIn.decompressionExtensions.Length; c++)
                        {
                            string ext = ZStreamIn.decompressionExtensions[c];
                            if (File.Exists(partial + ext))
                            {
                                archPath = partial + ext;
                                inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1).Trim('/', '\\');
                                break;
                            }
                        }
                        // quit when parent will not exist
                        break;
                    }
                }
                if (archPath != null)
                {
                    // found compressed archive for path segment
                    // check for path in archive.
                    inArch = inArch.Trim('\\');
                    if (inArch.Length != 0)
                    {
                        // slow:
                        if (Z7zDecodeStream.Exists(archPath, inArch))
                        {
                            // allow exception to throw if it occurs:
                            if (recursive)
                            {
                                Z7zEncodeStream.Delete(archPath, inArch);
                            }
                            else
                            {
                                if (Z7zDecodeStream.Exists(archPath, inArch + "\\*"))
                                {
                                    throw new IOException("Cannot delete non-empty directory in archive when not recursive: " +
                                        fileName);
                                }
                                Z7zEncodeStream.Delete(archPath, inArch);
                            }
                        }
                        // skip compressed versions operation!
                        return;
                    }
                }
            }

            // check for compressed versions, if needed:
            if (includeCompressedVersions)
            {
                if (ZStreamIn.FallbackExtension != null)
                {
                    if (ZStreamIn.FallbackExtension.Length != 0)
                    {
                        if (File.Exists(fileName + ZStreamIn.FallbackExtension))
                        {
                            Delete(fileName + ZStreamIn.FallbackExtension, false);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
                        {
                            if (File.Exists(fileName + ZStreamIn.decompressionExtensions[i]))
                            {
                                Delete(fileName + ZStreamIn.decompressionExtensions[i], false);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Get the length of the stream in bytes, or -1 if it cannot be obtained.
        /// </summary>
        /// <param name="path">the path or stream name</param>
        /// <returns>the length of the stream in bytes, or -1 if it cannot be obtained</returns>
        public static long GetLength(string path)
        {
            // this could be more efficient! ***
            // force buffered?
            try
            {
                using (Stream str = ZStreamIn.OpenBuffered(path))
                {
                    return str.Length;
                }
            }
            catch
            {
                return -1;
            }
        }

        /// <summary>
        /// Set the length of a file, truncating or padding as needed.
        /// </summary>
        /// <param name="fileName">the file to alter</param>
        /// <param name="length">the desired length, in bytes</param>
        /// <remarks>
        /// Note that this only operates on normal files, and cannot be successfully used on compressed
        /// files or special streams.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        /// <exception cref="NotSupportedException">The stream does not support both writing and seeking.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Attempted to set the value parameter to less than 0.</exception>
        public static void ResizeFile(string fileName, long length)
        {
            using (FileStream pad = new FileStream(fileName, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite))
            {
                pad.SetLength(length);
            }
        }

        /// <summary>
        /// Get the named streams present in the given file.
        /// </summary>
        /// <param name="fileName">the file to inspect</param>
        /// <returns>the list of named streams available for the file, excluding the default stream</returns>
        /// <remarks>
        /// <p>
        /// Named streams only exist on NTFS file systems, generally, and might not be
        /// preserved as files are moved around in various forms. Most applications can
        /// only access the default stream, but command-line redirection will write and read
        /// from named streams when specified. Named streams are specified with the syntax
        /// "filename:streamname". This syntax will also work when opening files with
        /// <see cref="ZStreamIn.Open(string)"/>, <see cref="ZStreamOut.Open(string)"/>,
        /// <see cref="ZStreamReader.Open(string)"/>, and <see cref="ZStreamWriter.Open(string)"/>.
        /// </p>
        /// <p>
        /// Note that this will not automatically fallback to compressed variations of
        /// the given filename.
        /// </p>
        /// </remarks>
        public static string[] GetNamedStreams(string fileName)
        {
            return NamedStream.GetNamedStreams(fileName);
        }

        #endregion

        #region Reading and Writing

        /// <summary>
        /// Copy one file or directory to another location.
        /// </summary>
        /// <param name="source">the file or stream to copy (potentially with a wildcard pattern)</param>
        /// <param name="destination">the destination, either as a directory or a file</param>
        /// <remarks>
        /// <p>
        /// It is best to put a trailing slash on directories in order to ensure that they are not
        /// treated as files.
        /// </p>
        /// <p>
        /// If a wildcard pattern is used, the files will all be copied into the destination. A
        /// <see cref="FileNotFoundException"/> will be thrown if no files match.
        /// If there are multiple matches, the destination should be a directory or a continuous stream
        /// (such as the console or the null stream), since the effect will be as if each copy overwrote
        /// the last one, otherwise.
        /// </p>
        /// </remarks>
        /// <exception cref="ArgumentNullException">source or destination is null.</exception>
        /// <exception cref="ArgumentException">source or destination is invalid.</exception>
        /// <exception cref="FileNotFoundException">source cannot be found.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        /// <exception cref="NotSupportedException">The destination does not support writing.</exception>
        public static void Copy(string source, string destination)
        {
            Copy(source, destination, false);
        }

        /// <summary>
        /// Copy one file or directory to another location, as lines of text, with any implicit conversion.
        /// </summary>
        /// <param name="source">the file or stream to copy (potentially with a wildcard pattern)</param>
        /// <param name="destination">the destination, either as a directory or a file</param>
        /// <remarks>
        /// <p>
        /// It is best to put a trailing slash on directories in order to ensure that they are not
        /// treated as files.
        /// </p>
        /// <p>
        /// If a wildcard pattern is used, the files will all be copied into the destination. A
        /// <see cref="FileNotFoundException"/> will be thrown if no files match.
        /// If there are multiple matches, the destination should be a directory or a continuous stream
        /// (such as the console or the null stream), since the effect will be as if each copy overwrote
        /// the last one, otherwise.
        /// </p>
        /// </remarks>
        /// <exception cref="ArgumentNullException">source or destination is null.</exception>
        /// <exception cref="ArgumentException">source or destination is invalid.</exception>
        /// <exception cref="FileNotFoundException">source cannot be found.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        /// <exception cref="NotSupportedException">The destination does not support writing.</exception>
        public static void CopyLines(string source, string destination)
        {
            Copy(source, destination, true);
        }

        private static void Copy(string source, string destination, bool translateText)
        {
            Contracts.CheckNonEmpty(source, nameof(source));
            Contracts.CheckNonEmpty(destination, nameof(destination));

            string[] files = ExpandWildcards(source);
            if (files.Length == 0)
                throw new FileNotFoundException("No files match the pattern: " + source);

            // this should be better encapsulated...

            if (files.Length > 1)
            {
                for (int i = 0; i < files.Length; i++)
                {
                    Copy(files[i], destination);
                }
                return;
            }
            source = files[0];

            string fullDestination = null;

            if (FileExists(destination))
            {
                fullDestination = destination;
            }
            else if (DirectoryExists(destination))
            {
                fullDestination = destination.TrimEnd(pathSeparators) + "/" + GetFileName(source.TrimEnd(pathSeparators));
            }
            else if (DirectoryExists(Path.GetDirectoryName(destination)))
            {
                // is that correct???
                fullDestination = destination;
            }
            else
            {
                // check for special names:
                string nameLower = destination.ToLower();

                if (ZStreamIn.IsNullStream(destination))
                {
                    fullDestination = destination;
                }
                else if (ZStreamIn.IsConsoleStream(destination))
                {
                    fullDestination = destination;
                }
                else if (ClipboardReadStream.IsClipboardStream(destination))
                {
                    // clipboard
                    fullDestination = destination;
                }
                else if (nameLower.StartsWith("cosmos:"))
                {
                    // cosmos
                    if (destination[destination.Length - 1] == '/' || destination[destination.Length - 1] == '\\')
                    {
                        fullDestination = destination + GetFileName(source.TrimEnd(pathSeparators));
                    }
                    else
                    {
                        fullDestination = destination;
                    }
                }
                else if (nameLower.StartsWith("cockpit:"))
                {
                    // Cockpit
                    throw new NotSupportedException("Cockpit streams cannot be written to.");
                }
                else if (nameLower.StartsWith("http://") || nameLower.StartsWith("https://"))
                {
                    // HTTP
                    throw new NotSupportedException("HTTP streams cannot be written to.");
                }
                else if (InternalStoreUtility.IsInternalStore(nameLower))
                {
                    // InternalStore:
                    throw new NotSupportedException("Cannot copy to a InternalStore.");
                }
                else if (SqlTextReader.IsSqlTextReader(nameLower))
                {
                    // SQL:
                    //throw new NotSupportedException("Cannot copy to a SQL table.");
                    fullDestination = destination;
                }
                else
                {
                    // default?
                    fullDestination = destination;
                }
            }
            // what about Multistream? ***

            // this could be optimized, of course...
            // should we special-case normal file copy to File.Copy()? ***
            if (PathsEqual(source, fullDestination))
                return;
            if (source[source.Length - 1] == '/' || source[source.Length - 1] == '\\' || DirectoryExists(source))
            {
                // recursive copy?
                // not for compressed files...
                bool copyCompressed = false;
                try
                {
                    if (File.Exists(source.TrimEnd(pathSeparators)))
                    {
                        source = source.TrimEnd(pathSeparators) + "$";
                        fullDestination = fullDestination.TrimEnd(pathSeparators) + "$";
                        copyCompressed = true;
                    }
                }
                catch
                {
                    // ignore
                }
                if (!copyCompressed)
                {
                    // we should really try and avoid infinite loops...
                    if (PathAncestor(source, fullDestination))
                    {
                        throw new ArgumentException("Cannot copy to a subdirectory of the source", "destination");
                    }
                    CreateDirectory(fullDestination.TrimEnd(pathSeparators));
                    foreach (string entry in DirectoryEntries(source))
                    {
                        Copy(entry, fullDestination + "/");
                    }
                    return;
                }
            }

            if (translateText)
            {
                // this could be optimized, of course...
                using (StreamReader reader = ZStreamReader.Open(source))
                {
                    using (StreamWriter writer = ZStreamWriter.Open(fullDestination))
                    {
                        // preallocate:
                        try
                        {
                            long length = reader.BaseStream.Length;
                            if (length > 0)
                            {
                                if (writer.BaseStream is LowFragmentationStream)
                                {
                                    ((LowFragmentationStream)writer.BaseStream).Reserve(length);
                                }
                            }
                        }
                        catch
                        {
                        }

                        for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
                        {
                            writer.WriteLine(line);
                        }
                    }
                }
            }
            else
            {
                using (Stream fileIn = ZStreamIn.Open(source))
                {
                    using (Stream fileOut = ZStreamOut.Open(fullDestination))
                    {
                        // preallocate:
                        // bad for compression... we really should give the length there
                        try
                        {
                            long length = fileIn.Length;
                            if (length > 0)
                            {
                                if (fileOut is FileStream || fileOut is LowFragmentationStream)
                                {
                                    fileOut.SetLength(length);
                                }
                            }
                        }
                        catch
                        {
                        }

                        byte[] buffer = new byte[256 * 1024];
                        for (int count = fileIn.Read(buffer, 0, buffer.Length); count > 0; count = fileIn.Read(buffer, 0, buffer.Length))
                        {
                            fileOut.Write(buffer, 0, count);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Copy one Stream to another.
        /// </summary>
        /// <param name="source">the Stream to copy</param>
        /// <param name="destination">the destination Stream</param>
        /// <remarks>
        /// The source and destination are both left open.
        /// </remarks>
        /// <exception cref="ArgumentNullException">source or destination is null.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        public static void Copy(Stream source, Stream destination)
        {
            if (source == null)
                throw new ArgumentException("source cannot be null", "source");
            if (destination == null)
                throw new ArgumentException("destination cannot be null", "destination");
            try
            {
                if (source is UnbufferedStream)
                {
                    UnbufferedStream us = (UnbufferedStream)source;
                    byte[] buffer;
                    for (int c = us.Read(out buffer); c > 0; c = us.Read(out buffer))
                    {
                        destination.Write(buffer, 0, c);
                    }
                }
                else
                {
                    //Console.Error.WriteLine("writing... " + source.GetType().Name + " -> " + destination.GetType().Name);
                    byte[] buffer = new byte[64 * 1024];
                    for (int c = source.Read(buffer, 0, buffer.Length); c > 0; c = source.Read(buffer, 0, buffer.Length))
                    {
                        //Console.Error.WriteLine("writing " + c + "...");
                        destination.Write(buffer, 0, c);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new IOException("Could not copy source to destination", ex);
            }
        }

        /// <summary>
        /// Copy a TextReader to a TextWriter, as lines of text, with any implicit conversion.
        /// </summary>
        /// <param name="source">the TextReader to copy</param>
        /// <param name="destination">the destination TextWriter</param>
        /// <remarks>
        /// The source and destination are both left open.
        /// </remarks>
        /// <exception cref="ArgumentNullException">source or destination is null.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        public static void CopyLines(TextReader source, TextWriter destination)
        {
            if (source == null)
                throw new ArgumentException("source cannot be null", "source");
            if (destination == null)
                throw new ArgumentException("destination cannot be null", "destination");
            try
            {
                for (string line = source.ReadLine(); line != null; line = source.ReadLine())
                {
                    destination.WriteLine(line);
                }
            }
            catch (Exception ex)
            {
                throw new IOException("Could not copy source to destination", ex);
            }
        }

        /// <summary>
        /// Create a stream that is the concatenation of a set of streams.
        /// </summary>
        /// <param name="destination">the name of the stream to create</param>
        /// <param name="sources">the names of the streams to concatenate</param>
        /// <remarks>
        /// <p>
        /// This will overwrite the destination.
        /// </p>
        /// <p>
        /// For Cosmos files, this maps to a <see cref="Cosmos.Concatenate"/>.
        /// </p>
        /// <p>
        /// Wildcard patterns are allowed.
        /// </p>
        /// </remarks>
        /// <exception cref="IOException">The join could not be completed.</exception>
        /// <exception cref="ArgumentException">The destination or some sources are not valid stream names.</exception>
        public static void Concatenate(string destination, params string[] sources)
        {
            if (destination == null || destination.Length == 0)
                throw new ArgumentException("destination cannot be empty", "destination");
            if (sources == null)
                sources = new string[0];
            bool allCosmos = (string.Compare(destination, 0, "cosmos://", 0, "cosmos://".Length, true) == 0);
            if (allCosmos)
            {
                for (int i = 0; i < sources.Length; i++)
                {
                    if (sources[i] == null || string.Compare(sources[i], 0, "cosmos://", 0, "cosmos://".Length, true) != 0)
                    {
                        allCosmos = false;
                        break;
                    }
                }
            }
            if (allCosmos)
            {
                Cosmos.Concatenate(destination, sources);
                return;
            }
            List<string> expanded = null;
            for (int i = 0; i < sources.Length; i++)
            {
                if (sources[i] == null || sources[i].Length == 0)
                    throw new ArgumentException("sources[" + i + "] cannot be empty", "sources");
                if (sources[i].IndexOf('*') >= 0 || sources[i].IndexOf('?') >= 0 ||
                    sources[i].IndexOf("...") >= 0)
                {
                    if (expanded == null)
                    {
                        expanded = new List<string>();
                        for (int j = 0; j < i; j++)
                        {
                            expanded.Add(sources[j]);
                        }
                    }
                    expanded.AddRange(IOUtil.ExpandWildcards(sources[i]));
                }
                else
                {
                    if (expanded != null)
                        expanded.Add(sources[i]);
                }
            }
            if (expanded != null)
            {
                sources = expanded.ToArray();
            }

            try
            {
                using (StreamWriter sw = ZStreamWriter.Open(destination))
                {
                    for (int i = 0; i < sources.Length; i++)
                    {
                        using (StreamReader sr = ZStreamReader.Open(sources[i]))
                        {
                            CopyLines(sr, sw);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new IOException("Could not concatenate all sources", ex);
            }
        }

        /// <summary>
        /// Consume all bytes from a stream, in the background.
        /// </summary>
        /// <param name="stream">the stream to consume</param>
        /// <returns>the thread that is consuming the data (which can be ignored)</returns>
        /// <remarks>
        /// <p>
        /// This method ignores all errors. It is useful for tasks such as ignoring
        /// StandardError from a process.
        /// </p>
        /// <p>
        /// The stream will be closed when the end is reached.
        /// </p>
        /// </remarks>
        public static Thread ConsumeBackground(Stream stream)
        {
            if (stream == null || !stream.CanRead)
                return null;
            ParameterizedThreadStart consumeStart = new ParameterizedThreadStart(ReadAll);
            Thread consume = Utils.CreateBackgroundThread(consumeStart);
            consume.Start(stream);
            return consume;
        }

        /// <summary>
        /// Consume all lines from a stream, in the background.
        /// </summary>
        /// <param name="reader">the TextReader to consume</param>
        /// <returns>the thread that is consuming the data (which can be ignored)</returns>
        /// <remarks>
        /// <p>
        /// This method ignores all errors. It is useful for tasks such as ignoring
        /// StandardError from a process.
        /// </p>
        /// <p>
        /// The stream will be closed when the end is reached.
        /// </p>
        /// </remarks>
        public static Thread ConsumeBackground(TextReader reader)
        {
            if (reader == null)
                return null;
            ParameterizedThreadStart consumeStart = new ParameterizedThreadStart(ReadAllLines);
            Thread consume = Utils.CreateBackgroundThread(consumeStart);
            consume.Start(reader);
            return consume;
        }

        private static void ReadAll(object streamObj)
        {
            Stream stream = (Stream)streamObj;
            try
            {
                byte[] buffer = new byte[1024];
                while (stream.Read(buffer, 0, buffer.Length) > 0)
                {
                }
            }
            catch
            {
            }
            finally
            {
                try
                {
                    stream.Close();
                }
                catch
                {
                }
            }
        }

        private static void ReadAllLines(object readerObj)
        {
            TextReader reader = (TextReader)readerObj;
            try
            {
                char[] buffer = new char[512];
                int c;
                while ((c = reader.Read(buffer, 0, buffer.Length)) > 0)
                {
                    
                }
            }
            catch
            {
            }
            finally
            {
                try
                {
                    reader.Close();
                }
                catch
                {
                }
            }
        }

        #region Line Enumeration

#if OLD_LINES
        /// <summary>
        /// Get an enumerator for the lines in the file or stream fileName
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>an IEnumerable that generates an IEnumerator for the lines in fileName</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// <para>
        /// This allows the lines in a stream to be read as:
        /// <code>
        ///    foreach (string line in IOUtil.Lines("file.txt"))
        ///    {
        ///        ...
        ///    }
        /// </code>
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static IEnumerable Lines(string fileName)
        {
            return Lines(fileName, false);
        }
        /// <summary>
        /// Get an enumerator for the lines in the file or stream fileName
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <param name="skipBlank">if true, skip blank lines; if false, read all lines</param>
        /// <returns>an IEnumerable that generates an IEnumerator for the lines in fileName</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// <para>
        /// This allows the lines in a stream to be read as:
        /// <code>
        ///    foreach (string line in IOUtil.Lines("file.txt"))
        ///    {
        ///        ...
        ///    }
        /// </code>
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static IEnumerable Lines(string fileName, bool skipBlank)
        {
            return new LineEnumerable(fileName, skipBlank);
        }

        private class LineEnumerable : IEnumerable
        {
            private string fileName;
            private bool skipBlank;

            public LineEnumerable(string fileName, bool skipBlank)
            {
                this.fileName = fileName;
                this.skipBlank = skipBlank;
            }

            public IEnumerator GetEnumerator()
            {
                return new LineEnumerator(fileName, skipBlank);
            }
        }

        /// <summary>
        /// Enumerator to read through the lines in a StreamReader.
        /// </summary>
        private class LineEnumerator : IEnumerator, IDisposable
        {
            private string fileName;
            private StreamReader reader;
            private string line = null;
            private bool skipBlank = false;

            /// <summary>
            /// Create a new enumerator to read through the lines.
            /// </summary>
            /// <param name="fileName">the file to read lines from</param>
            public LineEnumerator(string fileName)
                : this(fileName, false)
            {
            }
            /// <summary>
            /// Create a new enumerator to read through the lines.
            /// </summary>
            /// <param name="fileName">the file to read lines from</param>
            /// <param name="skipBlank">if true, skip blank lines; if false, read all lines</param>
            public LineEnumerator(string fileName, bool skipBlank)
            {
                this.fileName = fileName;
                this.skipBlank = skipBlank;
                this.reader = ZStreamReader.Open(fileName);
            }
            ~LineEnumerator()
            {
                Dispose();
            }
            /// <summary>
            /// Return the enumerator to the initial state.
            /// </summary>
            public void Reset()
            {
                //// can we do this? What about the BOM?
                //reader.DiscardBufferedData();
                //reader.BaseStream.Seek(0, SeekOrigin.Begin);
                //// should there be an IResetable?
                try
                {
                    reader.Close();
                }
                catch
                {
                }
                reader = ZStreamReader.Open(fileName);
                line = null;
            }
            /// <summary>
            /// Get the current line of the file.
            /// </summary>
            public string Current
            {
                get
                {
                    return line;
                }
            }

            /// <summary>
            /// Get the current line of the file.
            /// </summary>
            object IEnumerator.Current
            {
                get
                {
                    return ((LineEnumerator)this).Current;
                }
            }

            /// <summary>
            /// Move the enumerator to the next line.
            /// </summary>
            /// <returns>true if the next line exists, or false if at the end of the file</returns>
            public bool MoveNext()
            {
                do
                {
                    line = reader.ReadLine();
                }
                    while (skipBlank && (line != null && line.TrimEnd().Length == 0));

                return line != null;
            }
        #region IDisposable Members

            public void Dispose()
            {
                try
                {
                    reader.Close();
                }
                catch
                {
                }
                GC.SuppressFinalize(this);
            }

        #endregion
        }
#else
        /// <summary>
        /// Get an enumerator for the lines in the file or stream fileName
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>an IEnumerable that generates an IEnumerator for the lines in fileName</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// <para>
        /// This allows the lines in a stream to be read as:
        /// <code>
        ///    foreach (string line in IOUtil.Lines("file.txt"))
        ///    {
        ///        ...
        ///    }
        /// </code>
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static IEnumerable<string> Lines(string fileName)
        {
            return Lines(fileName, false);
        }
        /// <summary>
        /// Get an enumerator for the lines in the file or stream fileName
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <param name="skipBlank">if true, skip blank lines; if false, read all lines</param>
        /// <returns>an IEnumerable that generates an IEnumerator for the lines in fileName</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// <para>
        /// This allows the lines in a stream to be read as:
        /// <code>
        ///    foreach (string line in IOUtil.Lines("file.txt"))
        ///    {
        ///        ...
        ///    }
        /// </code>
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static IEnumerable<string> Lines(string fileName, bool skipBlank)
        {
            using (StreamReader reader = ZStreamReader.Open(fileName))
            {
                for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
                {
                    if (skipBlank)
                    {
                        bool foundNonSpace = false;
                        foreach (char c in line)
                        {
                            if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\v')
                            {
                                foundNonSpace = true;
                                break;
                            }
                        }
                        if (!foundNonSpace)
                            continue;
                    }
                    yield return line;
                }
            }
        }
#endif

        #endregion

        /// <summary>
        /// Read the file or stream specified by fileName.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>the text of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// </remarks>
        ///// <exception cref="ArgumentNullException">fileName is null.</exception>
        ///// <exception cref="ArgumentException">fileName is invalid.</exception>
        ///// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        ///// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static string ReadFile(string fileName)
        {
            try
            {
                using (StreamReader sr = ZStreamReader.Open(fileName))
                {
                    return sr.ReadToEnd();
                }
            }
            catch
            {
                return null;
            }
        }
        /// <summary>
        /// Read the file or stream specified by fileName.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <param name="encoding">the encoding to use for reading the text</param>
        /// <returns>the text of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// </remarks>
        ///// <exception cref="ArgumentNullException">fileName is null.</exception>
        ///// <exception cref="ArgumentException">fileName is invalid.</exception>
        ///// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        ///// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static string ReadFile(string fileName, System.Text.Encoding encoding)
        {
            try
            {
                using (StreamReader sr = ZStreamReader.Open(fileName, encoding))
                {
                    return sr.ReadToEnd();
                }
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Write text to the file specified by fileName.
        /// </summary>
        /// <param name="text">the text to write</param>
        /// <param name="fileName">the file to write</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(string text, string fileName)
        {
            using (StreamWriter sr = ZStreamWriter.Open(fileName))
            {
                if (text != null)
                {
                    sr.Write(text);
                }
            }
        }
        /// <summary>
        /// Write text to the file specified by fileName.
        /// </summary>
        /// <param name="text">the text to write</param>
        /// <param name="fileName">the file to write</param>
        /// <param name="append">if true, append to the file; otherwise, create or overwrite</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(string text, string fileName, bool append)
        {
            using (StreamWriter sr = ZStreamWriter.Open(fileName, append))
            {
                if (text != null)
                {
                    sr.Write(text);
                }
            }
        }

        /// <summary>
        /// Read the file or stream specified by fileName, as a byte array.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>the bytes of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamIn.Open(string)"/>.
        /// </remarks>
        public static byte[] ReadBytes(string fileName)
        {
            try
            {
                long len = GetLength(fileName);
                if (len >= 0)
                {
                    byte[] res = new byte[len];
                    using (Stream sr = ZStreamIn.Open(fileName))
                    {
                        sr.Read(res, 0, res.Length);
                        return res;
                    }
                }
                else
                {
                    // might just be forward-only...
                    // need double the memory, then...
                    List<byte[]> buffers = new List<byte[]>();
                    long totalLength = 0;
                    using (Stream sr = ZStreamIn.Open(fileName))
                    {
                        byte[] buf = new byte[256 * 1024];
                        int count;
                        while ((count = sr.Read(buf, 0, buf.Length)) > 0)
                        {
                            buffers.Add(buf);
                            totalLength += count;
                            if (count < buf.Length)
                                break;
                            buf = new byte[256 * 1024];
                        }
                        byte[] res = new byte[totalLength];
                        int cur = 0;
                        for (int i = 0; i < buffers.Count - 1; i++)
                        {
                            Buffer.BlockCopy((byte[])buffers[i], 0, res, cur, buf.Length);
                            cur += buf.Length;
                            buffers[i] = null;
                        }
                        if (buffers.Count != 0)
                        {
                            Buffer.BlockCopy((byte[])buffers[buffers.Count - 1], 0, res, cur, count);
                        }

                        return res;
                    }
                }
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Write bytes to the file specified by fileName.
        /// </summary>
        /// <param name="data">the bytes to write</param>
        /// <param name="fileName">the file to write</param>
        /// <para>
        /// This is an alias for <see cref="WriteBytes(byte[],string)"/>.
        /// </para>
        /// <remarks>
        /// <para>
        /// The writing is performed through <see cref="ZStreamOut.Open(string)"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(byte[] data, string fileName)
        {
            WriteBytes(data, fileName);
        }
        /// <summary>
        /// Write bytes to the file specified by fileName.
        /// </summary>
        /// <param name="data">the bytes to write</param>
        /// <param name="fileName">the file to write</param>
        /// <param name="append">if true, append to the file; otherwise, create or overwrite</param>
        /// <remarks>
        /// <para>
        /// This is an alias for <see cref="WriteBytes(byte[],string,bool)"/>.
        /// </para>
        /// <para>
        /// The writing is performed through <see cref="ZStreamOut.Open(string)"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(byte[] data, string fileName, bool append)
        {
            WriteBytes(data, fileName, append);
        }
        /// <summary>
        /// Write bytes to the file specified by fileName.
        /// </summary>
        /// <param name="data">the bytes to write</param>
        /// <param name="fileName">the file to write</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamOut.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteBytes(byte[] data, string fileName)
        {
            using (Stream sr = ZStreamOut.Open(fileName))
            {
                if (data != null)
                {
                    sr.Write(data, 0, data.Length);
                }
            }
        }
        /// <summary>
        /// Write bytes to the file specified by fileName.
        /// </summary>
        /// <param name="data">the bytes to write</param>
        /// <param name="fileName">the file to write</param>
        /// <param name="append">if true, append to the file; otherwise, create or overwrite</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamOut.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteBytes(byte[] data, string fileName, bool append)
        {
            using (Stream sr = ZStreamOut.Open(fileName, append))
            {
                if (data != null)
                {
                    sr.Write(data, 0, data.Length);
                }
            }
        }

        /// <summary>
        /// Read the lines of the file or stream specified by fileName.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>the lines of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static string[] ReadLines(string fileName)
        {
            try
            {
                using (StreamReader sr = ZStreamReader.Open(fileName))
                {
                    return ReadLines(sr);
                }
            }
            catch
            {
                return null;
            }
        }
        /// <summary>
        /// Read the lines of the file or stream specified by fileName.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <param name="encoding">the encoding to use for reading the text</param>
        /// <returns>the lines of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static string[] ReadLines(string fileName, System.Text.Encoding encoding)
        {
            try
            {
                using (StreamReader sr = ZStreamReader.Open(fileName, encoding))
                {
                    return ReadLines(sr);
                }
            }
            catch
            {
                return null;
            }
        }
        /// <summary>
        /// Read the lines of the StreamReader specified.
        /// </summary>
        /// <param name="input">the TextReader to read</param>
        /// <returns>the lines of input, or null if it cannot be read</returns>
        public static string[] ReadLines(TextReader input)
        {
            try
            {
                List<string> lines = new List<string>();
                for (string line = input.ReadLine(); line != null; line = input.ReadLine())
                {
                    lines.Add(line);
                }
                if (lines.Count > 0 && ((string)lines[lines.Count - 1]).Length == 0)
                    lines.RemoveAt(lines.Count - 1);
                return lines.ToArray();
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Write the lines to the file or stream specified by fileName.
        /// </summary>
        /// <param name="lines">the lines to write</param>
        /// <param name="fileName">the file to write</param>
        /// <remarks>
        /// <para>
        /// This is an alias for <see cref="WriteLines(string[],string)"/>.
        /// </para>
        /// <para>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open the stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(string[] lines, string fileName)
        {
            WriteLines(lines, fileName);
        }
        /// <summary>
        /// Write the lines to the file or stream specified by fileName.
        /// </summary>
        /// <param name="lines">the lines to write</param>
        /// <param name="fileName">the file to write</param>
        /// <param name="append">if true, append to the stream; otherwise, overwrite</param>
        /// <remarks>
        /// <para>
        /// This is an alias for <see cref="WriteLines(string[],string,bool)"/>.
        /// </para>
        /// <para>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open the stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteFile(string[] lines, string fileName, bool append)
        {
            WriteLines(lines, fileName, append);
        }
        /// <summary>
        /// Write the lines to the file or stream specified by fileName.
        /// </summary>
        /// <param name="lines">the lines to write</param>
        /// <param name="fileName">the file to write</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open the stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteLines(string[] lines, string fileName)
        {
            WriteLines(lines, fileName, false);
        }
        /// <summary>
        /// Write the lines to the file or stream specified by fileName.
        /// </summary>
        /// <param name="lines">the lines to write</param>
        /// <param name="fileName">the file to write</param>
        /// <param name="append">if true, append to the stream; otherwise, overwrite</param>
        /// <remarks>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open the stream are not available.</exception>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteLines(string[] lines, string fileName, bool append)
        {
            using (StreamWriter sw = ZStreamWriter.Open(fileName, append))
            {
                WriteLines(lines, sw);
            }
        }
        /// <summary>
        /// Write the lines to the specified TextWriter.
        /// </summary>
        /// <param name="lines">the lines to write</param>
        /// <param name="output">the TextWriter to write to</param>
        /// <remarks>
        /// <p>
        /// The writing is performed through <see cref="ZStreamWriter.Open(string)"/>.
        /// </p>
        /// <p>
        /// The output will not be closed.
        /// </p>
        /// </remarks>
        /// <exception cref="IOException">An error occurs while writing.</exception>
        public static void WriteLines(string[] lines, TextWriter output)
        {
            if (lines == null)
                lines = new string[0];
            try
            {
                for (int i = 0; i < lines.Length; i++)
                {
                    output.Write(lines[i]);
                }
            }
            catch (Exception ex)
            {
                throw new IOException("Cannot write lines to output", ex);
            }
        }

        /// <summary>
        /// Read a file compiled as Embedded Content in the calling assembly.
        /// </summary>
        /// <param name="name">The name of the file to read, without any namespace</param>
        /// <returns>The contents of the file, or null if it does not exist</returns>
        public static string ReadResource(string name)
        {
            try
            {
                using (System.IO.StreamReader str = ReadResourceReader(name))
                {
                    if (str == null)
                        return null;
                    return str.ReadToEnd();
                }
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Read a file compiled as Embedded Content in the calling assembly.
        /// </summary>
        /// <param name="name">The name of the file to read, without any namespace</param>
        /// <returns>A StreamReader for reading the file, or null if it does not exist</returns>
        public static StreamReader ReadResourceReader(string name)
        {
            Stream st = ReadResourceStream(name);
            if (st == null)
                return null;
            return new StreamReader(st);
        }

        /// <summary>
        /// Read a file compiled as Embedded Content in the calling assembly.
        /// </summary>
        /// <param name="name">The name of the file to read, without any namespace</param>
        /// <returns>A Stream for reading the file, or null if it does not exist</returns>
        public static Stream ReadResourceStream(string name)
        {
            // not a streamname, because it is really a code-level issue...
            if (name == null || name.Length == 0)
                return null;
            try
            {
                string[] names = System.Reflection.Assembly.GetCallingAssembly().GetManifestResourceNames();
                string fullName = null;
                string dotName = "." + name;
                foreach (string s in names)
                {
                    // this is a hack... ***
                    if (s == name || s.EndsWith(dotName, StringComparison.OrdinalIgnoreCase))
                    {
                        fullName = s;
                        break;
                    }
                }
                if (fullName == null)
                    return null;
                System.Reflection.Assembly assem = System.Reflection.Assembly.GetExecutingAssembly();
                return assem.GetManifestResourceStream(name);
            }
            catch
            {
                return null;
            }
        }

        #endregion

        #region Checksum, Etc
        #region Tables
        static readonly uint[] _crctab =
            {
                0x0,
                0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B,
                0x1A864DB2, 0x1E475005, 0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6,
                0x2B4BCB61, 0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
                0x4C11DB70, 0x48D0C6C7, 0x4593E01E, 0x4152FDA9, 0x5F15ADAC,
                0x5BD4B01B, 0x569796C2, 0x52568B75, 0x6A1936C8, 0x6ED82B7F,
                0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3, 0x709F7B7A,
                0x745E66CD, 0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039,
                0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5, 0xBE2B5B58,
                0xBAEA46EF, 0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033,
                0xA4AD16EA, 0xA06C0B5D, 0xD4326D90, 0xD0F37027, 0xDDB056FE,
                0xD9714B49, 0xC7361B4C, 0xC3F706FB, 0xCEB42022, 0xCA753D95,
                0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1, 0xE13EF6F4,
                0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D, 0x34867077, 0x30476DC0,
                0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5,
                0x2AC12072, 0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16,
                0x018AEB13, 0x054BF6A4, 0x0808D07D, 0x0CC9CDCA, 0x7897AB07,
                0x7C56B6B0, 0x71159069, 0x75D48DDE, 0x6B93DDDB, 0x6F52C06C,
                0x6211E6B5, 0x66D0FB02, 0x5E9F46BF, 0x5A5E5B08, 0x571D7DD1,
                0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
                0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B,
                0xBB60ADFC, 0xB6238B25, 0xB2E29692, 0x8AAD2B2F, 0x8E6C3698,
                0x832F1041, 0x87EE0DF6, 0x99A95DF3, 0x9D684044, 0x902B669D,
                0x94EA7B2A, 0xE0B41DE7, 0xE4750050, 0xE9362689, 0xEDF73B3E,
                0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2, 0xC6BCF05F,
                0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34,
                0xDC3ABDED, 0xD8FBA05A, 0x690CE0EE, 0x6DCDFD59, 0x608EDB80,
                0x644FC637, 0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB,
                0x4F040D56, 0x4BC510E1, 0x46863638, 0x42472B8F, 0x5C007B8A,
                0x58C1663D, 0x558240E4, 0x51435D53, 0x251D3B9E, 0x21DC2629,
                0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5, 0x3F9B762C,
                0x3B5A6B9B, 0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF,
                0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623, 0xF12F560E,
                0xF5EE4BB9, 0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65,
                0xEBA91BBC, 0xEF68060B, 0xD727BBB6, 0xD3E6A601, 0xDEA580D8,
                0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD, 0xCDA1F604, 0xC960EBB3,
                0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7, 0xAE3AFBA2,
                0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B, 0x9B3660C6, 0x9FF77D71,
                0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74,
                0x857130C3, 0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640,
                0x4E8EE645, 0x4A4FFBF2, 0x470CDD2B, 0x43CDC09C, 0x7B827D21,
                0x7F436096, 0x7200464F, 0x76C15BF8, 0x68860BFD, 0x6C47164A,
                0x61043093, 0x65C52D24, 0x119B4BE9, 0x155A565E, 0x18197087,
                0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
                0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D,
                0x2056CD3A, 0x2D15EBE3, 0x29D4F654, 0xC5A92679, 0xC1683BCE,
                0xCC2B1D17, 0xC8EA00A0, 0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB,
                0xDBEE767C, 0xE3A1CBC1, 0xE760D676, 0xEA23F0AF, 0xEEE2ED18,
                0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4, 0x89B8FD09,
                0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662,
                0x933EB0BB, 0x97FFAD0C, 0xAFB010B1, 0xAB710D06, 0xA6322BDF,
                0xA2F33668, 0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4
            };

        static readonly uint[] _zipCrctab =
            {
                0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
                0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
                0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
                0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
                0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
                0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
                0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
                0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
                0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
                0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
                0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
                0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
                0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
                0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
                0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
                0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
                0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
                0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
                0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
                0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
                0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
                0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
                0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
                0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
                0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
                0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
                0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
                0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
                0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
                0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
                0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
                0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
                0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
                0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
                0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
                0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
                0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
                0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
                0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
                0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
                0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
                0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
                0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
            };
        #endregion

        /// <summary>
        /// The types of checksums that can be calculated.
        /// </summary>
        public enum ChecksumType
        {
            /// <summary>
            /// The checksum used in the ZIP and gzip file formats.
            /// </summary>
            Zip,
            /// <summary>
            /// Standard CRC32.
            /// </summary>
            Crc32,
            /// <summary>
            /// The first bytes of the SHA1 hash.
            /// </summary>
            SHA1Prefix,
            /// <summary>
            /// The first bytes of the SHA256 hash.
            /// </summary>
            SHA256Prefix,
        }

        /// <summary>
        /// Calculate a standard ZIP/gzip checksum for the specified file.
        /// </summary>
        /// <param name="fileName">the file to find the checksum of</param>
        /// <returns>the checksum, as an integer</returns>
        /// <remarks>
        /// To convert this to a string as displayed for ZIP and gzip,
        /// use ToString("X8").
        /// </remarks>
        public static uint Checksum(string fileName)
        {
            unchecked
            {
                return (uint)(Checksum(fileName, ChecksumType.Zip) & 0xFFFFFFFF);
            }
        }

        /// <summary>
        /// Calculate a standard CRC32 checksum for the specified file.
        /// </summary>
        /// <param name="fileName">the file to find the checksum of</param>
        /// <param name="type">the type of checksum to calculate</param>
        /// <returns>the checksum, as a 64-bit integer</returns>
        /// <remarks>
        /// For checksums smaller than 64-bits, the high bits of the return
        /// value will be zero.
        /// </remarks>
        public static ulong Checksum(string fileName, ChecksumType type)
        {
            long length = 0;

            using (Stream inFile = ZStreamIn.Open(fileName))
            {
                // compute the basic crc
                switch (type)
                {
                case ChecksumType.Crc32:
                default:
                    unsafe
                    {
                        uint crc32 = 0;
                        if (inFile is UnbufferedStream)
                        {
                            UnbufferedStream inFileU = (UnbufferedStream)inFile;
                            byte* buffer;
                            for (int count = inFileU.Read(out buffer); count > 0; count = inFileU.Read(out buffer))
                            {
                                length += count;
                                byte* bEnd = buffer + count;
                                while (buffer != bEnd)
                                {
                                    crc32 = (crc32 << 8) ^ _crctab[(crc32 >> 24) ^ (((uint)(*buffer)) & 0xFF)];
                                    buffer++;
                                }
                            }
                        }
                        else
                        {
                            byte[] buffer = new byte[4 * 1024 * 1024];
                            for (int count = inFile.Read(buffer, 0, buffer.Length); count > 0; count = inFile.Read(buffer, 0, buffer.Length))
                            {
                                length += count;
                                fixed (byte* bb = buffer)
                                {
                                    byte* b = bb;
                                    byte* bEnd = bb + count;
                                    while (b != bEnd)
                                    {
                                        crc32 = (crc32 << 8) ^ _crctab[(crc32 >> 24) ^ (((uint)(*b)) & 0xFF)];
                                        b++;
                                    }
                                }
                            }
                        }
                        long adj = length;
                        while (adj != 0)
                        {
                            crc32 = (crc32 << 8) ^
                                _crctab[((crc32 >> 24) ^ adj) & 0xFF];
                            adj >>= 8;
                        }
                        crc32 = ~crc32 & 0xFFFFFFFF;
                        return crc32;
                    }

                case ChecksumType.Zip:
                    unsafe
                    {
                        uint crcZip = 0xffffffff;
                        if (inFile is UnbufferedStream)
                        {
                            UnbufferedStream inFileU = (UnbufferedStream)inFile;
                            byte* buffer;
                            for (int count = inFileU.Read(out buffer); count > 0; count = inFileU.Read(out buffer))
                            {
                                length += count;
                                byte* bEnd = buffer + count;
                                while (buffer != bEnd)
                                {
                                    // New checksum value
                                    crcZip = (crcZip >> 8) ^ _zipCrctab[((int)crcZip ^ (*buffer)) & 0xff];
                                    buffer++;
                                }
                            }
                        }
                        else
                        {
                            byte[] buffer = new byte[4 * 1024 * 1024];
                            for (int count = inFile.Read(buffer, 0, buffer.Length); count > 0; count = inFile.Read(buffer, 0, buffer.Length))
                            {
                                //Console.WriteLine("read: " + count + "  total: " + length);
                                length += count;
                                fixed (byte* bb = buffer)
                                {
                                    byte* b = bb;
                                    byte* bEnd = bb + count;
                                    while (b != bEnd)
                                    {
                                        // New checksum value
                                        crcZip = (crcZip >> 8) ^ _zipCrctab[((int)crcZip ^ (*b)) & 0xff];
                                        b++;
                                    }
                                }
                            }
                        }
                        // fix up the crc
                        crcZip = crcZip ^ 0xffffffff;
                        return crcZip;
                    }

                case ChecksumType.SHA1Prefix:
                    {
                        System.Security.Cryptography.SHA1Managed sha = new System.Security.Cryptography.SHA1Managed();
                        byte[] hash = sha.ComputeHash(inFile);
                        return BitConverter.ToUInt64(hash, 0);
                    }

                case ChecksumType.SHA256Prefix:
                    {
                        System.Security.Cryptography.SHA256Managed sha = new System.Security.Cryptography.SHA256Managed();
                        byte[] hash = sha.ComputeHash(inFile);
                        return BitConverter.ToUInt64(hash, 0);
                    }
                }
            }
        }
        #endregion
#else
        #region File-only operations

        /// <summary>
        /// Set the length of a file, truncating or padding as needed.
        /// </summary>
        /// <param name="fileName">the file to alter</param>
        /// <param name="length">the desired length, in bytes</param>
        /// <remarks>
        /// Note that this only operates on normal files, and cannot be successfully used on compressed
        /// files or special streams.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="IOException"> An I/O error has occurred.</exception>
        /// <exception cref="NotSupportedException">The stream does not support both writing and seeking.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Attempted to set the value parameter to less than 0.</exception>
        public static void ResizeFile(string fileName, long length)
        {
            using (FileStream pad = new FileStream(fileName, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite))
            {
                pad.SetLength(length);
            }
        }

        /// <summary>
        /// Determine if a file exists.
        /// </summary>
        /// <param name="fileName">the original filename</param>
        /// <returns>true if a file with a name of fileName exists, false otherwise</returns>
        public static bool FileExists(string fileName)
        {
            return !string.IsNullOrEmpty(fileName) && File.Exists(fileName);
        }

        private static readonly char[] wildChars = new char[] { '*', '?' };
        private static readonly char[] wildPlusChars = new char[] { '*', '?', '+' };

        /// <summary>
        /// Expand an extended wildcard pattern into a set of file paths.
        /// </summary>
        /// <param name="pattern">the pattern to expand</param>
        /// <returns>the set of file paths matching the pattern</returns>
        /// <remarks>
        /// The wildcard pattern accepts the standard "*" and "?" placeholders.
        /// "..." also refers to a recursive search over subdirectories.
        /// "+" can also be used to make a union of several filenames or patterns.
        /// In addition to filenames, HTTP URLs, <c>nul</c>, <c>null</c>, <c>$</c>,
        /// <c>-</c>, and Cosmos stream names are all recognized as elements.
        /// Names of files that do not exist will be excluded.
        /// </remarks>
        public static string[] ExpandWildcards(string pattern)
        {
            if (pattern == null || (pattern.IndexOfAny(wildPlusChars) < 0 && pattern.IndexOf("...") < 0))
            {
                if (FileExists(pattern))
                {
                    return new string[] { pattern };
                }
                else
                {
                    return new string[0];
                }
            }
            List<string> matchList = new List<string>();
            bool disjoint = false;
            int filePatternCount = 0;
            string[] patterns = pattern.Split('+');
            foreach (string pat in patterns)
            {
                // hard-code in special types??
                if (pat.Length == 0)
                    continue;
                string patLower = pat.ToLower();

                filePatternCount++;
                int prepatternCount = matchList.Count;
                if (pat.IndexOfAny(wildChars) >= 0 || pat.IndexOf("...") >= 0)
                {
                    // compressed extensions are not automatically used! ***
                    int recursiveIndex = pat.IndexOf("...");
                    if (recursiveIndex >= 0)
                    {
                        string left = pat.Substring(0, recursiveIndex);
                        string right = pat.Substring(recursiveIndex + 3);
                        right = right.TrimStart('\\', '/');
                        if (right.Length == 0)
                            right = "*";
                        string path = left;
                        bool pathEmpty = (path == null || path.Length == 0);
                        if (pathEmpty)
                            path = ".";
                        Stack dirsLeft = new Stack();
                        dirsLeft.Push(path);
                        while (dirsLeft.Count != 0)
                        {
                            string dir = (string)dirsLeft.Pop();

                            // watch for lack of access:
                            try
                            {
                                // this is actually incorrect, for 3-char extensions: ***
                                string[] files = Directory.GetFiles(dir, right);
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);

                                string[] subs = Directory.GetDirectories(dir);
                                for (int s = subs.Length - 1; s >= 0; s--)
                                {
                                    dirsLeft.Push(subs[s]);
                                }
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                    }
                    else
                    {
                        try
                        {
                            string path = Path.GetDirectoryName(pat);
                            bool pathEmpty = !(pat.StartsWith("./") || pat.StartsWith(".\\"));
                            if (path == null || path.Length == 0)
                                path = ".";
                            // watch for lack of access:
                            try
                            {
                                string[] files = Directory.GetFiles(path, Path.GetFileName(pat));
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                        catch
                        {
                            // ignore bad path?
                        }
                    }
                }
                else
                {
                    // what to do?? Filter to only those that exist?? ***
                    if (!FileExists(pat))
                        continue;
                    matchList.Add(pat);
                }
                if (filePatternCount > 1 && matchList.Count != prepatternCount)
                {
                    disjoint = true;
                }
            }
            if (disjoint || true)
            {
                // remove duplicates, very inefficiently - but it is simple, preserves
                // the order, uses no additional memory, and is case-insensitive...:
                for (int i = 0; i < matchList.Count - 1; i++)
                {
                    for (int j = i + 1; j < matchList.Count; j++)
                    {
                        if (string.Compare((string)matchList[i], (string)matchList[j], true) == 0)
                        {
                            matchList.RemoveAt(j);
                            j--;
                        }
                    }
                }
            }
            return matchList.ToArray();
        }

        /// <summary>
        /// Read the lines of the file or stream specified by fileName.
        /// </summary>
        /// <param name="fileName">the file to read (or URL, or Cosmos stream...)</param>
        /// <returns>the lines of fileName, or null if it cannot be read</returns>
        /// <remarks>
        /// The reading is performed through <see cref="ZStreamReader.Open(string)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static string[] ReadLines(string fileName)
        {
            try
            {
                using (StreamReader input = ZStreamReader.Open(fileName))
                {
                    try
                    {
                        List<string> lines = new List<string>();
                        for (string line = input.ReadLine(); line != null; line = input.ReadLine())
                        {
                            lines.Add(line);
                        }
                        if (lines.Count > 0 && ((string)lines[lines.Count - 1]).Length == 0)
                            lines.RemoveAt(lines.Count - 1);
                        return lines.ToArray();
                    }
                    catch
                    {
                        return null;
                    }
                }
            }
            catch
            {
                return null;
            }
        }
        #endregion
#endif
    }

#if TLCFULLBUILD
    /// <summary>
    /// Class representing file or directory information.
    /// </summary>
    /// <remarks>
    /// This is similar to the purpose of <see cref="FileSystemInfo"/> and its
    /// subclasses, <see cref="FileInfo"/> and <see cref="DirectoryInfo"/>.
    /// However, they are not easy to effectively subclass.
    /// </remarks>
    public class StreamInfo : IComparable
    {
        private readonly string _path;
        private long _length;
        private DateTime _lastWriteTime;
        private string _canonicalPath;

        /// <summary>
        /// Create a new StreamInfo.
        /// </summary>
        /// <param name="path">the name of the file or directory - a directory
        /// must end with "/" or "\"</param>
        /// <param name="length">the length of the item</param>
        /// <param name="lastWriteTime">the last write time of the item</param>
        public StreamInfo(string path, long length, DateTime lastWriteTime)
        {
            _length = length;
            _lastWriteTime = lastWriteTime;
            _canonicalPath = null;
            _path = path;
            if (_path == null)
            {
                _path = "";
            }
            else
            {
                _path = path;
            }
        }

        /// <summary>
        /// Create a new StreamInfo for an existing file or directory.
        /// </summary>
        /// <param name="path">the path to the file or directory</param>
        /// <exception cref="FileNotFoundException">The given path does not exist.</exception>
        public StreamInfo(string path)
        {
            if (path == null || path.Length == 0)
            {
                throw new FileNotFoundException("No directory or file for empty path");
            }
            try
            {
                _canonicalPath = null;
                _path = path;
                // this is not really efficient:
                if (IOUtil.FileExists(path))
                {
                    // this only works for file systems!!
                    FileInfo f = new FileInfo(path);
                    _length = f.Length;
                    _lastWriteTime = f.LastWriteTime;
                }
                else if (IOUtil.DirectoryExists(path))
                {
                    // this only works for file systems!!
                    if (path[path.Length - 1] != '/' && path[path.Length - 1] != '\\')
                    {
                        _path = _path + "/";
                    }
                    _length = 0;
                    _lastWriteTime = Directory.GetLastWriteTime(path);
                }
                else
                {
                    throw new FileNotFoundException("No directory or file: " + path);
                }
            }
            catch
            {
                throw new FileNotFoundException("Cannot access directory or file: " + path);
            }
        }

        /// <summary>
        /// Create a new StreamInfo for an existing file or directory.
        /// </summary>
        /// <param name="path">the path to the file or directory</param>
        /// <param name="delay">dummy parameter to indicate delaying the fetch of metadata</param>
        /// <exception cref="FileNotFoundException">The given path does not exist.</exception>
        internal StreamInfo(string path, bool delay)
        {
            _canonicalPath = null;
            _path = path;
            _length = -1;
            _lastWriteTime = DateTime.MinValue;
        }

        /// <summary>
        /// Compare two instances, based on the canonical path.
        /// </summary>
        /// <param name="obj">the instance to compare to</param>
        /// <returns>negative if this instance is less than obj; 0 if equal;
        /// postive otherwise</returns>
        public int CompareTo(object obj)
        {
            StreamInfo other = obj as StreamInfo;
            if (other == null)
                return -1;
            return CanonicalPath.CompareTo(other.CanonicalPath);
        }

        /// <summary>
        /// Determine if this instance is equal to another.
        /// </summary>
        /// <param name="obj">the instance to compare to</param>
        /// <returns>true if obj is a StreamInfo referring to the same canonical path;
        /// false otherwise</returns>
        public override bool Equals(object obj)
        {
            if (obj is StreamInfo)
            {
                return CanonicalPath.Equals(((StreamInfo)obj).CanonicalPath);
            }

            return false;
        }

        /// <summary>
        /// Get the hashcode, based on the canonical path.
        /// </summary>
        /// <returns>the hash code</returns>
        public override int GetHashCode()
        {
            return CanonicalPath.GetHashCode();
        }

        /// <summary>
        /// Get the <see cref="CanonicalPath"/>.
        /// </summary>
        /// <returns>the canonical name</returns>
        public override string ToString()
        {
            return CanonicalPath;
        }

        /// <summary>
        /// Gets the full path of the file or directory.
        /// </summary>
        public string Path
        {
            get { return _path; }
        }

        /// <summary>
        /// Gets the name of this item, with no parent path or salshes.
        /// </summary>
        public string Name
        {
            get
            {
                if (_path.Length == 0)
                    return "";
                int end = _path[_path.Length - 1] == '/' || _path[_path.Length - 1] == '\\' ?
                    _path.Length - 1 : _path.Length;
                // what if it ends in multiple slashes? ***
                int start = _path.LastIndexOfAny(_pathSeperators, end - 1);
                if (start < 0)
                {
                    start = 0;
                }
                else
                {
                    start++;
                }
                return _path.Substring(start, end - start);
            }
        }

        private static readonly char[] _pathSeperators = new char[] { '/', '\\' };

        /// <summary>
        /// Gets the full path of the file or directory in a standard form.
        /// </summary>
        public string CanonicalPath
        {
            get
            {
                if (_canonicalPath == null)
                {
                    _canonicalPath = IOUtil.GetCanonicalPath(_path);
                }
                return _canonicalPath;
            }
        }

        /// <summary>
        /// Gets whether the item is a directory (not including archives,
        /// which can be accessed as directories).
        /// </summary>
        public bool IsDirectory
        {
            get
            {
                return Path.Length != 0 && CanonicalPath[CanonicalPath.Length - 1] == '/';
            }
        }
        /// <summary>
        /// Gets whether the item is a compressed file or archive.
        /// </summary>
        public bool IsCompressed
        {
            get
            {
                string ext = System.IO.Path.GetExtension(Path);
                if (ext.Length == 0)
                    return false;
                ext = ext.ToLower();
                return Array.IndexOf(ZStreamIn.decompressionExtensions, ext) >= 0;
            }
        }

        /// <summary>
        /// Gets the last modification time.
        /// </summary>
        public DateTime LastWriteTime
        {
            get
            {
                if (_lastWriteTime == DateTime.MinValue)
                {
                    if (IsDirectory)
                    {
                        if (Directory.Exists(_path))
                        {
                            _lastWriteTime = Directory.GetLastWriteTime(_path);
                        }
                        else
                        {
                            // not actually a file...
                        }
                    }
                    else
                    {
                        if (File.Exists(_path))
                        {
                            _lastWriteTime = File.GetLastWriteTime(_path);
                        }
                        else
                        {
                            // not actually a file...
                        }
                    }
                }
                return _lastWriteTime;
            }
        }

        /// <summary>
        /// Gets the length of the file in bytes, or 0 for directories.
        /// </summary>
        public long Length
        {
            get
            {
                if (_length < 0)
                {
                    if (IsDirectory)
                    {
                        _length = 0;
                    }
                    else
                    {
                        if (File.Exists(_path))
                        {
                            FileInfo f = new FileInfo(_path);
                            _length = f.Length;
                        }
                        else
                        {
                            // this shouldn't happen...
                        }
                    }
                }
                return _length;
            }
        }
    }

#endif // TLCFULLBUILD
    #endregion

    #region ZStreams

#if TLCFULLBUILD
    /// <summary>
    /// Class to create StreamReaders that automatically decompress based on the file extensions.
    /// </summary>
    /// <remarks>
    /// <p>
    /// Compressed files are recognized by extension and automatically decompressed.
    /// Filenames that do not exist are checked to see if compressed versions exist; if so,
    /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
    /// "doc.txt" is requested but does not exist). To read compressed files directly,
    /// without decompression, append a "$" to the filename.
    /// </p>
    /// <para>There are several special filenames:</para>
    /// <list type="bullet">
    /// <item><description>
    /// The special names "nul" and "null" refer to an empty stream.
    /// </description></item>
    /// <item><description>
    /// The special names "-" and "$" refer to the console input.
    /// </description></item>
    /// <item><description>
    /// URLs starting with "http://" or "https://" are downloaded with HTTP.
    /// </description></item>
    /// <item><description>
    /// Names starting with "cosmos://" are fetched as Cosmos streams.
    /// </description></item>
    /// <item><description>
    /// The name "clip:" refers to the clipboard, for reading text.
    /// </description></item>
    /// <item><description>
    /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
    /// </description></item>
    /// <item><description>
    /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
    /// list of other files.
    /// </description></item>
    /// <item><description>
    /// Names ending with ":streamname" open the NTFS named stream "streamname".
    /// </description></item>
    /// </list>
    /// <para>
    /// Compression support relies on executable utilities to be in the path.
    /// See <see href="http://7-zip.org"/> for 7z.exe and 7za.exe (for many formats -
    /// .7z, .gz, .zip, .rar, .bz2, .cab, .arj), <see href="http://gnuwin32.sourceforge.net/packages/gzip.htm"/> for gzip.exe
    /// (for .gz), or <see href="http://rarsoft.com"/> for unrar.exe (for .rar).
    /// </para>
    /// </remarks>
#else
    /// <summary>
    /// Class to create StreamReaders given file paths.
    /// </summary>
#endif
    public class ZStreamReader //: StreamReader
    {
        //        private string tempDir = null;
        //        private static string fallbackExtension = "";
        private static bool _defaultToLocalEncoding = false;

        private static int _bufferSize = 32 * 1024; //-1; //1024*1024; //32768;

        /// <summary>
        /// Get or set whether to allow fallback to the compression library if executables
        /// are not found in the path. false by default. Using the fallback may result in
        /// slower performance and larger files. This setting is shared with ZStreamIn,
        /// ZStreamOut, ZStreamReader, and ZStreamWriter.
        /// </summary>
        public static bool AllowLibraryFallback
        {
            get { return ZStreamIn.AllowLibraryFallback; }
            set { ZStreamIn.AllowLibraryFallback = value; }
        }

        private ZStreamReader()
        {
        }

        /// <summary>
        /// Get or set whether to default to the local encoding, rather than a lenient UTF8.
        /// </summary>
        public static bool DefaultToLocalEncoding
        {
            get { return _defaultToLocalEncoding; }
            set { _defaultToLocalEncoding = value; }
        }

        /// <summary>
        /// Get or set extension to look to append when the given filename does not exist.
        /// If set to empty string (the default), try all known extensions;
        /// if set to null, disable.
        /// This is mapped to ZStreamIn.FallbackExtension.
        /// </summary>
        public static string FallbackExtension
        {
            //get { return fallbackExtension; }
            //set { fallbackExtension = value; }
            get { return ZStreamIn.FallbackExtension; }
            set { ZStreamIn.FallbackExtension = value; }
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamReader Open(string fileName)
        {
            Encoding enc = DefaultToLocalEncoding ? Encoding.Default : ZStreamWriter.UTF8Lenient;
            return Open(fileName, enc);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <param name="encoding">the encoding to use</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamReader Open(string fileName, Encoding encoding)
        {
#if TLCFULLBUILD
            // check for SqlStream - we don't want to open this as a Stream:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return new SqlTextReader(fileName);
            }
#endif
            // hack for console! It seems to break otherwise, at times, at least in 1.1...
            Stream s = null;
            try
            {
                s = ZStreamIn.Open(fileName);
                if (_bufferSize > 0)
                {
                    return new StreamReader(s, encoding, true, _bufferSize);
                }
                else
                {
                    return new StreamReader(s, encoding, true);  //, BUFFER_SIZE);
                }
            }
            catch
            {
                if (s != null)
                {
                    try
                    {
                        s.Close();
                    }
                    catch
                    {
                    }
                }
                throw;
            }
        }

#if UNBUFFERED

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file with normal file caching.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <para>
        /// This method opens the file with system caching, regardless of the setting of
        /// <see cref="ZStreamIn.DefaultUnbuffered"/>.
        /// </para>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>
        /// There are several special filenames:
        /// </para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamReader OpenBuffered(string fileName)
        {
            Encoding enc = DefaultToLocalEncoding ? Encoding.Default : ZStreamWriter.UTF8Lenient;
            return OpenBuffered(fileName, enc);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file with normal file caching.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <param name="encoding">the encoding to use</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <para>
        /// This method opens the file with system caching, regardless of the setting of
        /// <see cref="ZStreamIn.DefaultUnbuffered"/>.
        /// </para>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>
        /// There are several special filenames:
        /// </para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamReader OpenBuffered(string fileName, Encoding encoding)
        {
#if TLCFULLBUILD
            // check for  Azure Storage objects
            AzureStorageIO azureStorage = new AzureStorageIO();
            string fileNameLower = fileName.ToLower();

            //REVIEW: Standardize on a naming convention or employ some other way to distinguish azure vs general http streams.
            // check for HTTP:
            if (fileNameLower.StartsWith("http://") || fileNameLower.StartsWith("https://"))
            {
                if (azureStorage.BlockBlobExistsByUri(fileNameLower))
                {
                    return new StreamReader(azureStorage.GetBlobStream(fileNameLower), encoding);
                }
            }

            // check for SqlStream - we don't want to open this as a Stream:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return new SqlTextReader(fileName);
            }
            // hack for console!
#endif
            Stream s = null;
            try
            {
                s = ZStreamIn.OpenBuffered(fileName);
                if (_bufferSize > 0)
                {
                    return new StreamReader(s, encoding, true, _bufferSize);
                }
                else
                {
                    return new StreamReader(s, encoding, true);  //, BUFFER_SIZE);
                }
            }
            catch
            {
                if (s != null)
                {
                    try
                    {
                        s.Close();
                    }
                    catch
                    {
                    }
                }
                throw;
            }
        }

        /// <summary>
        /// Open the specified file (with unbuffered I/O, if possible).
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. This is the only way to get speeds over
        /// 60 MB/sec or more on reading (350 MB/sec or more is possible on a good array).
        /// </para>
        /// <para>
        /// While compressed files and special stream names will be understood, unbuffered I/O will
        /// not be enabled on anything but simple files.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamReader OpenUnbuffered(string fileName)
        {
            Encoding enc = DefaultToLocalEncoding ? Encoding.Default : ZStreamWriter.UTF8Lenient;
            return OpenUnbuffered(fileName, enc);
        }
        /// <summary>
        /// Open the specified file (with unbuffered I/O, if possible).
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <param name="encoding">the encoding to use</param>
        /// <returns>A StreamReader for the (possibly uncompressed) text</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. This is the only way to get speeds over
        /// 60 MB/sec or more on reading (350 MB/sec or more is possible on a good array).
        /// </para>
        /// <para>
        /// While compressed files and special stream names will be understood, unbuffered I/O will
        /// not be enabled on anything but simple files.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamReader OpenUnbuffered(string fileName, Encoding encoding)
        {
#if TLCFULLBUILD
            // check for  Azure Storage objects
            AzureStorageIO azureStorage = new AzureStorageIO();
            string fileNameLower = fileName.ToLower();
            if (fileNameLower.StartsWith("http://") || fileNameLower.StartsWith("https://"))
            {
                if (azureStorage.BlockBlobExistsByUri(fileNameLower))
                {
                    return new StreamReader(azureStorage.GetBlobStream(fileNameLower), encoding);
                }
            }

            // check for SqlStream - we don't want to open this as a Stream:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return new SqlTextReader(fileName);
            }
#endif
            Stream s = null;
            try
            {
                s = ZStreamIn.OpenUnbuffered(fileName);
                return new StreamReader(s, encoding, true, 64 * 1024);
            }
            catch
            {
                if (s != null)
                {
                    try
                    {
                        s.Close();
                    }
                    catch
                    {
                    }
                }
                throw;
            }
        }

#endif

#if TEMP_FILE_RAR
        /////////////////////////
        //// Old temp file code:
                    // assume only one file inside!
                    System.CodeDom.Compiler.TempFileCollection tempCollection = new System.CodeDom.Compiler.TempFileCollection();
                    tempCollection.KeepFiles = true;
                    ztempDir = tempCollection.BasePath;
                    Directory.CreateDirectory(ztempDir);
                    string fullName = "\"" + (new FileInfo(fileName)).FullName + "\"";
                    System.Diagnostics.ProcessStartInfo procInfo = new System.Diagnostics.ProcessStartInfo("unrar", "e -y -o+ -inul -dh " + fullName);
                    procInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
                    procInfo.CreateNoWindow = true;
                    procInfo.UseShellExecute = false;
                    procInfo.WorkingDirectory = ztempDir;
                    System.Diagnostics.Process proc = System.Diagnostics.Process.Start(procInfo);
                    proc.WaitForExit();
                    string[] fileNames = Directory.GetFiles(ztempDir);
                    if (fileNames.Length == 0)
                    {
                        try
                        {
                            try
                            {
                                string[] badFiles = Directory.GetFiles(ztempDir);
                                if (badFiles != null)
                                {
                                    for (int i = 0; i < badFiles.Length; i++)
                                    {
                                        File.Delete(Path.Combine(ztempDir, badFiles[i]));
                                    }
                                }
                            }
                            catch //(Exception e2)
                            {
                                // ignore any problems...
                                //Console.WriteLine("!! (Open_) Problem deleting in " + ztempDir + ": " + e2.ToString());
                            }
                            Directory.Delete(ztempDir, true);
                        }
                        catch //(Exception e)
                        {
                            // ignore any problems...
                            //Console.WriteLine("!! (Open_) Problem deleting " + ztempDir + ": " + e.ToString());
                        }
                        // should we open the original file, then? ***
                        throw new Exception("unrar failed on file '" + fileName + "'");
                    }
                    s = new FileStream(fileNames[0], FileMode.Open, FileAccess.Read);

        private ZStreamReader(Stream s, string ztempDir, Encoding e, bool detectEncoding, int bufferSize)
            : base(s, e, detectEncoding, bufferSize)
        {
            tempDir = ztempDir;
        }

        /// <summary>
        /// Destroy the ZStreamReader.
        /// </summary>
        ~ZStreamReader()
        {
            Dispose(true);
        }

        /// <summary>
        /// Close the StreamReader.
        /// </summary>
        public override void Close()
        {
            try
            {
                base.Close();
                isClosed = true;
            }
            finally
            {
                if (tempDir != null)
                {
                    try
                    {
                        //Console.WriteLine("(Close) Should be deleting: " + tempDir);
                        try
                        {
                            string[] badFiles = Directory.GetFiles(tempDir);
                            if (badFiles != null)
                            {
                                for (int i = 0; i < badFiles.Length; i++)
                                {
                                    File.Delete(Path.Combine(tempDir, badFiles[i]));
                                }
                            }
                        }
                        catch //(Exception e2)
                        {
                            // ignore any problems...
                            //Console.WriteLine("!! (Close) Problem deleting " + tempDir + ": " + e2.ToString());
                        }
                        Directory.Delete(tempDir, true);
                        tempDir = null;
                    }
                    catch //(Exception e)
                    {
                        // ignore any problems...
                        //Console.WriteLine("!! (Close) Problem deleting " + tempDir + ": " + e.ToString());
                    }
                }
            }
        }

        private bool isClosed = false;
        /// <summary>
        /// Release all resources
        /// </summary>
        /// <param name="disposing"></param>
        protected override void Dispose(bool disposing)
        {
            if (!isClosed)
            {
                try
                {
                    base.Close();
                    isClosed = true;
                }
                catch
                {
                    // ignore - probably already closed or some such.
                }
            }
            if (tempDir != null)
            {
                try
                {
                    //Console.WriteLine("(Dispose) Should be deleting: " + tempDir);
                    try
                    {
                        string[] badFiles = Directory.GetFiles(tempDir);
                        if (badFiles != null)
                        {
                            for (int i = 0; i < badFiles.Length; i++)
                            {
                                File.Delete(Path.Combine(tempDir, badFiles[i]));
                            }
                        }
                    }
                    catch
                    {
                        // ignore any problems...
                    }
                    Directory.Delete(tempDir, true);
                    tempDir = null;
                }
                catch
                {
                    // ignore any problems...
                }
            }
            base.Dispose (disposing);
        }
#endif
    }

#if TLCFULLBUILD
    /// <summary>
    /// Class to create StreamWriters that automatically compress based on the file extensions.
    /// </summary>
    /// <remarks>
    /// <p>
    /// Compressed files are recognized by extension and automatically compressed.
    /// To write to a file with a compression extension directly, without compression,
    /// append a "$" to the filename.
    /// </p>
    /// <para>There are several special filenames:</para>
    /// <list type="bullet">
    /// <item><description>
    /// The special names "nul" and "null" refer to an empty stream.
    /// </description></item>
    /// <item><description>
    /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
    /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
    /// </description></item>
    /// <item><description>
    /// Names starting with "cosmos://" are stored as Cosmos streams.
    /// </description></item>
    /// <item><description>
    /// The name "clip:" refers to the clipboard, for writing text.
    /// </description></item>
    /// <item><description>
    /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
    /// </description></item>
    /// <item><description>
    /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
    /// list of other files.
    /// </description></item>
    /// <item><description>
    /// Names ending with ":streamname" open the NTFS named stream "streamname".
    /// </description></item>
    /// </list>
    /// <para>
    /// Compression support relies on executable utilities to be in the path.
    /// See <see href="http://7-zip.org"/> for 7z.exe and 7za.exe (for
    /// .7z, .gz), <see href="http://gnuwin32.sourceforge.net/packages/gzip.htm"/> for gzip.exe
    /// (for .gz).
    /// </para>
    /// </remarks>
#else
    /// <summary>
    /// Class to create StreamWriters given file paths.
    /// </summary>
#endif
    public class ZStreamWriter //: StreamWriter
    {
        // backing field for UTF8Lenient
        private static readonly Encoding _utf8Lenient = new UTF8Encoding(false, false);

        /// <summary>
        /// A lenient UTF8 encoding that ignores problems and skips the BOM.
        /// </summary>
        public static Encoding UTF8Lenient
        {
            get
            {
                return _utf8Lenient;
            }
        }

        private const string WriteNewLine = "\r\n";
        private static Encoding _writeEncoding = UTF8Lenient;
        //private static int compressionLevel = 1;
        private static bool _breakChunksAtLines = true;

        private static int _bufferSize = 32 * 1024;  //-1;  //1024*1024; //32768;

        /// <summary>
        /// Get or set whether the Open method should use a LowFragmentationStream for files.
        /// true, by default.
        /// </summary>
        /// <remarks>
        /// The <see cref="LowFragmentationStream"/> has strong advantages, increasing write
        /// speed and decreasing fragmentation.
        /// </remarks>
        public static bool DefaultLowFragmentation
        {
            get { return ZStreamOut.DefaultLowFragmentation; }
            set { ZStreamOut.DefaultLowFragmentation = value; }
        }

        /// <summary>
        /// Get or set the encoding that is used for new StreamWriters.
        /// </summary>
        public static Encoding WriteEncoding
        {
            get { return _writeEncoding; }
            set { _writeEncoding = value; }
        }

#if TLCFULLBUILD
        /// <summary>
        /// Get or set whether to break at line boundaries when using chunked streams,
        /// such as <see cref="CosmosWriteStream"/> or <see cref="MultiStream"/>.
        /// True by default, unlike <see cref="ZStreamOut"/>.
        /// </summary>
        /// <remarks>
        /// This will not necessarily have any effect. It is currently unimplemented in
        /// <see cref="MultiStream"/>, and most streams have no concept of chunks.
        /// </remarks>
#else
        /// <summary>
        /// Get or set whether to break at line boundaries when using chunked streams.
        /// True by default, unlike <see cref="ZStreamOut"/>.
        /// </summary>
        /// <remarks>
        /// This will not necessarily have any effect. Most streams have no concept of chunks.
        /// </remarks>
#endif
        public static bool BreakChunksAtLines
        {
            get { return _breakChunksAtLines; }
            set { _breakChunksAtLines = value; }
        }

        /// <summary>
        /// Get or set whether to allow fallback to the compression library if executables
        /// are not found in the path. false by default. Using the fallback may result in
        /// slower performance and larger files. This setting is shared with ZStreamIn,
        /// ZStreamOut, ZStreamReader, and ZStreamWriter.
        /// </summary>
        public static bool AllowLibraryFallback
        {
            get { return ZStreamIn.AllowLibraryFallback; }
            set { ZStreamIn.AllowLibraryFallback = value; }
        }

        private ZStreamWriter()
        {
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
        /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamWriter Open(string outFileName)
        {
            return Open(outFileName, false);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="encoding">Encoding for writing</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
        /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamWriter Open(string outFileName, Encoding encoding)
        {
            return Open(outFileName, false, encoding);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
        /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamWriter Open(string outFileName, bool append)
        {
            return Open(outFileName, append, BreakChunksAtLines);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="encoding">Encoding for writing</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
        /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamWriter Open(string outFileName, bool append, Encoding encoding)
        {
            return Open(outFileName, append, BreakChunksAtLines, false, encoding);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="breakChunksAtLines">if true, break at line boundaries when using chunked streams</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is line buffered,
        /// which may cause problems if the data is needed immediately and is not as fast as full buffering.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static StreamWriter Open(string outFileName, bool append, bool breakChunksAtLines)
        {
            return Open(outFileName, append, breakChunksAtLines, false);
        }

        private static StreamWriter Open(string outFileName, bool append, bool breakChunksAtLines, bool unbuffered)
        {
            return Open(outFileName, append, breakChunksAtLines, unbuffered, WriteEncoding);
        }

        private static StreamWriter Open(string outFileName, bool append, bool breakChunksAtLines, bool unbuffered, Encoding encoding)
        {
            // check for SqlStream - we don't want to open this as a Stream:
#if TLCFULLBUILD
            if (SqlTextReader.IsSqlTextReader(outFileName))
            {
                return new SqlTextWriter(outFileName);
            }
#endif
            Stream s = unbuffered ? ZStreamOut.OpenUnbuffered(outFileName, append, breakChunksAtLines)
                : ZStreamOut.Open(outFileName, append, breakChunksAtLines);

            StreamWriter sw;
            int size = _bufferSize;
            if (string.Compare(outFileName, "clip:", true) == 0)
            {
                size = 4;
            }
            if (size > 0)
            {
                sw = new StreamWriter(s, encoding, size);
            }
            else
            {
                sw = new StreamWriter(s, encoding);
            }
            sw.NewLine = WriteNewLine;
#if TLCFULLBUILD
            if (string.Compare(outFileName, "clip:", true) == 0)
            {
                // we really want line-level flushing! ***
                //sw.AutoFlush = true;
            }
            if (ZStreamIn.IsConsoleStream(outFileName))
            {
                // match standard behavior, but not efficient in many cases!
                // we might really want line-level flushing! ***
                sw.AutoFlush = true;
                ((LineBufferedStream)sw.BaseStream).LineBuffer = true;
            }
#endif
            return sw;
        }

#if UNBUFFERED

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="encoding">Encoding for writing</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamWriter OpenUnbuffered(string outFileName, Encoding encoding)
        {
            return OpenUnbuffered(outFileName, false, encoding);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamWriter OpenUnbuffered(string outFileName)
        {
            return OpenUnbuffered(outFileName, false);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamWriter OpenUnbuffered(string outFileName, bool append)
        {
            return OpenUnbuffered(outFileName, append, BreakChunksAtLines);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="encoding">Encoding for writing</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamWriter OpenUnbuffered(string outFileName, bool append, Encoding encoding)
        {
            return Open(outFileName, append, _breakChunksAtLines, true, encoding);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="outFileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="breakChunksAtLines">if true, break at line boundaries when using chunked streams</param>
        /// <returns>A StreamWriter for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static StreamWriter OpenUnbuffered(string outFileName, bool append, bool breakChunksAtLines)
        {
            return Open(outFileName, append, breakChunksAtLines, true);
        }

#endif
    }

#if TLCFULLBUILD
    /// <summary>
    /// Class to create input Streams that automatically decompress based on the file extensions.
    /// </summary>
    /// <remarks>
    /// <p>
    /// Compressed files are recognized by extension and automatically decompressed.
    /// Filenames that do not exist are checked to see if compressed versions exist; if so,
    /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
    /// "doc.txt" is requested but does not exist). To read compressed files directly,
    /// without decompression, append a "$" to the filename.
    /// </p>
    /// <para>There are several special filenames:</para>
    /// <list type="bullet">
    /// <item><description>
    /// The special names "nul" and "null" refer to an empty stream.
    /// </description></item>
    /// <item><description>
    /// The special names "-" and "$" refer to the console input.
    /// </description></item>
    /// <item><description>
    /// URLs starting with "http://" or "https://" are downloaded with HTTP.
    /// </description></item>
    /// <item><description>
    /// Names starting with "cosmos://" are fetched as Cosmos streams.
    /// </description></item>
    /// <item><description>
    /// The name "clip:" refers to the clipboard, for reading text.
    /// </description></item>
    /// <item><description>
    /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
    /// </description></item>
    /// <item><description>
    /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
    /// list of other files.
    /// </description></item>
    /// <item><description>
    /// Names ending with ":streamname" open the NTFS named stream "streamname".
    /// </description></item>
    /// </list>
    /// <para>
    /// Compression support relies on executable utilities to be in the path.
    /// See <see href="http://7-zip.org"/> for 7z.exe and 7za.exe (for many formats -
    /// .7z, .gz, .zip, .rar, .bz2, .cab, .arj), <see href="http://gnuwin32.sourceforge.net/packages/gzip.htm"/> for gzip.exe
    /// (for .gz), or <see href="http://rarsoft.com"/> for unrar.exe (for .rar).
    /// </para>
    /// </remarks>
#else
    /// <summary>
    /// Class to create input Streams given file paths.
    /// </summary>
#endif
    public class ZStreamIn
    {
        private static string _fallbackExtension = "";
        private static bool _defaultUnbuffered = false;

        private static int _bufferSize = 32 * 1024;  //-1;  //64*1024; //32768;

        /// <summary>
        /// Get or set extension to look to append when the given filename does not exist.
        /// If set to empty string (the default), try all known extensions;
        /// if set to null, disable.
        /// </summary>
        public static string FallbackExtension
        {
            get { return _fallbackExtension; }
            set { _fallbackExtension = value; }
        }

        /// <summary>
        /// Get or set whether the Open method should use unbuffered I/O whenever possible.
        /// false, by default.
        /// </summary>
        public static bool DefaultUnbuffered
        {
            get { return _defaultUnbuffered; }
            set { _defaultUnbuffered = value; }
        }

        /// <summary>
        /// Get or set whether to allow fallback to the compression library if executables
        /// are not found in the path. false by default. Using the fallback may result in
        /// slower performance and larger files. This setting is shared with ZStreamIn,
        /// ZStreamOut, ZStreamReader, and ZStreamWriter.
        /// </summary>
        public static bool AllowLibraryFallback
        {
            get { return _allowLibraryFallback; }
            set { _allowLibraryFallback = value; }
        }
        private static bool _allowLibraryFallback = true;

        internal static readonly string[] decompressionArchiveExtensions = new string[]
            {
                // 7za:
                ".7z",
                ".zip",
                ".tar",
                // 7z:
                ".cab",
                ".arj",
                ".rar",
                ".lzh",
                ".chm"
            };
        internal static readonly string[] decompressionExtensions = new string[]
            {
                // 7za:
                ".gz",
                ".7z",
                ".zip",
                ".tar",
                ".bz2",
                ".z",
                // 7z:
                ".cab",
                ".arj",
                ".rar",
                ".lzh",
                ".chm"
            };
        internal static readonly string NullStreamName = "nul";
        private static readonly string[] _nullStreamNames = new string[]
            {
                "nul",
                "null"
            };
        internal static bool IsNullStream(string fileName)
        {
            if (fileName == null || fileName.Length == 0)
                return false;
            for (int i = 0; i < _nullStreamNames.Length; i++)
            {
                int len = _nullStreamNames[i].Length;
                if (string.Compare(fileName, 0, _nullStreamNames[i], 0, len, true) == 0)
                {
                    if (fileName.Length == len)
                        return true;

                    if (fileName[len] == '\\' ||
                        fileName[len] == '/')
                    {
                        len++;
                        // should collapse repeated slashes...
                        while (len < fileName.Length &&
                            (fileName[len] == '\\' || fileName[len] == '/'))
                        {
                            len++;
                        }
                        if (len == fileName.Length)
                            return true;
                        for (int j = 0; i < _nullStreamNames.Length; j++)
                        {
                            if (fileName.Length - len == _nullStreamNames[j].Length &&
                                string.Compare(fileName, len, _nullStreamNames[j], 0, _nullStreamNames[j].Length, true) == 0)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
        internal static readonly string ConsoleStreamName = "$";
        private static readonly string[] _consoleStreamNames = new string[]
            {
                "$",
                "-"
            };
        internal static bool IsConsoleStream(string fileName)
        {
            if (fileName == null || fileName.Length == 0)
                return false;
            for (int i = 0; i < _consoleStreamNames.Length; i++)
            {
                int len = _consoleStreamNames[i].Length;
                if (string.Compare(fileName, 0, _consoleStreamNames[i], 0, len, true) == 0)
                {
                    if (fileName.Length == len)
                        return true;

                    if (fileName[len] == '\\' ||
                        fileName[len] == '/')
                    {
                        len++;
                        // should collapse repeated slashes...
                        while (len < fileName.Length &&
                            (fileName[len] == '\\' || fileName[len] == '/'))
                        {
                            len++;
                        }
                        if (len == fileName.Length)
                            return true;
                        for (int j = 0; i < _consoleStreamNames.Length; j++)
                        {
                            if (fileName.Length - len == _consoleStreamNames[j].Length &&
                                string.Compare(fileName, len, _consoleStreamNames[j], 0, _consoleStreamNames[j].Length, true) == 0)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Gets the set of extensions (such as ".gz") that are accepted for decompression.
        /// </summary>
        public static string[] DecompressionExtensions
            => (string[])decompressionExtensions.Clone();

        /// <summary>
        /// Gets the set of extensions (such as ".gz") that are accepted for decompression
        /// as archives that can act as directories.
        /// </summary>
        public static string[] DecompressionArchiveExtensions
            => (string[])decompressionArchiveExtensions.Clone();

        private static readonly char[] _pathSeparators = new char[] { '/', '\\' };

        private ZStreamIn()
        {
        }

#if TLCFULLBUILD
        private static bool _gzipFailure;
        private static bool _gzip7ZFailure;
        private static bool _full7ZFailure;
#endif

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A Stream for the (possibly uncompressed) data</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static Stream Open(string fileName)
        {
            if (DefaultUnbuffered)
            {
                return Open(fileName, false, true, false);
            }
            else
            {
                return Open(fileName, true);
            }
        }

        private static Stream Open(string fileName, bool buffered)
        {
            return Open(fileName, buffered, false, false);
        }
        private static Stream Open(string fileName, bool buffered, bool bufferedFallback, bool async)
        {
            Contracts.CheckNonEmpty(fileName, nameof(fileName));

            // check for special names:
            string fileNameLower = fileName.ToLower();
#if TLCFULLBUILD
            if (IsNullStream(fileName))
            {
                //return new NullStream();
                return Stream.Null;
            }
            if (IsConsoleStream(fileName))
            {
                // note that ReadByte() and WriteByte() are pathetic, and allocate an array! ***
                // They also make too many managed->unmanaged transitions.
                // This is 200x slower than it should be.
                // Position also will not work.
                // Using a BufferedStream wrapper fixes the first problem, with some overhead.
                // A custom wrapper should probably be made, but it is a pain.
                //return new BufferedStream(Console.OpenStandardInput(), 4096);
                //return Console.OpenStandardInput();
                // larger sizes somehow break line buffering, making it be every *two* lines...
                return new BufferedStream(Console.OpenStandardInput(), 1024);
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(fileName))
            {
                return new ClipboardReadStream();
            }

            // check for Cosmos:
            if (fileNameLower.StartsWith("cosmos:"))
            {
                try
                {
                    // no compression support, anyway...
                    fileName = fileName.TrimEnd('$');
                    return new CosmosReadStream(fileName);
                }
                catch (InvalidOperationException)
                {
                    throw new InvalidOperationException("ZStreamIn requires cosmos.cmd or cosmos.exe to be in the " +
                        "path for Cosmos reading.");
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    throw new InvalidOperationException("ZStreamIn requires cosmos.cmd or cosmos.exe  to be in the " +
                        "path for Cosmos reading.");
                }
            }

            // check for Cockpit:
            if (fileNameLower.StartsWith("cockpit:") || fileNameLower.StartsWith("cockpit:"))
            {
                if (fileName.Length <= "cockpit:".Length)
                    throw new ArgumentException("Invalid cockpit streamname", "fileName");
                // should we URL-encode?
                fileName = fileName.Substring("cockpit:".Length);
                fileName = fileName.Replace('/', '\\');

                string cockpitServer = "cockpit.search.msn.com:81";

                // check for forced cockpit server:
                int hostStart = 0;
                while (hostStart < fileName.Length && fileName[hostStart] == '\\')
                    hostStart++;
                if (hostStart < fileName.Length - 2)
                {
                    int hostEnd = hostStart + 1;
                    while (hostEnd < fileName.Length && fileName[hostEnd] != '\\')
                        hostEnd++;
                    int div = fileName.IndexOf('@', hostStart + 1, hostEnd - hostStart - 1);
                    if (div > 0)
                    {
                        // extract cockpit server!
                        cockpitServer = fileName.Substring(div + 1, hostEnd - div - 1);
                        fileName = fileName.Substring(0, div) + fileName.Substring(hostEnd);
                    }
                }

                // ls is really for internal use...
                string cmd = "get";
                if (fileName.EndsWith("\\") || fileName.EndsWith("/"))
                    cmd = "ls";
                // force searchmsn proxy!!
                fileName = "http://" + cockpitServer + "@searchmsn/files?cmd=" + cmd + "&path=" +
                    fileName;
                fileNameLower = fileName.ToLower();
            }

            // check for Multistream:
            if (fileNameLower.StartsWith("multi:"))
            {
                fileName = fileName.Substring("multi:".Length);
                return new MultiStream(fileName);
            }
            if (fileNameLower.StartsWith("filelist:"))
            {
                fileName = fileName.Substring("filelist:".Length);
                return new MultiStream(fileName);
            }

            // check for HTTP:
            if (fileNameLower.StartsWith("http:") || fileNameLower.StartsWith("https:"))
            {
                // check for  Azure Storage objects
                AzureStorageIO azureStorage = new AzureStorageIO();
                if (azureStorage.BlockBlobExistsByUri(fileNameLower))
                {
                    return azureStorage.GetBlobStream(fileNameLower);
                }
                return new HttpStream(fileName);
            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return new SqlTextReader(fileName).CreateStream();
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(fileName))
            {
                try
                {
                    fileName = fileName.TrimEnd('$');
                    return new InternalStoreStream(fileName);
                }
                catch (InvalidOperationException)
                {
                    throw new InvalidOperationException("ZStreamIn requires tstore.exe to be in the " +
                        "path for InternalStore reading.");
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    throw new InvalidOperationException("ZStreamIn requires tstore.exe to be in the " +
                        "path for InternalStore reading.");
                }
            }

            // remove trailing "$"
            bool forceRaw = false;
            if (fileName[fileName.Length - 1] == '$')
            {
                forceRaw = true;
                fileName = fileName.Substring(0, fileName.Length - 1);
            }

            // check for named stream:
            // should this be based on file existance, first?
            int cIndex = fileName.LastIndexOf(':');
            if (cIndex > 0)
            {
                if (File.Exists(fileName.Substring(0, cIndex)))
                {
                    if (cIndex > 1 ||
                        fileName.IndexOfAny(_pathSeparators, 2) < 0)
                    {
                        // named:
                        return new NamedStream(fileName, false);
                    }
                }
            }

            Stream s = null;
            try
            {
                if (!forceRaw)
                {
                    // check for compressed versions, if needed:
                    if (FallbackExtension != null)
                    {
                        if (!File.Exists(fileName))
                        {
                            if (FallbackExtension.Length != 0)
                            {
                                if (File.Exists(fileName + FallbackExtension))
                                {
                                    fileName = fileName + FallbackExtension;
                                }
                            }
                            else
                            {
                                for (int i = 0; i < decompressionExtensions.Length; i++)
                                {
                                    if (File.Exists(fileName + decompressionExtensions[i]))
                                    {
                                        fileName = fileName + decompressionExtensions[i];
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    string ext = Path.GetExtension(fileName).ToLower();
                    switch (ext)
                    {
                    case ".gz":
                        {
                            // try using gzip, fall back to 7zip, and fall back to SharpZipLib:
                            if (!_gzipFailure)
                            {
#if GZIP_UNBUFFERED
                                    if (DefaultUnbuffered)
                                    {
                                        Stream sRaw = null;
                                        try
                                        {
                                            sRaw = OpenUnbuffered(fileName);
                                            try
                                            {
                                                s = new GzipDecodeStream(sRaw);
                                            }
                                            catch (InvalidOperationException)
                                            {
                                                gzipFailure = true;
                                                try
                                                {
                                                    if (sRaw != null) sRaw.Close();
                                                }
                                                catch
                                                {
                                                }
                                            }
                                            catch (System.ComponentModel.Win32Exception)
                                            {
                                                gzipFailure = true;
                                                try
                                                {
                                                    if (sRaw != null) sRaw.Close();
                                                }
                                                catch
                                                {
                                                }
                                            }
                                        }
                                        catch
                                        {
                                            try
                                            {
                                                if (sRaw != null) sRaw.Close();
                                            }
                                            catch
                                            {
                                            }
                                        }
                                    }
                                    else
                                    {
                                        try
                                        {
                                            s = new GzipDecodeStream(fileName);
                                        }
                                        catch (InvalidOperationException)
                                        {
                                            gzipFailure = true;
                                        }
                                        catch (System.ComponentModel.Win32Exception)
                                        {
                                            gzipFailure = true;
                                        }
                                    }
#else
                                try
                                {
                                    s = new GzipDecodeStream(fileName);
                                }
                                catch (InvalidOperationException)
                                {
                                    _gzipFailure = true;
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                    _gzipFailure = true;
                                }
#endif
                            }
                            if (s == null && !_gzip7ZFailure)
                            {
                                try
                                {
                                    s = new Z7zDecodeStream(fileName);  //, true);
                                }
                                catch (InvalidOperationException)
                                {
                                    _gzip7ZFailure = true;
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                    _gzip7ZFailure = true;
                                }
                            }
                            if (AllowLibraryFallback && s == null)
                            {
                                // this could support unbuffered, Cosmos, etc... ***
                                //// NOTE:
                                ////   - .NET's gzip is very slow (30% or more longer)
                                ////   - .NET's gzip is very large (30% larger compressed files)
                                ////   - .NET's gzip breaks for files over 4 GB
                                // could wrap to get length, maybe fix 4GB problem...

                                s = new System.IO.Compression.GZipStream(
                                    new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite),
                                    System.IO.Compression.CompressionMode.Decompress);
                            }
                            if (s == null)
                            {
                                throw new InvalidOperationException("ZStreamIn requires gzip.exe, 7za.exe, or 7z.exe to be in the " +
                                    "path for " + ext + " decompression, unless AllowLibraryFallback is set. " +
                                    "See http://7-zip.org or http://gnuwin32.sourceforge.net/packages/gzip.htm");
                            }
                        }
                        break;

                    //case ".gz":
                    case ".7z":
                    case ".zip":
                    //case ".bzip2":
                    case ".bz2":
                    case ".z":
                    case ".tar":
                        {
                            try
                            {
                                s = new Z7zDecodeStream(fileName, true);
                            }
                            catch (InvalidOperationException)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe or 7za.exe to be in the " +
                                    "path for " + ext + " decompression. " +
                                    "See http://7-zip.org");
                            }
                            catch (System.ComponentModel.Win32Exception)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe or 7za.exe to be in the " +
                                    "path for " + ext + " decompression. " +
                                    "See http://7-zip.org");
                            }
                        }
                        break;

                    //case ".rar":
                    //case ".rpm":
                    //case ".deb":
                    case ".cab":
                    case ".arj":
                    case ".lzh":
                    case ".chm":
                        {
                            try
                            {
                                s = new Z7zDecodeStream(fileName, true);
                            }
                            catch (InvalidOperationException)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe to be in the " +
                                    "path for " + ext + " decompression. " +
                                    "See http://7-zip.org");
                            }
                            catch (System.ComponentModel.Win32Exception)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe to be in the " +
                                    "path for " + ext + " decompression. " +
                                    "See http://7-zip.org");
                            }
                        }
                        break;

                    case ".rar":
                        {
                            if (!_full7ZFailure)
                            {
                                try
                                {
                                    s = new Z7zDecodeStream(fileName, true);
                                }
                                catch (InvalidOperationException)
                                {
                                    _full7ZFailure = true;
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                    _full7ZFailure = true;
                                }
                            }
                            if (s == null)
                            {
                                try
                                {
                                    s = new RarDecodeStream(fileName);
                                }
                                catch (InvalidOperationException)
                                {
                                    //unrarFailure = true;
                                    throw new InvalidOperationException("ZStreamIn requires unrar.exe or 7z.exe to be in the " +
                                        "path for " + ext + " decompression. " +
                                        "See http://7-zip.org or http://rarsoft.com");
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                    //unrarFailure = true;
                                    throw new InvalidOperationException("ZStreamIn requires unrar.exe or 7z.exe to be in the " +
                                        "path for " + ext + " decompression. " +
                                        "See http://7-zip.org or http://rarsoft.com");
                                }
                            }
                        }
                        break;

#if ENABLE_LZMA
                        case ".lzma":
                            {
                                try
                                {
                                    s = new LzmaDecodeStream(fileName, false);
                                }
                                catch (InvalidOperationException)
                                {
                                    throw new InvalidOperationException("ZStreamIn requires lzma.exe to be in the " +
                                        "path for " + ext + " decompression.");
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                    throw new InvalidOperationException("ZStreamIn requires lzma.exe to be in the " +
                                        "path for " + ext + " decompression.");
                                }
                            }
                            break;
#endif
                    }

                    if (s == null && !File.Exists(fileName))
                    {
                        // check for compressed archive as directory segment:
                        // *** TODO !
                        // check for compressed archives in path:
                        // only one path segment is allowed to be an archive...
                        // normalize path:
                        //if (fileName[fileName.Length - 1] != '\\')  fileName = fileName + "\\";
                        string zfileName = fileName.Replace('/', '\\');
                        bool isUnc = zfileName.StartsWith("\\\\");
                        while (zfileName.IndexOf("\\\\") >= 0)
                        {
                            zfileName = zfileName.Replace("\\\\", "\\");
                        }
                        if (isUnc)
                            zfileName = "\\" + zfileName;
                        string zfileNameLower = zfileName.ToLower();

                        string archPath = null;
                        string inArch = null;
                        // should this really be only archives?? ***
                        for (int i = 0; i < ZStreamIn.decompressionExtensions.Length; i++)
                        {
                            ext = ZStreamIn.decompressionExtensions[i];
                            int seg = zfileNameLower.IndexOf(ext + "\\");
                            if (seg > 0)
                            {
                                archPath = zfileName.Substring(0, seg + ext.Length);
                                if (File.Exists(archPath))
                                {
                                    inArch = zfileName.Substring(seg + ext.Length).Trim('/', '\\');
                                    break;
                                }
                                archPath = null;
                            }
                        }
                        if (archPath == null)
                        {
                            // add in extension to each segment...
                            string[] segs = zfileName.Split('\\');
                            for (int i = 0; i < segs.Length; i++)
                            {
                                if (segs[i].Length == 0)
                                    continue;
                                string partial = string.Join("\\", segs, 0, i + 1);
                                if (partial.Length == 2 && partial[1] == ':')
                                    continue;
                                if (Directory.Exists(partial))
                                    continue;
                                // should this really be only archives?? ***
                                for (int c = 0; c < ZStreamIn.decompressionExtensions.Length; c++)
                                {
                                    ext = ZStreamIn.decompressionExtensions[c];
                                    if (File.Exists(partial + ext))
                                    {
                                        archPath = partial + ext;
                                        inArch = string.Join("\\", segs, i + 1, segs.Length - i - 1).Trim('/', '\\');
                                        break;
                                    }
                                }
                                // quit when parent will not exist
                                break;
                            }
                        }
                        if (archPath != null)
                        {
                            //Console.WriteLine(archPath + " :: " + inArch);
                            // check for path in archive.
                            inArch = inArch.Trim('\\');
                            if (inArch.Length != 0)
                            {
                                // optimize for gzip and support unrar.exe:
                                if (Path.GetExtension(archPath).ToLower() == ".gz")
                                {
                                    try
                                    {
                                        return Open(archPath);
                                    }
                                    catch (InvalidOperationException)
                                    {
                                    }
                                }
                                try
                                {
                                    s = new Z7zDecodeStream(archPath, inArch);
                                }
                                catch (InvalidOperationException)
                                {
                                }
                                catch (System.ComponentModel.Win32Exception)
                                {
                                }
                                if (s == null && !Z7zDecodeStream.Exists7z &&
                                    Path.GetExtension(archPath).ToLower() == ".rar")
                                {
                                    try
                                    {
                                        s = new RarDecodeStream(archPath, inArch);
                                    }
                                    catch (InvalidOperationException)
                                    {
                                    }
                                    catch (System.ComponentModel.Win32Exception)
                                    {
                                    }
                                }
                                if (s == null)
                                {
                                    throw new InvalidOperationException("ZStreamIn requires 7z.exe or 7za.exe to be in the " +
                                        "path for " + ext + " decompression. " +
                                        "See http://7-zip.org");
                                }
                            }
                        }
                    }
                }
#else
            Stream s = null;
            try
            {
#endif
                if (s == null)
                {
                    // don't use unbuffered for small files:
                    if (!buffered && bufferedFallback)
                    {
                        try
                        {
                            long len = (new FileInfo(fileName)).Length;
                            if (len <= 2 * 8 * 1024 * 1024)
                            {
                                buffered = true;
                            }
                        }
                        catch
                        {
                        }
                    }
                    // unbuffered:
                    if (!buffered)
                    {
#if UNBUFFERED
                        // assume sequential and not async? ***
                        // why are we ignoring the async flag?
                        try
                        {
                            try
                            {
                                //return new BufferedStream(new UnbufferedStream(fileName), 8 * 1024 * 1024);
                                return new UnbufferedStream(fileName);
                            }
                            catch (UnbufferedStream.VirtualAllocException)
                            {
                                // always fallback on memory allocation failure: ?
                                //return Open(fileName, true, async);
                                throw;
                            }
                        }
                        catch
                        {
                            if (!bufferedFallback)
                                throw;
                        }
#else
                        throw new NotSupportedException("Unbuffered IO is not supported.");
#endif
                    }

                    // buffered:
                    if (_bufferSize > 0)
                    {
                        s = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, _bufferSize);
                    }
                    else
                    {
                        s = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                    }
                }
                return s;
            }
            catch
            {
                if (s != null)
                {
                    try
                    {
                        s.Close();
                    }
                    catch
                    {
                    }
                }
                throw;
            }
        }

        //        private static FileStream OpenUnbufferedStream(string filename)
        //        {
        //#if UNBUFFERED
        //            return UnbufferedStream.Open(filename, FileMode.Open, FileAccess.Read, FileShare.None, true, false, BUFFER_SIZE);
        //#else
        //            throw new NotSupportedException("Unbuffered IO is not supported.");
        //#endif
        //        }

#if UNBUFFERED

#if TLCFULLBUILD
        /// <summary>
        /// Open the specified file with normal file caching.
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A Stream for the (possibly uncompressed) data</returns>
        /// <remarks>
        /// <para>
        /// This method opens the file with system caching, regardless of the setting of
        /// <see cref="ZStreamIn.DefaultUnbuffered"/>.
        /// </para>
        /// </remarks>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically decompressed.
        /// Filenames that do not exist are checked to see if compressed versions exist; if so,
        /// the compressed file is silently opened. (For example, "doc.txt.gz" will be used if
        /// "doc.txt" is requested but does not exist). To read compressed files directly,
        /// without decompression, append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console input.
        /// </description></item>
        /// <item><description>
        /// URLs starting with "http://" or "https://" are downloaded with HTTP.
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are fetched as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for reading text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" or "sql:server/db/{query}" refers to SQL Server.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static Stream OpenBuffered(string fileName)
        {
            return Open(fileName, true);
        }

        /// <summary>
        /// Open the specified file (with unbuffered I/O, if possible).
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <returns>A Stream for the data</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. This is the only way to get speeds over
        /// 60 MB/sec or more on reading (350 MB/sec or more is possible on a good array).
        /// </para>
        /// <para>
        /// While compressed files and special stream names will be understood, unbuffered I/O will
        /// not be enabled on anything but simple files.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static Stream OpenUnbuffered(string fileName)
        {
            return OpenUnbuffered(fileName, false);
        }
        /// <summary>
        /// Open the specified file (with unbuffered I/O, if possible).
        /// </summary>
        /// <param name="fileName">name of the file to open</param>
        /// <param name="async">whether to use asynchronous I/O</param>
        /// <returns>A Stream for the data</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. This is the only way to get speeds over
        /// 60 MB/sec or more on reading (350 MB/sec or more is possible on a good array).
        /// </para>
        /// <para>
        /// While compressed files and special stream names will be understood, unbuffered I/O will
        /// not be enabled on anything but simple files.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="FileNotFoundException">fileName cannot be found.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        public static Stream OpenUnbuffered(string fileName, bool async)
        {
            return Open(fileName, false, false, async);
        }

#endif
    }

#if TLCFULLBUILD
    /// <summary>
    /// Class to create output Streams that automatically compress based on the file extensions.
    /// </summary>
    /// <remarks>
    /// <p>
    /// Compressed files are recognized by extension and automatically compressed.
    /// To write to a file with a compression extension directly, without compression,
    /// append a "$" to the filename.
    /// </p>
    /// <para>There are several special filenames:</para>
    /// <list type="bullet">
    /// <item><description>
    /// The special names "nul" and "null" refer to an empty stream.
    /// </description></item>
    /// <item><description>
    /// The special names "-" and "$" refer to the console output. Note that this is fully buffered,
    /// which may cause problems if the data is needed immediately
    /// (<see cref="ZStreamWriter.Open(string)"/> uses line buffering).
    /// </description></item>
    /// <item><description>
    /// Names starting with "cosmos://" are stored as Cosmos streams.
    /// </description></item>
    /// <item><description>
    /// The name "clip:" refers to the clipboard, for writing text.
    /// </description></item>
    /// <item><description>
    /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
    /// </description></item>
    /// <item><description>
    /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
    /// list of other files.
    /// </description></item>
    /// <item><description>
    /// Names ending with ":streamname" open the NTFS named stream "streamname".
    /// </description></item>
    /// </list>
    /// <para>
    /// Compression support relies on executable utilities to be in the path.
    /// See <see href="http://7-zip.org"/> for 7z.exe and 7za.exe (for
    /// .7z, .gz), <see href="http://gnuwin32.sourceforge.net/packages/gzip.htm"/> for gzip.exe
    /// (for .gz).
    /// </para>
    /// </remarks>
#else
    /// <summary>
    /// Class to create output Streams given file paths.
    /// </summary>
#endif
    public class ZStreamOut
    {
        private static int _compressionLevel = 6;
        private static bool _defaultLowFragmentation = true;
        private static bool _breakChunksAtLines = false;

        private static int _bufferSize = 32 * 1024;  //64*1024;  //-1;  //64*1024; //32768;

        /// <summary>
        /// Get or set the compression level (0 - 9) used for compressed streams.
        /// </summary>
        /// <remarks>
        /// <p>
        /// The default is 1, which is the worst (but fastest) compression.
        /// Setting a higher level can significantly improve the compression ratio,
        /// especially for tighter compression methods (such as 7z), but the time
        /// needed will increase.
        /// </p>
        /// <p>
        /// If file size is a problem, raising this value can help.
        /// </p>
        /// <p>
        /// A setting of 0 represents storing without compression for methods that
        /// support this.
        /// </p>
        /// </remarks>
        public static int CompressionLevel
        {
            get { return _compressionLevel; }
            set { _compressionLevel = Math.Max(0, Math.Min(9, value)); }
        }

        /// <summary>
        /// Get or set whether the Open method should use a LowFragmentationStream for files.
        /// true, by default.
        /// </summary>
        /// <remarks>
        /// The <see cref="LowFragmentationStream"/> has strong advantages, increasing write
        /// speed and decreasing fragmentation.
        /// </remarks>
        public static bool DefaultLowFragmentation
        {
            get { return _defaultLowFragmentation; }
            set { _defaultLowFragmentation = value; }
        }

#if TLCFULLBUILD
        /// <summary>
        /// Get or set whether to break at line boundaries when using chunked streams,
        /// such as <see cref="CosmosWriteStream"/> or <see cref="MultiStream"/>. False by default,
        /// unlike <see cref="ZStreamOut"/>.
        /// </summary>
        /// <remarks>
        /// This will not necessarily have any effect. It is currently unimplemented in
        /// <see cref="MultiStream"/>, and most streams have no concept of chunks.
        /// </remarks>
#else
        /// <summary>
        /// Get or set whether to break at line boundaries when using chunked streams.
        /// False by default.
        /// </summary>
        /// <remarks>
        /// This will not necessarily have any effect. Most streams have no concept of chunks.
        /// </remarks>
#endif
        public static bool BreakChunksAtLines
        {
            get { return _breakChunksAtLines; }
            set { _breakChunksAtLines = value; }
        }

        internal static readonly string[] compressionArchiveExtensions = new string[]
            {
                // 7za:
                ".7z"
                //".zip",  // broken
                //".tar"   // broken
            };
        internal static readonly string[] compressionExtensions = new string[]
            {
                // 7za:
                ".gz",
                ".7z",
                //".zip",  // broken
                //".tar",  // broken
                //".bzip2",
                ".bz2",
            };

        /// <summary>
        /// Gets the set of extensions (such as ".gz") that are accepted for compression.
        /// </summary>
        public static string[] CompressionExtensions
        {
            get
            {
                return (string[])compressionExtensions.Clone();
            }
        }

        /// <summary>
        /// Gets the set of extensions (such as ".7z") that are accepted for multi-file
        /// archive compression.
        /// </summary>
        public static string[] CompressionArchiveExtensions
        {
            get
            {
                return (string[])compressionArchiveExtensions.Clone();
            }
        }

        /// <summary>
        /// Get whether to allow fallback to the compression library if executables
        /// are not found in the path. false by default. Using the fallback may result in
        /// slower performance and larger files. This setting is shared with ZStreamIn,
        /// ZStreamOut, ZStreamReader, and ZStreamWriter.
        /// </summary>
        public static bool AllowLibraryFallback
        {
            get { return ZStreamIn.AllowLibraryFallback; }
            set { ZStreamIn.AllowLibraryFallback = value; }
        }

        private static readonly char[] _pathSeparators = new char[] { '/', '\\' };

        private ZStreamOut()
        {
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is fully buffered,
        /// which may cause problems if the data is needed immediately
        /// (<see cref="ZStreamWriter.Open(string)"/> uses line buffering).
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
#endif
        public static Stream Open(string fileName)
        {
            return Open(fileName, false);
        }

#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is fully buffered,
        /// which may cause problems if the data is needed immediately
        /// (<see cref="ZStreamWriter.Open(string)"/> uses line buffering).
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid, or appending cannot be done.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
#endif
        public static Stream Open(string fileName, bool append)
        {
            return Open(fileName, append, BreakChunksAtLines);
        }
        public static Stream Open(string fileName, bool append, out string pathFull)
        {
            return Open(fileName, append, BreakChunksAtLines, out pathFull);
        }
#if TLCFULLBUILD
        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="breakChunksAtLines">if true, break at line boundaries when using chunked streams</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <p>
        /// Compressed files are recognized by extension and automatically compressed.
        /// To write to a file with a compression extension directly, without compression,
        /// append a "$" to the filename.
        /// </p>
        /// <para>There are several special filenames:</para>
        /// <list type="bullet">
        /// <item><description>
        /// The special names "nul" and "null" refer to an empty stream.
        /// </description></item>
        /// <item><description>
        /// The special names "-" and "$" refer to the console output. Note that this is fully buffered,
        /// which may cause problems if the data is needed immediately
        /// (<see cref="ZStreamWriter.Open(string)"/> uses line buffering).
        /// </description></item>
        /// <item><description>
        /// Names starting with "cosmos://" are stored as Cosmos streams.
        /// </description></item>
        /// <item><description>
        /// The name "clip:" refers to the clipboard, for writing text.
        /// </description></item>
        /// <item><description>
        /// The form "sql:server/db/table" refers to SQL Server, for an existing table.
        /// </description></item>
        /// <item><description>
        /// "multi:filename" or "filelist:filename" refers to a <see cref="MultiStream"/>
        /// list of other files.
        /// </description></item>
        /// <item><description>
        /// Names ending with ":streamname" open the NTFS named stream "streamname".
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid, or appending cannot be done.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
#endif
        public static Stream Open(string fileName, bool append, bool breakChunksAtLines)
        {
            string pathFull;
            return Open(fileName, append, breakChunksAtLines, true, out pathFull);
        }
        public static Stream Open(string fileName, bool append, bool breakChunksAtLines, out string pathFull)
        {
            return Open(fileName, append, breakChunksAtLines, true, out pathFull);
        }

        /// <summary>
        /// Open the given file, accepting special stream names and decompressing by extension.
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="buffered">if true, use buffered IO; if false, use unbuffered IO</param>
        /// <param name="breakChunksAtLines">if true, break at line boundaries when using chunked streams</param>
        /// <returns>A Stream for the file</returns>
        /// <exception cref="ArgumentNullException">fileName is null.</exception>
        /// <exception cref="ArgumentException">fileName is invalid, or appending cannot be done.</exception>
        /// <exception cref="InvalidOperationException">The utilities needed to open a stream are not available.</exception>
        /// <exception cref="FileNotFoundException">Append is specified, and fileName cannot be found.</exception>
        private static Stream Open(string fileName, bool append, bool breakChunksAtLines, bool buffered, out string pathFull)
        {
            Contracts.CheckNonEmpty(fileName, nameof(fileName));

            pathFull = null;

            // check for special names:
            string fileNameLower = fileName.ToLower();
#if TLCFULLBUILD
            if (ZStreamIn.IsNullStream(fileName))
            {
                //return new NullStream(true);
                return Stream.Null;
            }
            if (ZStreamIn.IsConsoleStream(fileName))
            {
                // buffer size is ignored for console streams!
                // terrible perf for ReadByte(), by default - too many managed->unmanaged transitions.
                // buffering, however, leads to problems with timing for some applications that expect
                // streaming results.

                //return Console.OpenStandardOutput();

                LineBufferedStream res = new LineBufferedStream(Console.OpenStandardOutput(), _bufferSize);
                res.LineBuffer = false;
                return res;
                //return new BufferedStream(Console.OpenStandardOutput(), BUFFER_SIZE);

                //return new LineBufferedStream(
                //new FileStream(
                //    new Microsoft.Win32.SafeHandles.SafeFileHandle(IOUtil.Win32.GetStdHandle(-11), false),
                //    FileAccess.Write, BUFFER_SIZE, false)
                //    );

                //return new FileStream(
                //    new Microsoft.Win32.SafeHandles.SafeFileHandle(IOUtil.Win32.GetStdHandle(-11), false),
                //    FileAccess.Write, BUFFER_SIZE, false);

                //return new FileStream(
                //    new Microsoft.Win32.SafeHandles.SafeFileHandle(IOUtil.Win32.GetStdHandle(-11), false),
                //    FileAccess.Write, 1, false);
            }
            // check for clipboard:
            if (ClipboardReadStream.IsClipboardStream(fileName))
            {
                return new ClipboardWriteStream(append);
            }

            // check for Cosmos:
            if (fileNameLower.StartsWith("cosmos:"))
            {
                try
                {
                    // no compression support, anyway...
                    fileName = fileName.TrimEnd('$');
                    return new CosmosWriteStream(fileName, append, breakChunksAtLines);
                }
                catch (InvalidOperationException)
                {
                    throw new InvalidOperationException("ZStreamIn requires cosmos.cmd to be in the " +
                        "path for Cosmos reading.");
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    throw new InvalidOperationException("ZStreamIn requires cosmos.cmd to be in the " +
                        "path for Cosmos reading.");
                }
            }

            // check for Multistream:
            if (fileNameLower.StartsWith("multi:"))
            {
                fileName = fileName.Substring("multi:".Length);
                return new MultiStream(fileName, true);
            }
            if (fileNameLower.StartsWith("filelist:"))
            {
                fileName = fileName.Substring("filelist:".Length);
                return new MultiStream(fileName, true);
            }

            if (fileNameLower.StartsWith("http:") || fileNameLower.StartsWith("https:"))
            {
                AzureStorageIO azureStorage = new AzureStorageIO();
                return azureStorage.GetBlobStreamForWriting(fileNameLower);

            }

            // check for SqlStream:
            if (SqlTextReader.IsSqlTextReader(fileName))
            {
                return new SqlTextWriter(fileName).CreateStream();
            }

            // check for InternalStore:
            if (InternalStoreUtility.IsInternalStore(fileName))
            {
                throw new NotSupportedException("Cannot write to a InternalStore.");
            }

            // remove trailing "$"
            bool forceRaw = false;
            if (fileName[fileName.Length - 1] == '$')
            {
                forceRaw = true;
                fileName = fileName.Substring(0, fileName.Length - 1);
            }

            // check for named stream:
            // should this be based on file existance, first?
            int cIndex = fileName.LastIndexOf(':');
            if (cIndex > 0)
            {
                if (cIndex > 1 ||
                    fileName.IndexOfAny(_pathSeparators, 2) < 0)
                {
                    // named:
                    return new NamedStream(fileName, true, append);
                }
            }

            Stream s = null;
            if (!forceRaw)
            {
                string ext = Path.GetExtension(fileName).ToLower();
                // use special cases for efficiency:
                switch (ext)
                {
                case ".gz":
                    {
                        // appending really could be enabled... ***
                        //bool failed = false;
                        if (append)
                            throw new ArgumentException("Cannot append to a gz stream.", "append");
                        try
                        {
                            s = new GzipEncodeStream(fileName, CompressionLevel);
                        }
                        catch (InvalidOperationException)
                        {
                            //failed = true;
                        }
                        catch (System.ComponentModel.Win32Exception)
                        {
                            //failed = true;
                        }
                        if (s == null)
                        {
                            try
                            {
                                s = new Z7zEncodeStream(fileName, Z7zEncodeStream.CompressionFormat.Gzip, CompressionLevel);
                            }
                            catch (InvalidOperationException)
                            {
                                if (!AllowLibraryFallback)
                                {
                                    // should fallback here if SharpZip is used!! ***
                                    throw new InvalidOperationException("ZStreamOut requires 7za.exe, 7z.exe, or gzip.exe to be in the " +
                                        "path for " + ext + " compression, unless AllowLibraryFallback is set. " +
                                        "See http://7-zip.org");
                                }
                            }
                            catch (System.ComponentModel.Win32Exception)
                            {
                                if (!AllowLibraryFallback)
                                {
                                    // should fallback here if SharpZip is used!! ***
                                    throw new InvalidOperationException("ZStreamOut requires 7za.exe, 7z.exe, or gzip.exe to be in the " +
                                        "path for " + ext + " compression, unless AllowLibraryFallback is set. " +
                                        "See http://7-zip.org");
                                }
                            }
                        }
                        if (AllowLibraryFallback && s == null)
                        {
                            // this could support unbuffered, Cosmos, etc... ***
                            //// NOTE:
                            ////   - .NET's gzip is very slow (30% or more longer)
                            ////   - .NET's gzip is very large (30% larger compressed files)
                            ////   - .NET's gzip breaks for files over 4 GB
                            ////   - Huffman trees are hard-coded and poor - binary data often *inflates*
                            ////   - Position and Length are not supported
                            ////   - Seeking is not supported
                            ////   - Compression and decompression are not parallelized
                            s = new System.IO.Compression.GZipStream(
                                //new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Read),
                                ZStreamOut.Open(fileName + "$"),
                                System.IO.Compression.CompressionMode.Compress);
                            // without a buffer, this is even worse...
                            s = new BufferedStream(s, 32 * 1024);
                        }
                    }
                    break;

                //case ".gz":
                //case ".zip":  // broken!!
                //case ".tar":  // broken!!
                case ".7z":
                //case ".bzip2":
                case ".bz2":
                    {
                        if (append)
                            throw new ArgumentException("Cannot append to a 7z stream.", "append");
                        try
                        {
                            s = new Z7zEncodeStream(fileName, CompressionLevel);
                        }
                        catch (InvalidOperationException)
                        {
                            throw new InvalidOperationException("ZStreamOut requires 7z.exe or 7za.exe to be in the " +
                                "path for " + ext + " compression. " +
                                "See http://7-zip.org");
                        }
                        catch (System.ComponentModel.Win32Exception)
                        {
                            throw new InvalidOperationException("ZStreamOut requires 7z.exe or 7za.exe to be in the " +
                                "path for " + ext + " compression. " +
                                "See http://7-zip.org");
                        }
                    }
                    break;

#if ENABLE_LZMA
                    case ".lzma":
                        {
                            if (append)  throw new ArgumentException("Cannot append to an lzma stream.", "append");
                            try
                            {
                                s = new LzmaEncodeStream(fileName, true);
                            }
                            catch (InvalidOperationException)
                            {
                                throw new InvalidOperationException("ZStreamOut requires lzma.exe to be in the " +
                                    "path for " + ext + " compression.");
                            }
                            catch (System.ComponentModel.Win32Exception)
                            {
                                throw new InvalidOperationException("ZStreamOut requires lzma.exe to be in the " +
                                    "path for " + ext + " compression.");
                            }
                        }
                        break;
#endif
                }

                if (s == null)
                {
                    // check for compressed archive as directory segment:
                    // *** TODO !
                    // check for compressed archives in path:
                    // only one path segment is allowed to be an archive...
                    // normalize path:
                    //if (fileName[fileName.Length - 1] != '\\')  fileName = fileName + "\\";
                    string zfileName = fileName.Replace('/', '\\');
                    bool isUnc = zfileName.StartsWith("\\\\");
                    while (zfileName.IndexOf("\\\\") >= 0)
                    {
                        zfileName = zfileName.Replace("\\\\", "\\");
                    }
                    if (isUnc)
                        zfileName = "\\" + zfileName;
                    string zfileNameLower = zfileName.ToLower();

                    string archPath = null;
                    string inArch = null;
                    // this could be a problem... We can't write to non-archives in this way...
                    for (int i = 0; i < ZStreamIn.decompressionArchiveExtensions.Length; i++)
                    {
                        ext = ZStreamIn.decompressionArchiveExtensions[i];
                        int seg = zfileNameLower.IndexOf(ext + "\\");
                        if (seg > 0)
                        {
                            archPath = zfileName.Substring(0, seg + ext.Length);
                            inArch = zfileName.Substring(seg + ext.Length).Trim('/', '\\');
                            break;
                        }
                    }
                    if (archPath != null)
                    {
                        //Console.WriteLine(archPath + " :: " + inArch);
                        // check for path in archive.
                        inArch = inArch.Trim('\\');
                        if (inArch.Length != 0)
                        {
                            // what about unrar, etc? ****
                            try
                            {
                                s = new Z7zEncodeStream(archPath, inArch, CompressionLevel);
                            }
                            catch (InvalidOperationException)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe or 7za.exe to be in the " +
                                    "path for " + ext + " compression. " +
                                    "See http://7-zip.org");
                            }
                            catch (System.ComponentModel.Win32Exception)
                            {
                                throw new InvalidOperationException("ZStreamIn requires 7z.exe or 7za.exe to be in the " +
                                    "path for " + ext + " compression. " +
                                    "See http://7-zip.org");
                            }
                        }
                    }
                }
            }
#else
            Stream s = null;
#endif
            if (s == null)
            {
                // Report the full path.
                pathFull = Path.GetFullPath(fileName);

                if (DefaultLowFragmentation)
                {
                    if (_bufferSize > 0)
                    {
                        s = new LowFragmentationStream(fileName, append, _bufferSize);
                    }
                    else
                    {
                        s = new LowFragmentationStream(fileName, append);
                    }
                }
                else
                {
                    if (_bufferSize > 0)
                    {
                        s = new FileStream(fileName, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read, _bufferSize);
                    }
                    else
                    {
                        s = new FileStream(fileName, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read);
                    }
                }
            }
            return s;
        }

#if UNBUFFERED

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        public static Stream OpenUnbuffered(string fileName)
        {
            return OpenUnbuffered(fileName, false);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        public static Stream OpenUnbuffered(string fileName, bool append)
        {
            return OpenUnbuffered(fileName, append, BreakChunksAtLines);
        }

        /// <summary>
        /// Open the given file (unbuffered, if possible).
        /// </summary>
        /// <param name="fileName">file to write to</param>
        /// <param name="append">if true, append; if false, overwrite</param>
        /// <param name="breakChunksAtLines">if true, break at line boundaries when using chunked streams</param>
        /// <returns>A Stream for the file</returns>
        /// <remarks>
        /// <para>
        /// Unbuffered I/O can give better performance, especially on fast RAID arrays.
        /// It does not use the system file cache. However, for writing, this currently
        /// has no effect.
        /// </para>
        /// </remarks>
        public static Stream OpenUnbuffered(string fileName, bool append, bool breakChunksAtLines)
        {
            string pathFull;
            return Open(fileName, append, breakChunksAtLines, false, out pathFull);
        }

#endif
    }

    #endregion

    #region Var
    /// <summary>
    /// Convenience type to represent a value that should be easily converted.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Var instance are immutable.
    /// </para>
    /// <para>
    /// In the current form, Var is not very efficient - the value is always stored
    /// internally as a string, and it is a reference type.
    /// </para>
    /// <para>
    /// A Var will implicitly convert to and from primitive numerical types, as well as
    /// <see cref="Guid"/>, <see cref="DateTime"/>, and <see cref="String"/>. It can also be compared for
    /// equality or order with those types. Comparisons are performed by converting the
    /// Var to the type of the other value (for example, comparing to 123 will cause a
    /// numerical comparison, while "123" will cause a string comparison).
    /// </para>
    /// <para>
    /// Numbers will be parsed properly if they contain commas, are in exponential notation,
    /// or have surrounding whitespace. Hex numbers can be specified by starting with "0x"
    /// or "-0x".
    /// </para>
    /// <para>
    /// String methods can be used directly, without casting the value.
    /// </para>
    /// </remarks>
    public class Var :
        IEquatable<Var>, IEquatable<string>, IEquatable<double>, IEquatable<float>,
        IEquatable<int>, IEquatable<uint>, IEquatable<long>, IEquatable<ulong>,
        IEquatable<short>, IEquatable<ushort>, IEquatable<byte>, IEquatable<sbyte>,
        IEquatable<char>, IEquatable<decimal>, IEquatable<Guid>, IEquatable<DateTime>,
        IComparable<Var>, IComparable<string>, IComparable<double>, IComparable<float>,
        IComparable<int>, IComparable<uint>, IComparable<long>, IComparable<ulong>,
        IComparable<short>, IComparable<ushort>, IComparable<byte>, IComparable<sbyte>,
        IComparable<char>, IComparable<decimal>, IComparable<Guid>, IComparable<DateTime>,
        IEnumerable<char>,
        IComparable, IEnumerable, IConvertible
    {
        private string _raw;

        /// <summary>
        /// Create a new Var, based on the string representation.
        /// </summary>
        /// <param name="raw">the string representation</param>
        public Var(string raw)
        {
            _raw = raw;
        }

        /// <summary>
        /// Convert the specified Var to a string.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the string version of the value</returns>
        public static implicit operator string(Var v)
        {
            return v._raw;
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="s">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(string s)
        {
            return new Var(s);
        }

        /// <summary>
        /// Convert the specified Var to an int.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the int version of the value</returns>
        public static implicit operator int(Var v)
        {
            return ParseInt32(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(int n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a uint.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the uint version of the value</returns>
        public static implicit operator uint(Var v)
        {
            return ParseUInt32(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(uint n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a long.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the long version of the value</returns>
        public static implicit operator long(Var v)
        {
            return ParseInt64(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(long n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a ulong.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the ulong version of the value</returns>
        public static implicit operator ulong(Var v)
        {
            return ParseUInt64(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(ulong n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a short.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the short version of the value</returns>
        public static implicit operator short(Var v)
        {
            return ParseInt16(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(short n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a ushort.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the ushort version of the value</returns>
        public static implicit operator ushort(Var v)
        {
            return ParseUInt16(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(ushort n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a byte.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the byte version of the value</returns>
        public static implicit operator byte(Var v)
        {
            return ParseByte(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(byte n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a sbyte.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the sbyte version of the value</returns>
        public static implicit operator sbyte(Var v)
        {
            return ParseSByte(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(sbyte n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a float.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the float version of the value</returns>
        public static implicit operator float(Var v)
        {
            return ParseSingle(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(float n)
        {
            return new Var(n.ToString("R"));
        }

        /// <summary>
        /// Convert the specified Var to a double.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the double version of the value</returns>
        public static implicit operator double(Var v)
        {
            return ParseDouble(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(double n)
        {
            return new Var(n.ToString("R"));
        }

        /// <summary>
        /// Convert the specified Var to a char.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the char version of the value</returns>
        public static implicit operator char(Var v)
        {
            return v._raw == null || v._raw.Length == 0 ? '\0' : v._raw[0];
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="c">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(char c)
        {
            return new Var(c.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a decimal.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the decimal version of the value</returns>
        public static implicit operator decimal(Var v)
        {
            return v._raw == null || v._raw.Length == 0 ? decimal.Zero : decimal.Parse(v._raw);
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="n">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(decimal n)
        {
            return new Var(n.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a bool.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the bool version of the value</returns>
        public static implicit operator bool(Var v)
        {
            return v._raw != null && v._raw != "0" && string.Compare(v._raw, "false", true) != 0;
        }
        /// <summary>
        /// Convert the specified value to a Var.
        /// </summary>
        /// <param name="b">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(bool b)
        {
            return new Var(b ? "1" : null);
        }

        /// <summary>
        /// Convert the specified Var to a Guid.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the Guid version of the value</returns>
        public static implicit operator Guid(Var v)
        {
            //return v.raw == null || v.raw.Length == 0 ? Guid.Empty : new Guid(v.raw);
            if (v._raw == null || v._raw.Length == 0)
                return Guid.Empty;

            string s = v._raw;
            if (s.Length == 32)
            {
                // xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                s = s.Substring(0, 8) + "-" + s.Substring(8, 4) + "-" + s.Substring(12, 4) + "-" + s.Substring(16, 4) + "-" + s.Substring(20);
                // OK, maybe that's not the most efficient...
            }
            return new Guid(s);
        }
        /// <summary>
        /// Convert the specified Guid to a Var.
        /// </summary>
        /// <param name="g">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(Guid g)
        {
            return new Var(g.ToString());
        }

        /// <summary>
        /// Convert the specified Var to a DateTime.
        /// </summary>
        /// <param name="v">the value to convert</param>
        /// <returns>the DateTime version of the value</returns>
        public static implicit operator DateTime(Var v)
        {
            return v._raw == null || v._raw.Length == 0 ? DateTime.MinValue :
                DateTime.Parse(v._raw, null,
                System.Globalization.DateTimeStyles.AllowWhiteSpaces |
                System.Globalization.DateTimeStyles.NoCurrentDateDefault
                );
        }
        /// <summary>
        /// Convert the specified DateTime to a Var.
        /// </summary>
        /// <param name="d">the value to convert</param>
        /// <returns>the Var version of the value</returns>
        public static implicit operator Var(DateTime d)
        {
            return new Var(d.ToString());
        }

        /// <summary>
        /// Convert an array of values.
        /// </summary>
        /// <param name="vals">the array to convert</param>
        /// <returns>the converted array</returns>
        public static string[] Convert(Var[] vals)
        {
            if (vals == null)
                return null;
            string[] res = new string[vals.Length];
            for (int i = 0; i < vals.Length; i++)
            {
                res[i] = vals[i] == null ? null : (string)vals[i];
            }
            return res;
        }
        /// <summary>
        /// Convert an array of values.
        /// </summary>
        /// <param name="vals">the array to convert</param>
        /// <returns>the converted array</returns>
        public static Var[] Convert(string[] vals)
        {
            if (vals == null)
                return null;
            Var[] res = new Var[vals.Length];
            for (int i = 0; i < vals.Length; i++)
            {
                res[i] = new Var(vals[i]);
            }
            return res;
        }

        /// <summary>
        /// Return the string representation of this value.
        /// </summary>
        /// <returns>the string representation of this value</returns>
        public override string ToString()
        {
            // leave null?
            return _raw;
        }

        private static long ParseInt64(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToInt64(s, 16);
                }
                if (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X'))
                {
                    return -System.Convert.ToInt64(s.Substring(3), 16);
                }
                return long.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static ulong ParseUInt64(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToUInt64(s, 16);
                }
                return ulong.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static int ParseInt32(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToInt32(s, 16);
                }
                if (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X'))
                {
                    checked
                    {
                        return (int)(-System.Convert.ToInt64(s.Substring(3), 16));
                    }
                }
                return int.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static uint ParseUInt32(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToUInt32(s, 16);
                }
                return uint.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static short ParseInt16(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToInt16(s, 16);
                }
                if (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X'))
                {
                    checked
                    {
                        return (short)(-System.Convert.ToInt32(s.Substring(3), 16));
                    }
                }
                return short.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static ushort ParseUInt16(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToUInt16(s, 16);
                }
                return ushort.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static sbyte ParseSByte(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToSByte(s, 16);
                }
                if (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X'))
                {
                    checked
                    {
                        return (sbyte)(-System.Convert.ToInt32(s.Substring(3), 16));
                    }
                }
                return sbyte.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static byte ParseByte(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0;
            s = s.Trim();
            if (s.Length == 0)
                return 0;
            try
            {
                if (s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                {
                    return System.Convert.ToByte(s, 16);
                }
                return byte.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static float ParseSingle(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0.0F;
            s = s.Trim();
            if (s.Length == 0)
                return 0.0F;
            try
            {
                if ((s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ||
                    (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X')))
                {
                    checked
                    {
                        return (float)ParseInt64(s);
                    }
                }
                return float.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }
        private static double ParseDouble(string s)
        {
            // could be many times faster...
            if (s == null)
                return 0.0;
            s = s.Trim();
            if (s.Length == 0)
                return 0.0;
            try
            {
                if ((s.Length > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ||
                    (s.Length > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X')))
                {
                    checked
                    {
                        return (double)ParseInt64(s);
                    }
                }
                return double.Parse(s, System.Globalization.NumberStyles.AllowThousands | System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint);
            }
            catch (Exception ex)
            {
                if (ex is InvalidCastException || ex is FormatException || ex is OverflowException)
                    throw Contracts.ExceptDecode("String cannot be converted to integer: '{0}'", s);
                throw;
            }
        }

        /// <summary>Returns the hash code for this value.</summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            // this may be bad - numbers have many string representations... ***
            return _raw == null ? 0 : _raw.GetHashCode();
        }

        #region Comparisons
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public override bool Equals(object obj)
        {
            if (obj is Var)
            {
                return Equals((Var)obj);
            }
            else if (obj is string)
            {
                return Equals((string)obj);
            }
            else if (obj is double)
            {
                return Equals((double)obj);
            }
            else if (obj is float)
            {
                return Equals((float)obj);
            }
            else if (obj is int)
            {
                return Equals((int)obj);
            }
            else if (obj is uint)
            {
                return Equals((uint)obj);
            }
            else if (obj is long)
            {
                return Equals((long)obj);
            }
            else if (obj is ulong)
            {
                return Equals((ulong)obj);
            }
            else if (obj is short)
            {
                return Equals((short)obj);
            }
            else if (obj is ushort)
            {
                return Equals((ushort)obj);
            }
            else if (obj is byte)
            {
                return Equals((byte)obj);
            }
            else if (obj is sbyte)
            {
                return Equals((sbyte)obj);
            }
            else if (obj is char)
            {
                return Equals((char)obj);
            }
            else if (obj is decimal)
            {
                return Equals((decimal)obj);
            }
            else if (obj is Guid)
            {
                return Equals((Guid)obj);
            }
            else if (obj is DateTime)
            {
                return Equals((DateTime)obj);
            }
            return false;
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(Var obj)
        {
            if (_raw == null)
            {
                return obj == null || obj._raw == null;
            }
            return Equals(obj._raw);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(string obj)
        {
            return string.CompareOrdinal(_raw, obj) == 0;
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(double obj)
        {
            return ((double)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(float obj)
        {
            return ((double)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(int obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(uint obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(long obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(ulong obj)
        {
            return ((ulong)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(short obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(ushort obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(byte obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(sbyte obj)
        {
            return ((long)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(char obj)
        {
            return ((char)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(decimal obj)
        {
            return ((decimal)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(Guid obj)
        {
            return ((Guid)this).Equals(obj);
        }
        /// <summary>Determines whether this instance and a specified value are the same.</summary>
        /// <returns>true if obj is the same as this instance; otherwise, false.</returns>
        /// <param name="obj">the value to compare to</param>
        public bool Equals(DateTime obj)
        {
            return ((DateTime)this).Equals(obj);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, Var v2)
        {
            // this may be bad - numbers have many string representations... ***
            if ((object)v1 == null)
                return (object)v2 == null;
            if ((object)v2 == null)
                return false;
            return v1.Equals(v2);
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, long v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, long v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(long v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(long v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, ulong v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (ulong)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, ulong v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(ulong v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(ulong v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, int v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, int v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(int v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(int v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, uint v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, uint v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(uint v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(uint v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, short v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, short v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(short v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(short v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, ushort v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, ushort v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(ushort v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(ushort v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, byte v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, byte v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(byte v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(byte v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, sbyte v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (long)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, sbyte v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(sbyte v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(sbyte v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, decimal v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (decimal)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, decimal v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(decimal v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(decimal v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, float v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (double)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, float v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(float v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(float v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, double v2)
        {
            if ((object)v1 == null)
                return v2 == 0;
            return (double)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, double v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(double v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(double v1, Var v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, Guid v2)
        {
            if ((object)v1 == null)
                return v2 == Guid.Empty;
            return (Guid)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, Guid v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(Var v1, DateTime v2)
        {
            if ((object)v1 == null)
                return v2 == DateTime.MinValue;
            return (DateTime)v1 == v2;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(Var v1, DateTime v2)
        {
            return !(v1 == v2);
        }
        /// <summary>Determines whether the specified values are the same.</summary>
        /// <returns>true if v1 is the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator ==(DateTime v1, Var v2)
        {
            return v2 == v1;
        }
        /// <summary>Determines whether the specified values are not the same.</summary>
        /// <returns>true if v1 is not the same as v2; otherwise, false.</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator !=(DateTime v1, Var v2)
        {
            return !(v1 == v2);
        }

        int IComparable.CompareTo(object obj)
        {
            if (obj == null)
                return _raw == null ? 0 : 1;
            if (obj is Var)
            {
                return CompareTo((Var)obj);
            }
            else if (obj is string)
            {
                return CompareTo((string)obj);
            }
            else if (obj is double)
            {
                return CompareTo((double)obj);
            }
            else if (obj is float)
            {
                return CompareTo((float)obj);
            }
            else if (obj is int)
            {
                return CompareTo((int)obj);
            }
            else if (obj is uint)
            {
                return CompareTo((uint)obj);
            }
            else if (obj is long)
            {
                return CompareTo((long)obj);
            }
            else if (obj is ulong)
            {
                return CompareTo((ulong)obj);
            }
            else if (obj is short)
            {
                return CompareTo((short)obj);
            }
            else if (obj is ushort)
            {
                return CompareTo((ushort)obj);
            }
            else if (obj is byte)
            {
                return CompareTo((byte)obj);
            }
            else if (obj is sbyte)
            {
                return CompareTo((sbyte)obj);
            }
            else if (obj is char)
            {
                return CompareTo((char)obj);
            }
            else if (obj is decimal)
            {
                return CompareTo((decimal)obj);
            }
            else if (obj is Guid)
            {
                return CompareTo((Guid)obj);
            }
            else if (obj is DateTime)
            {
                return CompareTo((DateTime)obj);
            }
            return -1;
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(Var obj)
        {
            // possibly should use numerical if both are numerical...
            if (obj == null)
                return _raw == null ? 0 : 1;
            return CompareTo(obj._raw);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(string obj)
        {
            return string.CompareOrdinal(_raw, obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(double obj)
        {
            return ((double)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(float obj)
        {
            return ((double)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(int obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(uint obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(long obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(ulong obj)
        {
            return ((ulong)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(short obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(ushort obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(byte obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(sbyte obj)
        {
            return ((long)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(char obj)
        {
            return ((char)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(decimal obj)
        {
            return ((decimal)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(Guid obj)
        {
            return ((Guid)this).CompareTo(obj);
        }
        /// <summary>Compares this instance with a specified value.</summary>
        /// <returns>
        /// A 32-bit signed integer indicating the lexical relationship between the two comparands.
        /// negative if this instance is less than obj, zero if this instance is equal to obj,
        /// positive if this instance is greater than obj.
        /// </returns>
        /// <param name="obj">the value to compare to</param>
        public int CompareTo(DateTime obj)
        {
            return ((DateTime)this).CompareTo(obj);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, Var v2)
        {
            if ((object)v2 == null)
                return false;
            if ((object)v1 == null)
                return v2._raw != null;
            return v1.CompareTo(v2) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, Var v2)
        {
            if ((object)v1 == null)
                return false;
            if ((object)v2 == null)
                return v1._raw != null;
            return v2.CompareTo(v1) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, long v2)
        {
            long vv1 = ((object)v1 == null) ? 0 : (long)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, long v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, long v2)
        {
            long vv1 = ((object)v1 == null) ? 0 : (long)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, long v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(long v1, Var v2)
        {
            long vv2 = ((object)v2 == null) ? 0 : (long)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(long v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(long v1, Var v2)
        {
            long vv2 = ((object)v2 == null) ? 0 : (long)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(long v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, ulong v2)
        {
            ulong vv1 = ((object)v1 == null) ? 0 : (ulong)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, ulong v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, ulong v2)
        {
            ulong vv1 = ((object)v1 == null) ? 0 : (ulong)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, ulong v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(ulong v1, Var v2)
        {
            ulong vv2 = ((object)v2 == null) ? 0 : (ulong)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(ulong v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(ulong v1, Var v2)
        {
            ulong vv2 = ((object)v2 == null) ? 0 : (ulong)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(ulong v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, int v2)
        {
            int vv1 = ((object)v1 == null) ? 0 : (int)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, int v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, int v2)
        {
            int vv1 = ((object)v1 == null) ? 0 : (int)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, int v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(int v1, Var v2)
        {
            int vv2 = ((object)v2 == null) ? 0 : (int)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(int v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(int v1, Var v2)
        {
            int vv2 = ((object)v2 == null) ? 0 : (int)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(int v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, uint v2)
        {
            uint vv1 = ((object)v1 == null) ? 0 : (uint)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, uint v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, uint v2)
        {
            uint vv1 = ((object)v1 == null) ? 0 : (uint)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, uint v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(uint v1, Var v2)
        {
            uint vv2 = ((object)v2 == null) ? 0 : (uint)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(uint v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(uint v1, Var v2)
        {
            uint vv2 = ((object)v2 == null) ? 0 : (uint)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(uint v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, short v2)
        {
            short vv1 = ((object)v1 == null) ? (short)0 : (short)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, short v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, short v2)
        {
            short vv1 = ((object)v1 == null) ? (short)0 : (short)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, short v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(short v1, Var v2)
        {
            short vv2 = ((object)v2 == null) ? (short)0 : (short)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(short v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(short v1, Var v2)
        {
            short vv2 = ((object)v2 == null) ? (short)0 : (short)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(short v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, ushort v2)
        {
            ushort vv1 = ((object)v1 == null) ? (ushort)0 : (ushort)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, ushort v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, ushort v2)
        {
            ushort vv1 = ((object)v1 == null) ? (ushort)0 : (ushort)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, ushort v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(ushort v1, Var v2)
        {
            ushort vv2 = ((object)v2 == null) ? (ushort)0 : (ushort)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(ushort v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(ushort v1, Var v2)
        {
            ushort vv2 = ((object)v2 == null) ? (ushort)0 : (ushort)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(ushort v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, byte v2)
        {
            byte vv1 = ((object)v1 == null) ? (byte)0 : (byte)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, byte v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, byte v2)
        {
            byte vv1 = ((object)v1 == null) ? (byte)0 : (byte)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, byte v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(byte v1, Var v2)
        {
            byte vv2 = ((object)v2 == null) ? (byte)0 : (byte)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(byte v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(byte v1, Var v2)
        {
            byte vv2 = ((object)v2 == null) ? (byte)0 : (byte)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(byte v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, sbyte v2)
        {
            sbyte vv1 = ((object)v1 == null) ? (sbyte)0 : (sbyte)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, sbyte v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, sbyte v2)
        {
            sbyte vv1 = ((object)v1 == null) ? (sbyte)0 : (sbyte)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, sbyte v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(sbyte v1, Var v2)
        {
            sbyte vv2 = ((object)v2 == null) ? (sbyte)0 : (sbyte)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(sbyte v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(sbyte v1, Var v2)
        {
            sbyte vv2 = ((object)v2 == null) ? (sbyte)0 : (sbyte)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(sbyte v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, decimal v2)
        {
            decimal vv1 = ((object)v1 == null) ? 0 : (decimal)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, decimal v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, decimal v2)
        {
            decimal vv1 = ((object)v1 == null) ? 0 : (decimal)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, decimal v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(decimal v1, Var v2)
        {
            decimal vv2 = ((object)v2 == null) ? 0 : (decimal)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(decimal v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(decimal v1, Var v2)
        {
            decimal vv2 = ((object)v2 == null) ? 0 : (decimal)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(decimal v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, Guid v2)
        {
            Guid vv1 = ((object)v1 == null) ? Guid.Empty : (Guid)v1;
            return vv1.CompareTo(v2) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, Guid v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, Guid v2)
        {
            Guid vv1 = ((object)v1 == null) ? Guid.Empty : (Guid)v1;
            return vv1.CompareTo(v2) > 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, Guid v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Guid v1, Var v2)
        {
            Guid vv2 = ((object)v2 == null) ? Guid.Empty : (Guid)v2;
            return v1.CompareTo(vv2) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Guid v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Guid v1, Var v2)
        {
            Guid vv2 = ((object)v2 == null) ? Guid.Empty : (Guid)v2;
            return v1.CompareTo(vv2) > 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Guid v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, DateTime v2)
        {
            DateTime vv1 = ((object)v1 == null) ? DateTime.MinValue : (DateTime)v1;
            return vv1.CompareTo(v2) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, DateTime v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, DateTime v2)
        {
            DateTime vv1 = ((object)v1 == null) ? DateTime.MinValue : (DateTime)v1;
            return vv1.CompareTo(v2) > 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, DateTime v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(DateTime v1, Var v2)
        {
            DateTime vv2 = ((object)v2 == null) ? DateTime.MinValue : (DateTime)v2;
            return v1.CompareTo(vv2) < 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(DateTime v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(DateTime v1, Var v2)
        {
            DateTime vv2 = ((object)v2 == null) ? DateTime.MinValue : (DateTime)v2;
            return v1.CompareTo(vv2) > 0;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(DateTime v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, float v2)
        {
            float vv1 = ((object)v1 == null) ? 0 : (float)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, float v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, float v2)
        {
            float vv1 = ((object)v1 == null) ? 0 : (float)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, float v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(float v1, Var v2)
        {
            float vv2 = ((object)v2 == null) ? 0 : (float)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(float v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(float v1, Var v2)
        {
            float vv2 = ((object)v2 == null) ? 0 : (float)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(float v1, Var v2)
        {
            return !(v1 > v2);
        }

        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(Var v1, double v2)
        {
            double vv1 = ((object)v1 == null) ? 0 : (double)v1;
            return vv1 < v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(Var v1, double v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(Var v1, double v2)
        {
            double vv1 = ((object)v1 == null) ? 0 : (double)v1;
            return vv1 > v2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(Var v1, double v2)
        {
            return !(v1 > v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <(double v1, Var v2)
        {
            double vv2 = ((object)v2 == null) ? 0 : (double)v2;
            return v1 < vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >=(double v1, Var v2)
        {
            return !(v1 < v2);
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &gt; v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator >(double v1, Var v2)
        {
            double vv2 = ((object)v2 == null) ? 0 : (double)v2;
            return v1 > vv2;
        }
        /// <summary>Compares two values.</summary>
        /// <returns>true if v1 &lt;= v2; false otherwise</returns>
        /// <param name="v1">the first value</param>
        /// <param name="v2">the second value</param>
        public static bool operator <=(double v1, Var v2)
        {
            return !(v1 > v2);
        }
        #endregion

        /// <summary>
        /// Negate the value.
        /// </summary>
        /// <param name="v">the value to negate</param>
        /// <returns>the negated value</returns>
        public static Var operator -(Var v)
        {
            if (v == null)
                return null;
            if (v._raw == null || v._raw.Length == 0)
                return new Var(v._raw);
            string s = v._raw.Trim();
            if (s.Length != 0)
            {
                s = (s[0] == '-') ? s.Substring(1) : "-" + s;
            }
            return new Var(s);
        }

        /// <summary>
        /// Negate the value.
        /// </summary>
        /// <param name="v">the value to negate</param>
        /// <returns>the negated value</returns>
        public static Var operator !(Var v)
        {
            if (v == null)
                return null;
            return (Var)(!((bool)v));
        }

        /// <summary>
        /// Increment the value.
        /// </summary>
        /// <param name="v">the value to increment</param>
        /// <returns>the incremented value</returns>
        public static Var operator ++(Var v)
        {
            if (v == null)
                return null;
            if (v._raw == null || v._raw.Length == 0 || v._raw == "0")
                return (Var)1;
            // how to guess the type? ***
            //char?
            // ugly...
            if (v._raw.Length == 1 && (v._raw[0] < '0' || v._raw[0] > '9'))
            {
                return (Var)((char)v + 1);
            }
            // decimal? **
            // ulong? **
            // integer?
            Var res = (long)v + 1;
            return res;
            // double? **
        }
        /// <summary>
        /// Decrement the value.
        /// </summary>
        /// <param name="v">the value to decrement</param>
        /// <returns>the decremented value</returns>
        public static Var operator --(Var v)
        {
            if (v == null)
                return null;
            if (v._raw == null || v._raw.Length == 0 || v._raw == "0")
                return (Var)(-1);
            // how to guess the type? ***
            //char?
            // ugly...
            if (v._raw.Length == 1 && (v._raw[0] < '0' || v._raw[0] > '9'))
            {
                return (Var)((char)v - 1);
            }
            // decimal? **
            // ulong? **
            // integer?
            return (Var)((long)v - 1);
            // double? **
        }

        #region String Methods
        /// <summary>Concatenates a specified separator <see cref="System.String"></see> between each element of a specified <see cref="Var"></see> array, yielding a single concatenated string.</summary>
        /// <returns>A <see cref="System.String"></see> consisting of the elements of value interspersed with the separator string.</returns>
        /// <param name="separator">A <see cref="System.String"></see>. </param>
        /// <param name="value">An array of <see cref="Var"></see>. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public static string Join(string separator, Var[] value)
        {
            Contracts.CheckValue(value, nameof(value));
            return Join(separator, value, 0, value.Length);
        }

        /// <summary>Concatenates a specified separator <see cref="System.String"></see> between each element of a specified <see cref="Var"></see> array, yielding a single concatenated string. Parameters specify the first array element and number of elements to use.</summary>
        /// <returns>A <see cref="System.String"></see> object consisting of the strings in value joined by separator. Or, <see cref="F:System.String.Empty"></see> if count is zero, value has no elements, or separator and all the elements of value are <see cref="F:System.String.Empty"></see>.</returns>
        /// <param name="count">The number of elements of value to use. </param>
        /// <param name="separator">A <see cref="System.String"></see>. </param>
        /// <param name="value">An array of <see cref="Var"></see>. </param>
        /// <param name="startIndex">The first array element in value to use. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex or count is less than 0.-or- startIndex plus count is greater than the number of elements in value. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public static unsafe string Join(string separator, Var[] value, int startIndex, int count)
        {
            if (separator == null)
                separator = "";
            Contracts.CheckValue(value, nameof(value));
            if (startIndex < 0)
                throw Contracts.ExceptParam(nameof(startIndex), "Must be non-negative.");
            if (count < 0)
                throw Contracts.ExceptParam(nameof(count), "Must be non-negative.");
            if (startIndex > value.Length - count)
                throw Contracts.ExceptParam(nameof(startIndex));
            if (count == 0)
                return "";
            if (separator.Length == 0 && startIndex == 0 && count == value.Length)
                return Concat(value);
            int length = 0;
            int num2 = (startIndex + count) - 1;
            for (int i = startIndex; i <= num2; i++)
            {
                if (value[i] != null)
                {
                    string s = (string)value[i];
                    if (s != null)
                        length += s.Length;
                }
            }
            length += (count - 1) * separator.Length;
            if ((length < 0) || ((length + 1) < 0))
                throw Contracts.Process(new InsufficientMemoryException());
            if (length == 0)
                return "";
            StringBuilder sb = new StringBuilder(length);
            if (value[startIndex] != null)
                sb.Append((string)value[startIndex]);
            for (int j = startIndex + 1; j <= num2; j++)
            {
                sb.Append(separator);
                if (value[j] != null)
                    sb.Append((string)value[j]);
            }
            return sb.ToString();
        }

        /// <summary>Concatenates the elements of a specified <see cref="Var"></see> array.</summary>
        /// <returns>The concatenated elements of values.</returns>
        /// <param name="values">An array of <see cref="Var"></see> instances. </param>
        /// <exception cref="System.ArgumentNullException">values is null. </exception>
        public static string Concat(params Var[] values)
        {
            return string.Concat(Convert(values));
        }

        /// <summary>Returns a value indicating whether the specified <see cref="System.String"></see> object occurs within this string.</summary>
        /// <returns>true if the value parameter occurs within this string, or if value is the empty string (""); otherwise, false.</returns>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool Contains(string value) { return _raw.Contains(value); }

        /// <summary>Copies a specified number of characters from a specified position in this instance to a specified position in an array of Unicode characters.</summary>
        /// <param name="count">The number of characters in this instance to copy to destination. </param>
        /// <param name="destinationIndex">An array element in destination. </param>
        /// <param name="sourceIndex">A character position in this instance. </param>
        /// <param name="destination">An array of Unicode characters. </param>
        /// <exception cref="System.ArgumentNullException">destination is null. </exception>
        /// <exception cref="System.ArgumentOutOfRangeException">sourceIndex, destinationIndex, or count is negative -or- count is greater than the length of the substring from startIndex to the end of this instance -or- count is greater than the length of the subarray from destinationIndex to the end of destination</exception>
        public void CopyTo(int sourceIndex, char[] destination, int destinationIndex, int count) { _raw.CopyTo(sourceIndex, destination, destinationIndex, count); }

        /// <summary>Determines whether the end of this instance matches the specified string.</summary>
        /// <returns>true if value matches the end of this instance; otherwise, false.</returns>
        /// <param name="value">A <see cref="System.String"></see> to compare to. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool EndsWith(string value) { return _raw.EndsWith(value); }

        /// <summary>Determines whether the end of this string matches the specified string when compared using the specified comparison option.</summary>
        /// <returns>true if the value parameter matches the end of this string; otherwise, false.</returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values that determines how this string and value are compared. </param>
        /// <param name="value">A <see cref="System.String"></see> object to compare to. </param>
        /// <exception cref="System.ArgumentException">comparisonType is not a <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool EndsWith(string value, StringComparison comparisonType) { return _raw.EndsWith(value, comparisonType); }

        /// <summary>Determines whether the end of this string matches the specified string when compared using the specified culture.</summary>
        /// <returns>true if the value parameter matches the end of this string; otherwise, false.</returns>
        /// <param name="culture">Cultural information that determines how this instance and value are compared. If culture is null, the current culture is used.</param>
        /// <param name="ignoreCase">true to ignore case when comparing this instance and value; otherwise, false.</param>
        /// <param name="value">A <see cref="System.String"></see> object to compare to. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool EndsWith(string value, bool ignoreCase, System.Globalization.CultureInfo culture) { return _raw.EndsWith(value, ignoreCase, culture); }

        /// <summary>Determines whether this string and a specified <see cref="System.String"></see> object have the same value. A parameter specifies the culture, case, and sort rules used in the comparison.</summary>
        /// <returns>true if the value of the value parameter is the same as this string; otherwise, false.</returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">A <see cref="System.String"></see> object.</param>
        /// <exception cref="System.NullReferenceException">This string is null. </exception>
        /// <exception cref="System.ArgumentException">comparisonType is not a <see cref="System.StringComparison"></see> value. </exception>
        public bool Equals(string value, StringComparison comparisonType) { return _raw.Equals(value, comparisonType); }

        /// <summary>Retrieves an object that can iterate through the individual characters in this string.</summary>
        /// <returns>A <see cref="System.CharEnumerator"></see> object.</returns>
        public CharEnumerator GetEnumerator() { return _raw.GetEnumerator(); }

        /// <summary>Reports the index of the first occurrence of the specified Unicode character in this string.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="value">A Unicode character to seek. </param>
        public int IndexOf(char value) { return _raw.IndexOf(value); }

        /// <summary>Reports the index of the first occurrence of the specified <see cref="System.String"></see> in this instance.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is 0.</returns>
        /// <param name="value">The <see cref="System.String"></see> to seek. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value) { return _raw.IndexOf(value); }

        /// <summary>Reports the index of the first occurrence of the specified Unicode character in this string. The search starts at a specified character position.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="value">A Unicode character to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or specifies a position beyond the end of this instance. </exception>
        public int IndexOf(char value, int startIndex) { return _raw.IndexOf(value, startIndex); }

        /// <summary>Reports the index of the first occurrence of the specified <see cref="System.String"></see> in this instance. The search starts at a specified character position.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is startIndex.</returns>
        /// <param name="value">The <see cref="System.String"></see> to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is negative.-or- startIndex specifies a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value, int startIndex) { return _raw.IndexOf(value, startIndex); }

        /// <summary>Reports the index of the first occurrence of the specified string in the current <see cref="System.String"></see> object. A parameter specifies the type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is 0.</returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value, StringComparison comparisonType) { return _raw.IndexOf(value, comparisonType); }

        /// <summary>Reports the index of the first occurrence of the specified character in this instance. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="value">A Unicode character to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count + startIndex specifies a position beyond the end of this instance. </exception>
        public int IndexOf(char value, int startIndex, int count) { return _raw.IndexOf(value, startIndex, count); }

        /// <summary>Reports the index of the first occurrence of the specified <see cref="System.String"></see> in this instance. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is startIndex.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="value">The <see cref="System.String"></see> to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count plus startIndex specify a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value, int startIndex, int count) { return _raw.IndexOf(value, startIndex, count); }

        /// <summary>Reports the index of the first occurrence of the specified string in the current <see cref="System.String"></see> object. Parameters specify the starting search position in the current string and the type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is 0.</returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is negative, or specifies a position that is not within this instance. </exception>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value, int startIndex, StringComparison comparisonType) { return _raw.IndexOf(value, startIndex, comparisonType); }

        /// <summary>Reports the index of the first occurrence of the specified string in the current <see cref="System.String"></see> object. Parameters specify the starting search position in the current string, the number of characters in the current string to search, and the type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is 0.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count plus startIndex specify a position that is not within this instance. </exception>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int IndexOf(string value, int startIndex, int count, StringComparison comparisonType) { return _raw.IndexOf(value, startIndex, count, comparisonType); }

        /// <summary>Reports the index of the first occurrence in this instance of any character in a specified array of Unicode characters.</summary>
        /// <returns>The index position of the first occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        /// <filterpriority>2</filterpriority>
        public int IndexOfAny(char[] anyOf) { return _raw.IndexOfAny(anyOf); }

        /// <summary>Reports the index of the first occurrence in this instance of any character in a specified array of Unicode characters. The search starts at a specified character position.</summary>
        /// <returns>The index position of the first occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is negative.-or- startIndex is greater than the number of characters in this instance. </exception>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        public int IndexOfAny(char[] anyOf, int startIndex) { return _raw.IndexOfAny(anyOf, startIndex); }

        /// <summary>Reports the index of the first occurrence in this instance of any character in a specified array of Unicode characters. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of the first occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count + startIndex is greater than the number of characters in this instance. </exception>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        public int IndexOfAny(char[] anyOf, int startIndex, int count) { return _raw.IndexOfAny(anyOf, startIndex, count); }

        /// <summary>Inserts a specified instance of <see cref="System.String"></see> at a specified index position in this instance.</summary>
        /// <returns>A new <see cref="System.String"></see> equivalent to this instance but with value inserted at position startIndex.</returns>
        /// <param name="value">The <see cref="System.String"></see> to insert. </param>
        /// <param name="startIndex">The index position of the insertion. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is negative or greater than the length of this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public string Insert(int startIndex, string value) { return _raw.Insert(startIndex, value); }

        /// <summary>Indicates whether this string is in Unicode normalization form C.</summary>
        /// <returns>true if this string is in normalization form C; otherwise, false.</returns>
        public bool IsNormalized() { return _raw.IsNormalized(); }

        /// <summary>Indicates whether this string is in the specified Unicode normalization form.</summary>
        /// <returns>true if this string is in the normalization form specified by the normalizationForm parameter; otherwise, false.</returns>
        /// <param name="normalizationForm">A Unicode normalization form. </param>
        public bool IsNormalized(NormalizationForm normalizationForm) { return _raw.IsNormalized(normalizationForm); }

        /// <summary>Reports the index position of the last occurrence of a specified Unicode character within this instance.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="value">A Unicode character to seek. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(char value) { return _raw.LastIndexOf(value); }

        /// <summary>Reports the index position of the last occurrence of a specified <see cref="System.String"></see> within this instance.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is the last index position in value.</returns>
        /// <param name="value">A <see cref="System.String"></see> to seek. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value) { return _raw.LastIndexOf(value); }

        /// <summary>Reports the index position of the last occurrence of a specified Unicode character within this instance. The search starts at a specified character position.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="value">A Unicode character to seek. </param>
        /// <param name="startIndex">The starting position of a substring within this instance. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or greater than the length of this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(char value, int startIndex) { return _raw.LastIndexOf(value, startIndex); }

        /// <summary>Reports the index position of the last occurrence of a specified <see cref="System.String"></see> within this instance. The search starts at a specified character position.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is startIndex.</returns>
        /// <param name="value">The <see cref="System.String"></see> to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or specifies a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value, int startIndex) { return _raw.LastIndexOf(value, startIndex); }

        /// <summary>Reports the index of the last occurrence of a specified string within the current <see cref="System.String"></see> object. A parameter specifies the type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. </returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value, StringComparison comparisonType) { return _raw.LastIndexOf(value, comparisonType); }

        /// <summary>Reports the index position of the last occurrence of the specified Unicode character in a substring within this instance. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of value if that character is found, or -1 if it is not.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="value">A Unicode character to seek. </param>
        /// <param name="startIndex">The starting position of a substring within this instance. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex or count is less than zero or greater than the length of this instance. </exception>
        public int LastIndexOf(char value, int startIndex, int count) { return _raw.LastIndexOf(value, startIndex, count); }
        /// <summary>Reports the index position of the last occurrence of a specified <see cref="System.String"></see> within this instance. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of value if that string is found, or -1 if it is not. If value is <see cref="F:System.String.Empty"></see>, the return value is startIndex.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="value">The <see cref="System.String"></see> to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count plus startIndex specify a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value, int startIndex, int count) { return _raw.LastIndexOf(value, startIndex, count); }

        /// <summary>Reports the index of the last occurrence of a specified string within the current <see cref="System.String"></see> object. Parameters specify the starting search position in the current string, and type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. </returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or specifies a position that is not within this instance. </exception>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value, int startIndex, StringComparison comparisonType) { return _raw.LastIndexOf(value, startIndex, comparisonType); }

        /// <summary>Reports the index position of the last occurrence of a specified <see cref="System.String"></see> object within this instance. Parameters specify the starting search position in the current string, the number of characters in the current string to search, and the type of search to use for the specified string.</summary>
        /// <returns>The index position of the value parameter if that string is found, or -1 if it is not. </returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values. </param>
        /// <param name="value">The <see cref="System.String"></see> object to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count plus startIndex specify a position that is not within this instance. </exception>
        /// <exception cref="System.ArgumentException">comparisonType is not a valid <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public int LastIndexOf(string value, int startIndex, int count, StringComparison comparisonType) { return _raw.LastIndexOf(value, startIndex, count, comparisonType); }

        /// <summary>Reports the index position of the last occurrence in this instance of one or more characters specified in a Unicode array.</summary>
        /// <returns>The index position of the last occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        public int LastIndexOfAny(char[] anyOf) { return _raw.LastIndexOfAny(anyOf); }

        /// <summary>Reports the index position of the last occurrence in this instance of one or more characters specified in a Unicode array. The search starts at a specified character position.</summary>
        /// <returns>The index position of the last occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex specifies a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        public int LastIndexOfAny(char[] anyOf, int startIndex) { return _raw.LastIndexOfAny(anyOf, startIndex); }

        /// <summary>Reports the index position of the last occurrence in this instance of one or more characters specified in a Unicode array. The search starts at a specified character position and examines a specified number of character positions.</summary>
        /// <returns>The index position of the last occurrence in this instance where any character in anyOf was found; otherwise, -1 if no character in anyOf was found.</returns>
        /// <param name="count">The number of character positions to examine. </param>
        /// <param name="anyOf">A Unicode character array containing one or more characters to seek. </param>
        /// <param name="startIndex">The search starting position. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count or startIndex is negative.-or- count plus startIndex specify a position not within this instance. </exception>
        /// <exception cref="System.ArgumentNullException">anyOf is null. </exception>
        public int LastIndexOfAny(char[] anyOf, int startIndex, int count) { return _raw.LastIndexOfAny(anyOf, startIndex, count); }

        /// <summary>Returns a new string whose textual value is the same as this string, but whose binary representation is in Unicode normalization form C.</summary>
        /// <returns>A new, normalized string whose textual value is the same as this string, but whose binary representation is in normalization form C.</returns>
        public string Normalize() { return _raw.Normalize(); }

        /// <summary>Returns a new string whose textual value is the same as this string, but whose binary representation is in the specified Unicode normalization form.</summary>
        /// <returns>A new string whose textual value is the same as this string, but whose binary representation is in the normalization form specified by the normalizationForm parameter.</returns>
        /// <param name="normalizationForm">A Unicode normalization form. </param>
        public string Normalize(NormalizationForm normalizationForm) { return _raw.Normalize(normalizationForm); }

        /// <summary>Right-aligns the characters in this instance, padding with spaces on the left for a specified total length.</summary>
        /// <returns>A new <see cref="System.String"></see> that is equivalent to this instance, but right-aligned and padded on the left with as many spaces as needed to create a length of totalWidth. Or, if totalWidth is less than the length of this instance, a new <see cref="System.String"></see> object that is identical to this instance.</returns>
        /// <param name="totalWidth">The number of characters in the resulting string, equal to the number of original characters plus any additional padding characters. </param>
        /// <exception cref="System.ArgumentException">totalWidth is less than zero. </exception>
        public string PadLeft(int totalWidth) { return _raw.PadLeft(totalWidth); }

        /// <summary>Right-aligns the characters in this instance, padding on the left with a specified Unicode character for a specified total length.</summary>
        /// <returns>A new <see cref="System.String"></see> that is equivalent to this instance, but right-aligned and padded on the left with as many paddingChar characters as needed to create a length of totalWidth. Or, if totalWidth is less than the length of this instance, a new <see cref="System.String"></see> that is identical to this instance.</returns>
        /// <param name="paddingChar">A Unicode padding character. </param>
        /// <param name="totalWidth">The number of characters in the resulting string, equal to the number of original characters plus any additional padding characters. </param>
        /// <exception cref="System.ArgumentException">totalWidth is less than zero. </exception>
        public string PadLeft(int totalWidth, char paddingChar) { return _raw.PadLeft(totalWidth, paddingChar); }

        /// <summary>Left-aligns the characters in this string, padding with spaces on the right, for a specified total length.</summary>
        /// <returns>A new <see cref="System.String"></see> that is equivalent to this instance, but left-aligned and padded on the right with as many spaces as needed to create a length of totalWidth. Or, if totalWidth is less than the length of this instance, a new <see cref="System.String"></see> that is identical to this instance.</returns>
        /// <param name="totalWidth">The number of characters in the resulting string, equal to the number of original characters plus any additional padding characters. </param>
        /// <exception cref="System.ArgumentException">totalWidth is less than zero. </exception>
        public string PadRight(int totalWidth) { return _raw.PadRight(totalWidth); }

        /// <summary>Left-aligns the characters in this string, padding on the right with a specified Unicode character, for a specified total length.</summary>
        /// <returns>A new <see cref="System.String"></see> that is equivalent to this instance, but left-aligned and padded on the right with as many paddingChar characters as needed to create a length of totalWidth. Or, if totalWidth is less than the length of this instance, a new <see cref="System.String"></see> that is identical to this instance.</returns>
        /// <param name="paddingChar">A Unicode padding character. </param>
        /// <param name="totalWidth">The number of characters in the resulting string, equal to the number of original characters plus any additional padding characters. </param>
        /// <exception cref="System.ArgumentException">totalWidth is less than zero. </exception>
        public string PadRight(int totalWidth, char paddingChar) { return _raw.PadRight(totalWidth, paddingChar); }

        /// <summary>Deletes all the characters from this string beginning at a specified position and continuing through the last position.</summary>
        /// <returns>A new <see cref="System.String"></see> object that is equivalent to this string less the removed characters.</returns>
        /// <param name="startIndex">The position to begin deleting characters. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero.-or- startIndex specifies a position that is not within this string. </exception>
        public string Remove(int startIndex) { return _raw.Remove(startIndex); }

        /// <summary>Deletes a specified number of characters from this instance beginning at a specified position.</summary>
        /// <returns>A new <see cref="System.String"></see> that is equivalent to this instance less count number of characters.</returns>
        /// <param name="count">The number of characters to delete. </param>
        /// <param name="startIndex">The position to begin deleting characters. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">Either startIndex or count is less than zero.-or- startIndex plus count specify a position outside this instance. </exception>
        public string Remove(int startIndex, int count) { return _raw.Remove(startIndex, count); }

        /// <summary>Replaces all occurrences of a specified Unicode character in this instance with another specified Unicode character.</summary>
        /// <returns>A <see cref="System.String"></see> equivalent to this instance but with all instances of oldChar replaced with newChar.</returns>
        /// <param name="newChar">A Unicode character to replace all occurrences of oldChar. </param>
        /// <param name="oldChar">A Unicode character to be replaced. </param>
        public string Replace(char oldChar, char newChar) { return _raw.Replace(oldChar, newChar); }

        /// <summary>Replaces all occurrences of a specified <see cref="System.String"></see> in this instance, with another specified <see cref="System.String"></see>.</summary>
        /// <returns>A <see cref="System.String"></see> equivalent to this instance but with all instances of oldValue replaced with newValue.</returns>
        /// <param name="oldValue">A <see cref="System.String"></see> to be replaced. </param>
        /// <param name="newValue">A <see cref="System.String"></see> to replace all occurrences of oldValue. </param>
        /// <exception cref="System.ArgumentNullException">oldValue is null. </exception>
        /// <exception cref="System.ArgumentException">oldValue is the empty string (""). </exception>
        public string Replace(string oldValue, string newValue) { return _raw.Replace(oldValue, newValue); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this instance that are delimited by elements of a specified <see cref="System.Char"></see> array.</summary>
        /// <returns>An array whose elements contain the substrings in this instance that are delimited by one or more characters in separator. For more information, see the Remarks section.</returns>
        /// <param name="separator">An array of Unicode characters that delimit the substrings in this instance, an empty array containing no delimiters, or null. </param>
        public string[] Split(params char[] separator) { return _raw.Split(separator); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this instance that are delimited by elements of a specified <see cref="System.Char"></see> array. A parameter specifies the maximum number of substrings to return.</summary>
        /// <returns>An array whose elements contain the substrings in this instance that are delimited by one or more characters in separator. For more information, see the Remarks section.</returns>
        /// <param name="count">The maximum number of substrings to return. </param>
        /// <param name="separator">An array of Unicode characters that delimit the substrings in this instance, an empty array containing no delimiters, or null. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">count is negative. </exception>
        public string[] Split(char[] separator, int count) { return _raw.Split(separator, count); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this string that are delimited by elements of a specified <see cref="System.Char"></see> array. A parameter specifies whether to return empty array elements.</summary>
        /// <returns>An array whose elements contain the substrings in this string that are delimited by one or more characters in separator. For more information, see the Remarks section.</returns>
        /// <param name="options">Specify <see cref="F:System.StringSplitOptions.RemoveEmptyEntries"></see> to omit empty array elements from the array returned, or <see cref="F:System.StringSplitOptions.None"></see> to include empty array elements in the array returned. </param>
        /// <param name="separator">An array of Unicode characters that delimit the substrings in this string, an empty array containing no delimiters, or null. </param>
        /// <exception cref="System.ArgumentException">options is not one of the <see cref="System.StringSplitOptions"></see> values.</exception>
        public string[] Split(char[] separator, StringSplitOptions options) { return _raw.Split(separator, options); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this string that are delimited by elements of a specified <see cref="System.String"></see> array. A parameter specifies whether to return empty array elements.</summary>
        /// <returns>An array whose elements contain the substrings in this string that are delimited by one or more strings in separator. For more information, see the Remarks section.</returns>
        /// <param name="options">Specify <see cref="F:System.StringSplitOptions.RemoveEmptyEntries"></see> to omit empty array elements from the array returned, or <see cref="F:System.StringSplitOptions.None"></see> to include empty array elements in the array returned. </param>
        /// <param name="separator">An array of strings that delimit the substrings in this string, an empty array containing no delimiters, or null. </param>
        /// <exception cref="System.ArgumentException">options is not one of the <see cref="System.StringSplitOptions"></see> values.</exception>
        public string[] Split(string[] separator, StringSplitOptions options) { return _raw.Split(separator, options); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this string that are delimited by elements of a specified <see cref="System.Char"></see> array. Parameters specify the maximum number of substrings to return and whether to return empty array elements.</summary>
        /// <returns>An array whose elements contain the substrings in this stringthat are delimited by one or more characters in separator. For more information, see the Remarks section.</returns>
        /// <param name="count">The maximum number of substrings to return. </param>
        /// <param name="options">Specify <see cref="F:System.StringSplitOptions.RemoveEmptyEntries"></see> to omit empty array elements from the array returned, or <see cref="F:System.StringSplitOptions.None"></see> to include empty array elements in the array returned. </param>
        /// <param name="separator">An array of Unicode characters that delimit the substrings in this string, an empty array containing no delimiters, or null. </param>
        /// <exception cref="System.ArgumentException">options is not one of the <see cref="System.StringSplitOptions"></see> values.</exception>
        /// <exception cref="System.ArgumentOutOfRangeException">count is negative. </exception>
        public string[] Split(char[] separator, int count, StringSplitOptions options) { return _raw.Split(separator, count, options); }

        /// <summary>Returns a <see cref="System.String"></see> array containing the substrings in this string that are delimited by elements of a specified <see cref="System.String"></see> array. Parameters specify the maximum number of substrings to return and whether to return empty array elements.</summary>
        /// <returns>An array whose elements contain the substrings in this string that are delimited by one or more strings in separator. For more information, see the Remarks section.</returns>
        /// <param name="count">The maximum number of substrings to return. </param>
        /// <param name="options">Specify <see cref="F:System.StringSplitOptions.RemoveEmptyEntries"></see> to omit empty array elements from the array returned, or <see cref="F:System.StringSplitOptions.None"></see> to include empty array elements in the array returned. </param>
        /// <param name="separator">An array of strings that delimit the substrings in this string, an empty array containing no delimiters, or null. </param>
        /// <exception cref="System.ArgumentException">options is not one of the <see cref="System.StringSplitOptions"></see> values.</exception>
        /// <exception cref="System.ArgumentOutOfRangeException">count is negative. </exception>
        public string[] Split(string[] separator, int count, StringSplitOptions options) { return _raw.Split(separator, count, options); }

        /// <summary>Determines whether the beginning of this instance matches the specified string.</summary>
        /// <returns>true if value matches the beginning of this string; otherwise, false.</returns>
        /// <param name="value">The <see cref="System.String"></see> to compare. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool StartsWith(string value) { return _raw.StartsWith(value); }

        /// <summary>Determines whether the beginning of this string matches the specified string when compared using the specified comparison option.</summary>
        /// <returns>true if the value parameter matches the beginning of this string; otherwise, false.</returns>
        /// <param name="comparisonType">One of the <see cref="System.StringComparison"></see> values that determines how this string and value are compared. </param>
        /// <param name="value">A <see cref="System.String"></see> object to compare to. </param>
        /// <exception cref="System.ArgumentException">comparisonType is not a <see cref="System.StringComparison"></see> value.</exception>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool StartsWith(string value, StringComparison comparisonType) { return _raw.StartsWith(value, comparisonType); }

        /// <summary>Determines whether the beginning of this string matches the specified string when compared using the specified culture.</summary>
        /// <returns>true if the value parameter matches the beginning of this string; otherwise, false.</returns>
        /// <param name="culture">Cultural information that determines how this string and value are compared. If culture is null, the current culture is used.</param>
        /// <param name="ignoreCase">true to ignore case when comparing this string and value; otherwise, false.</param>
        /// <param name="value">The <see cref="System.String"></see> object to compare. </param>
        /// <exception cref="System.ArgumentNullException">value is null. </exception>
        public bool StartsWith(string value, bool ignoreCase, System.Globalization.CultureInfo culture) { return _raw.StartsWith(value, ignoreCase, culture); }

        /// <summary>Retrieves a substring from this instance. The substring starts at a specified character position.</summary>
        /// <returns>A <see cref="System.String"></see> object equivalent to the substring that begins at startIndex in this instance, or <see cref="F:System.String.Empty"></see> if startIndex is equal to the length of this instance.</returns>
        /// <param name="startIndex">The starting character position of a substring in this instance. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex is less than zero or greater than the length of this instance. </exception>
        public string Substring(int startIndex) { return _raw.Substring(startIndex); }

        /// <summary>Retrieves a substring from this instance. The substring starts at a specified character position and has a specified length.</summary>
        /// <returns>A <see cref="System.String"></see> equivalent to the substring of length length that begins at startIndex in this instance, or <see cref="F:System.String.Empty"></see> if startIndex is equal to the length of this instance and length is zero.</returns>
        /// <param name="startIndex">The index of the start of the substring. </param>
        /// <param name="length">The number of characters in the substring. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex plus length indicates a position not within this instance.-or- startIndex or length is less than zero. </exception>
        public string Substring(int startIndex, int length) { return _raw.Substring(startIndex, length); }

        IEnumerator<char> IEnumerable<char>.GetEnumerator() { return ((IEnumerable<char>)_raw).GetEnumerator(); }

        IEnumerator IEnumerable.GetEnumerator() { return ((IEnumerable)_raw).GetEnumerator(); }

        /// <summary>Copies the characters in this instance to a Unicode character array.</summary>
        /// <returns>A Unicode character array whose elements are the individual characters of this instance. If this instance is an empty string, the returned array is empty and has a zero length.</returns>
        public char[] ToCharArray() { return _raw.ToCharArray(); }

        /// <summary>Copies the characters in a specified substring in this instance to a Unicode character array.</summary>
        /// <returns>A Unicode character array whose elements are the length number of characters in this instance starting from character position startIndex.</returns>
        /// <param name="startIndex">The starting position of a substring in this instance. </param>
        /// <param name="length">The length of the substring in this instance. </param>
        /// <exception cref="System.ArgumentOutOfRangeException">startIndex or length is less than zero.-or- startIndex plus length is greater than the length of this instance. </exception>
        public char[] ToCharArray(int startIndex, int length) { return _raw.ToCharArray(startIndex, length); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> converted to lowercase, using the casing rules of the current culture.</summary>
        /// <returns>A <see cref="System.String"></see> in lowercase.</returns>
        public string ToLower() { return _raw.ToLower(); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> converted to lowercase, using the casing rules of the specified culture.</summary>
        /// <returns>A <see cref="System.String"></see> in lowercase.</returns>
        /// <param name="culture">A <see cref="System.Globalization.CultureInfo"></see> object that supplies culture-specific casing rules. </param>
        /// <exception cref="System.ArgumentNullException">culture is null. </exception>
        public string ToLower(System.Globalization.CultureInfo culture) { return _raw.ToLower(culture); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> object converted to lowercase using the casing rules of the invariant culture.</summary>
        /// <returns>A <see cref="System.String"></see> object in lowercase.</returns>
        public string ToLowerInvariant() { return _raw.ToLowerInvariant(); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> converted to uppercase, using the casing rules of the current culture.</summary>
        /// <returns>A <see cref="System.String"></see> in uppercase.</returns>
        public string ToUpper() { return _raw.ToUpper(); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> converted to uppercase, using the casing rules of the specified culture.</summary>
        /// <returns>A <see cref="System.String"></see> in uppercase.</returns>
        /// <param name="culture">A <see cref="System.Globalization.CultureInfo"></see> object that supplies culture-specific casing rules. </param>
        /// <exception cref="System.ArgumentNullException">culture is null. </exception>
        public string ToUpper(System.Globalization.CultureInfo culture) { return _raw.ToUpper(culture); }

        /// <summary>Returns a copy of this <see cref="System.String"></see> object converted to uppercase using the casing rules of the invariant culture.</summary>
        /// <returns>A <see cref="System.String"></see> object in uppercase.</returns>
        public string ToUpperInvariant() { return _raw.ToUpperInvariant(); }

        /// <summary>Removes all occurrences of white space characters from the beginning and end of this instance.</summary>
        /// <returns>A new <see cref="System.String"></see> equivalent to this instance after white space characters are removed from the beginning and end.</returns>
        public string Trim() { return _raw.Trim(); }

        /// <summary>Removes all occurrences of a set of characters specified in an array from the beginning and end of this instance.</summary>
        /// <returns>The <see cref="System.String"></see> that remains after all occurrences of the characters in trimChars are removed from the beginning and end of this instance. If trimChars is null, white space characters are removed instead.</returns>
        /// <param name="trimChars">An array of Unicode characters to be removed or null. </param>
        public string Trim(params char[] trimChars) { return _raw.Trim(trimChars); }

        /// <summary>Removes all occurrences of a set of characters specified in an array from the end of this instance.</summary>
        /// <returns>The <see cref="System.String"></see> that remains after all occurrences of the characters in trimChars are removed from the end. If trimChars is null, white space characters are removed instead.</returns>
        /// <param name="trimChars">An array of Unicode characters to be removed or null. </param>
        public string TrimEnd(params char[] trimChars) { return _raw.TrimEnd(trimChars); }

        /// <summary>Removes all occurrences of a set of characters specified in an array from the beginning of this instance.</summary>
        /// <returns>The <see cref="System.String"></see> that remains after all occurrences of characters in trimChars are removed from the beginning. If trimChars is null, white space characters are removed instead.</returns>
        /// <param name="trimChars">An array of Unicode characters to be removed or null. </param>
        public string TrimStart(params char[] trimChars) { return _raw.TrimStart(trimChars); }

        /// <summary>Gets the character at a specified character position in this instance.</summary>
        /// <returns>A Unicode character.</returns>
        /// <param name="index">A character position in this instance. </param>
        /// <exception cref="System.IndexOutOfRangeException">index is greater than or equal to the length of this object or less than zero. </exception>
        public char this[int index] { get { return _raw[index]; } }

        /// <summary>Gets the number of characters in this instance.</summary>
        /// <returns>The number of characters in this instance.</returns>
        public int Length { get { return _raw.Length; } }
        #endregion

        #region IConvertible Members

        TypeCode IConvertible.GetTypeCode()
        {
            return TypeCode.Object;
        }

        bool IConvertible.ToBoolean(IFormatProvider provider)
        {
            return this;
        }

        byte IConvertible.ToByte(IFormatProvider provider)
        {
            return this;
        }

        char IConvertible.ToChar(IFormatProvider provider)
        {
            return this;
        }

        DateTime IConvertible.ToDateTime(IFormatProvider provider)
        {
            return this;
        }

        decimal IConvertible.ToDecimal(IFormatProvider provider)
        {
            return this;
        }

        double IConvertible.ToDouble(IFormatProvider provider)
        {
            return this;
        }

        short IConvertible.ToInt16(IFormatProvider provider)
        {
            return this;
        }

        int IConvertible.ToInt32(IFormatProvider provider)
        {
            return this;
        }

        long IConvertible.ToInt64(IFormatProvider provider)
        {
            return this;
        }

        sbyte IConvertible.ToSByte(IFormatProvider provider)
        {
            return this;
        }

        float IConvertible.ToSingle(IFormatProvider provider)
        {
            return this;
        }

        string IConvertible.ToString(IFormatProvider provider)
        {
            return this;
        }

        ushort IConvertible.ToUInt16(IFormatProvider provider)
        {
            return this;
        }

        uint IConvertible.ToUInt32(IFormatProvider provider)
        {
            return this;
        }

        ulong IConvertible.ToUInt64(IFormatProvider provider)
        {
            return this;
        }

        object IConvertible.ToType(Type conversionType, IFormatProvider provider)
        {
            //throw new Exception("The method or operation is not implemented.");
            // *** what do we really need here?
            //return this;
            if (conversionType == typeof(Var))
            {
                return (Var)this;
            }
            else if (conversionType == typeof(string))
            {
                return (string)this;
            }
            else if (conversionType == typeof(double))
            {
                return (double)this;
            }
            else if (conversionType == typeof(float))
            {
                return (float)this;
            }
            else if (conversionType == typeof(int))
            {
                return (int)this;
            }
            else if (conversionType == typeof(uint))
            {
                return (uint)this;
            }
            else if (conversionType == typeof(long))
            {
                return (long)this;
            }
            else if (conversionType == typeof(ulong))
            {
                return (ulong)this;
            }
            else if (conversionType == typeof(short))
            {
                return (short)this;
            }
            else if (conversionType == typeof(ushort))
            {
                return (ushort)this;
            }
            else if (conversionType == typeof(byte))
            {
                return (byte)this;
            }
            else if (conversionType == typeof(sbyte))
            {
                return (sbyte)this;
            }
            else if (conversionType == typeof(char))
            {
                return (char)this;
            }
            else if (conversionType == typeof(decimal))
            {
                return (decimal)this;
            }
            else if (conversionType == typeof(Guid))
            {
                return (Guid)this;
            }
            else if (conversionType == typeof(DateTime))
            {
                return (DateTime)this;
            }
            return this;
        }

        #endregion
    }
    #endregion

    #region NoPreamble
    /// <summary>
    /// An encoding based on another encoding but with no preamble (BOM).
    /// </summary>
    public class NoPreambleEncoding : System.Text.Encoding
    {
        private readonly Encoding _baseEncoding;
        private static readonly byte[] _preamble = new byte[0];

        // private backing field for UTF16
        private readonly Encoding _utf16 = new NoPreambleEncoding(Encoding.Unicode);

        /// <summary>
        /// A UTF16 encoding with no preamble.
        /// </summary>
        public Encoding UTF16
        {
            get
            {
                return _utf16;
            }
        }

        // backing field for BigEndianUTF16
        private readonly Encoding _bigEndianUtf16 = new NoPreambleEncoding(Encoding.BigEndianUnicode);

        /// <summary>
        /// A big-endian UTF16 encoding with no preamble.
        /// </summary>
        public Encoding BigEndianUTF16
        {
            get
            {
                return _bigEndianUtf16;
            }
        }

        /// <summary>
        /// Create a new encoding based on the specified encoding but with no preamble.
        /// </summary>
        /// <param name="baseEncoding">the encoding to base this one on</param>
        /// <exception cref="ArgumentNullException">The baseEncoding was null.</exception>
        public NoPreambleEncoding(Encoding baseEncoding)
        {
            Contracts.CheckValue(baseEncoding, nameof(baseEncoding));
            _baseEncoding = baseEncoding;
        }

        /// <summary>
        /// Returns a sequence of bytes that specifies the encoding used (empty, in this case).
        /// </summary>
        /// <returns>
        /// A byte array of length zero.
        /// </returns>
        public override byte[] GetPreamble()
        {
            return _preamble;
        }
        /// <summary>
        /// Gets a name for the current encoding that can be used with mail agent body tags.
        /// </summary>
        /// <returns>A name for the current <see cref="T:System.Text.Encoding"></see> that can be used with mail agent body tags.-or- An empty string (""), if the current <see cref="T:System.Text.Encoding"></see> cannot be used.</returns>
        public override string BodyName
        {
            get
            {
                return _baseEncoding.BodyName;
            }
        }
        /// <summary>
        /// Creates a shallow copy of the current <see cref="T:System.Text.Encoding"></see> object.
        /// </summary>
        /// <returns>
        /// A copy of the current <see cref="T:System.Text.Encoding"></see> object.
        /// </returns>
        public override object Clone()
        {
            return new NoPreambleEncoding((Encoding)_baseEncoding.Clone());
        }
        /// <summary>
        /// Gets the code page identifier of the current <see cref="T:System.Text.Encoding"></see>.
        /// </summary>
        /// <returns>The code page identifier of the current <see cref="T:System.Text.Encoding"></see>.</returns>
        public override int CodePage
        {
            get
            {
                return _baseEncoding.CodePage;
            }
        }
        /// <summary>
        /// Gets the human-readable description of the current encoding.
        /// </summary>
        /// <returns>The human-readable description of the current <see cref="T:System.Text.Encoding"></see>.</returns>
        public override string EncodingName
        {
            get
            {
                return _baseEncoding.EncodingName;
            }
        }
        /// <summary>
        /// Calculates the number of bytes produced by encoding all the characters in the specified character array.
        /// </summary>
        /// <param name="chars">The character array containing the characters to encode.</param>
        /// <returns>
        /// The number of bytes produced by encoding all the characters in the specified character array.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">chars is null. </exception>
        public override int GetByteCount(char[] chars)
        {
            return _baseEncoding.GetByteCount(chars);
        }
        /// <summary>
        /// Calculates the number of bytes produced by encoding the characters in the specified <see cref="T:System.String"></see>.
        /// </summary>
        /// <param name="s">The <see cref="T:System.String"></see> containing the set of characters to encode.</param>
        /// <returns>
        /// The number of bytes produced by encoding the specified characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">s is null. </exception>
        public override int GetByteCount(string s)
        {
            return _baseEncoding.GetByteCount(s);
        }
        /// <summary>
        /// Encodes all the characters in the specified character array into a sequence of bytes.
        /// </summary>
        /// <param name="chars">The character array containing the characters to encode.</param>
        /// <returns>
        /// A byte array containing the results of encoding the specified set of characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">chars is null. </exception>
        public override byte[] GetBytes(char[] chars)
        {
            return _baseEncoding.GetBytes(chars);
        }
        /// <summary>
        /// Encodes a set of characters from the specified character array into a sequence of bytes.
        /// </summary>
        /// <param name="chars">The character array containing the set of characters to encode.</param>
        /// <param name="index">The index of the first character to encode.</param>
        /// <param name="count">The number of characters to encode.</param>
        /// <returns>
        /// A byte array containing the results of encoding the specified set of characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">index or count is less than zero.-or- index and count do not denote a valid range in chars. </exception>
        /// <exception cref="T:System.ArgumentNullException">chars is null. </exception>
        public override byte[] GetBytes(char[] chars, int index, int count)
        {
            return _baseEncoding.GetBytes(chars, index, count);
        }
        /// <summary>
        /// Encodes all the characters in the specified <see cref="T:System.String"></see> into a sequence of bytes.
        /// </summary>
        /// <param name="s">The <see cref="T:System.String"></see> containing the characters to encode.</param>
        /// <returns>
        /// A byte array containing the results of encoding the specified set of characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">s is null. </exception>
        public override byte[] GetBytes(string s)
        {
            return _baseEncoding.GetBytes(s);
        }
        /// <summary>
        /// Encodes a set of characters from the specified <see cref="T:System.String"></see> into the specified byte array.
        /// </summary>
        /// <param name="s">The <see cref="T:System.String"></see> containing the set of characters to encode.</param>
        /// <param name="charIndex">The index of the first character to encode.</param>
        /// <param name="charCount">The number of characters to encode.</param>
        /// <param name="bytes">The byte array to contain the resulting sequence of bytes.</param>
        /// <param name="byteIndex">The index at which to start writing the resulting sequence of bytes.</param>
        /// <returns>
        /// The actual number of bytes written into bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentException">bytes does not have enough capacity from byteIndex to the end of the array to accommodate the resulting bytes. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">charIndex or charCount or byteIndex is less than zero.-or- charIndex and charCount do not denote a valid range in chars.-or- byteIndex is not a valid index in bytes. </exception>
        /// <exception cref="T:System.ArgumentNullException">s is null.-or- bytes is null. </exception>
        public override int GetBytes(string s, int charIndex, int charCount, byte[] bytes, int byteIndex)
        {
            return _baseEncoding.GetBytes(s, charIndex, charCount, bytes, byteIndex);
        }
        /// <summary>
        /// Calculates the number of characters produced by decoding all the bytes in the specified byte array.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <returns>
        /// The number of characters produced by decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        public override int GetCharCount(byte[] bytes)
        {
            return _baseEncoding.GetCharCount(bytes);
        }
        /// <summary>
        /// Decodes all the bytes in the specified byte array into a set of characters.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <returns>
        /// A character array containing the results of decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        public override char[] GetChars(byte[] bytes)
        {
            return _baseEncoding.GetChars(bytes);
        }
        /// <summary>
        /// Decodes a sequence of bytes from the specified byte array into a set of characters.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <param name="index">The index of the first byte to decode.</param>
        /// <param name="count">The number of bytes to decode.</param>
        /// <returns>
        /// A character array containing the results of decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">index or count is less than zero.-or- index and count do not denote a valid range in bytes. </exception>
        public override char[] GetChars(byte[] bytes, int index, int count)
        {
            return _baseEncoding.GetChars(bytes, index, count);
        }
        /// <summary>
        /// Obtains a decoder that converts an encoded sequence of bytes into a sequence of characters.
        /// </summary>
        /// <returns>
        /// A <see cref="T:System.Text.Decoder"></see> that converts an encoded sequence of bytes into a sequence of characters.
        /// </returns>
        public override Decoder GetDecoder()
        {
            return _baseEncoding.GetDecoder();
        }
        /// <summary>
        /// Obtains an encoder that converts a sequence of Unicode characters into an encoded sequence of bytes.
        /// </summary>
        /// <returns>
        /// An <see cref="T:System.Text.Encoder"></see> that converts a sequence of Unicode characters into an encoded sequence of bytes.
        /// </returns>
        public override Encoder GetEncoder()
        {
            return _baseEncoding.GetEncoder();
        }
        /// <summary>
        /// Decodes all the bytes in the specified byte array into a string.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <returns>
        /// A <see cref="T:System.String"></see> containing the results of decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        public override string GetString(byte[] bytes)
        {
            return _baseEncoding.GetString(bytes);
        }
        /// <summary>
        /// Decodes a sequence of bytes from the specified byte array into a string.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <param name="index">The index of the first byte to decode.</param>
        /// <param name="count">The number of bytes to decode.</param>
        /// <returns>
        /// A <see cref="T:System.String"></see> containing the results of decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">index or count is less than zero.-or- index and count do not denote a valid range in bytes. </exception>
        public override string GetString(byte[] bytes, int index, int count)
        {
            return _baseEncoding.GetString(bytes, index, count);
        }
        /// <summary>
        /// Gets a name for the current encoding that can be used with mail agent header tags.
        /// </summary>
        /// <returns>A name for the current <see cref="T:System.Text.Encoding"></see> that can be used with mail agent header tags.-or- An empty string (""), if the current <see cref="T:System.Text.Encoding"></see> cannot be used.</returns>
        public override string HeaderName
        {
            get
            {
                return _baseEncoding.HeaderName;
            }
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding is always normalized, using the specified normalization form.
        /// </summary>
        /// <param name="form">One of the <see cref="T:System.Text.NormalizationForm"></see> values.</param>
        /// <returns>
        /// true if the current <see cref="T:System.Text.Encoding"></see> object is always normalized using the specified <see cref="T:System.Text.NormalizationForm"></see> value; otherwise, false. The default is false.
        /// </returns>
        public override bool IsAlwaysNormalized(NormalizationForm form)
        {
            return _baseEncoding.IsAlwaysNormalized(form);
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding can be used by browser clients for displaying content.
        /// </summary>
        /// <value></value>
        /// <returns>true if the current <see cref="T:System.Text.Encoding"></see> can be used by browser clients for displaying content; otherwise, false.</returns>
        public override bool IsBrowserDisplay
        {
            get
            {
                return _baseEncoding.IsBrowserDisplay;
            }
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding can be used by browser clients for saving content.
        /// </summary>
        /// <value></value>
        /// <returns>true if the current <see cref="T:System.Text.Encoding"></see> can be used by browser clients for saving content; otherwise, false.</returns>
        public override bool IsBrowserSave
        {
            get
            {
                return _baseEncoding.IsBrowserSave;
            }
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding can be used by mail and news clients for displaying content.
        /// </summary>
        /// <value></value>
        /// <returns>true if the current <see cref="T:System.Text.Encoding"></see> can be used by mail and news clients for displaying content; otherwise, false.</returns>
        public override bool IsMailNewsDisplay
        {
            get
            {
                return _baseEncoding.IsMailNewsDisplay;
            }
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding can be used by mail and news clients for saving content.
        /// </summary>
        /// <value></value>
        /// <returns>true if the current <see cref="T:System.Text.Encoding"></see> can be used by mail and news clients for saving content; otherwise, false.</returns>
        public override bool IsMailNewsSave
        {
            get
            {
                return _baseEncoding.IsMailNewsSave;
            }
        }
        /// <summary>
        /// Gets a value indicating whether the current encoding uses single-byte code points.
        /// </summary>
        /// <value></value>
        /// <returns>true if the current <see cref="T:System.Text.Encoding"></see> uses single-byte code points; otherwise, false.</returns>
        public override bool IsSingleByte
        {
            get
            {
                return _baseEncoding.IsSingleByte;
            }
        }
        /// <summary>
        /// Returns a <see cref="T:System.String"></see> that represents the current <see cref="T:System.Object"></see>.
        /// </summary>
        /// <returns>
        /// A <see cref="T:System.String"></see> that represents the current <see cref="T:System.Object"></see>.
        /// </returns>
        public override string ToString()
        {
            return _baseEncoding.ToString();
        }
        /// <summary>
        /// Gets the name registered with the Internet Assigned Numbers Authority (IANA) for the current encoding.
        /// </summary>
        /// <value></value>
        /// <returns>The IANA name for the current <see cref="T:System.Text.Encoding"></see>.</returns>
        public override string WebName
        {
            get
            {
                return _baseEncoding.WebName;
            }
        }
        /// <summary>
        /// Gets the Windows operating system code page that most closely corresponds to the current encoding.
        /// </summary>
        /// <value></value>
        /// <returns>The Windows operating system code page that most closely corresponds to the current <see cref="T:System.Text.Encoding"></see>.</returns>
        public override int WindowsCodePage
        {
            get
            {
                return _baseEncoding.WindowsCodePage;
            }
        }
        /// <summary>
        /// Determines whether the specified <see cref="T:System.Object"></see> is equal to the current instance.
        /// </summary>
        /// <param name="value">The <see cref="T:System.Object"></see> to compare with the current instance.</param>
        /// <returns>
        /// true if value is an instance of <see cref="T:System.Text.Encoding"></see> and is equal to the current instance; otherwise, false.
        /// </returns>
        public override bool Equals(object value)
        {
            return value is NoPreambleEncoding && ((NoPreambleEncoding)value)._baseEncoding.Equals(_baseEncoding);
        }
        /// <summary>
        /// Returns the hash code for the current instance.
        /// </summary>
        /// <returns>The hash code for the current instance.</returns>
        public override int GetHashCode()
        {
            return _baseEncoding.GetHashCode();
        }
        /// <summary>
        /// Calculates the number of bytes produced by encoding a set of characters from the specified character array.
        /// </summary>
        /// <param name="chars">The character array containing the set of characters to encode.</param>
        /// <param name="index">The index of the first character to encode.</param>
        /// <param name="count">The number of characters to encode.</param>
        /// <returns>
        /// The number of bytes produced by encoding the specified characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">index or count is less than zero.-or- index and count do not denote a valid range in chars. </exception>
        /// <exception cref="T:System.ArgumentNullException">chars is null. </exception>
        public override int GetByteCount(char[] chars, int index, int count)
        {
            return _baseEncoding.GetByteCount(chars, index, count);
        }

        /// <summary>
        /// Encodes a set of characters from the specified character array into the specified byte array.
        /// </summary>
        /// <param name="chars">The character array containing the set of characters to encode.</param>
        /// <param name="charIndex">The index of the first character to encode.</param>
        /// <param name="charCount">The number of characters to encode.</param>
        /// <param name="bytes">The byte array to contain the resulting sequence of bytes.</param>
        /// <param name="byteIndex">The index at which to start writing the resulting sequence of bytes.</param>
        /// <returns>
        /// The actual number of bytes written into bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">chars is null.-or- bytes is null. </exception>
        /// <exception cref="T:System.ArgumentException">bytes does not have enough capacity from byteIndex to the end of the array to accommodate the resulting bytes. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">charIndex or charCount or byteIndex is less than zero.-or- charIndex and charCount do not denote a valid range in chars.-or- byteIndex is not a valid index in bytes. </exception>
        public override int GetBytes(char[] chars, int charIndex, int charCount, byte[] bytes, int byteIndex)
        {
            return _baseEncoding.GetBytes(chars, charIndex, charCount, bytes, byteIndex);
        }

        /// <summary>
        /// Calculates the number of characters produced by decoding a sequence of bytes from the specified byte array.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <param name="index">The index of the first byte to decode.</param>
        /// <param name="count">The number of bytes to decode.</param>
        /// <returns>
        /// The number of characters produced by decoding the specified sequence of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentNullException">bytes is null. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">index or count is less than zero.-or- index and count do not denote a valid range in bytes. </exception>
        public override int GetCharCount(byte[] bytes, int index, int count)
        {
            return _baseEncoding.GetCharCount(bytes, index, count);
        }

        /// <summary>
        /// Decodes a sequence of bytes from the specified byte array into the specified character array.
        /// </summary>
        /// <param name="bytes">The byte array containing the sequence of bytes to decode.</param>
        /// <param name="byteIndex">The index of the first byte to decode.</param>
        /// <param name="byteCount">The number of bytes to decode.</param>
        /// <param name="chars">The character array to contain the resulting set of characters.</param>
        /// <param name="charIndex">The index at which to start writing the resulting set of characters.</param>
        /// <returns>
        /// The actual number of characters written into chars.
        /// </returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">byteIndex or byteCount or charIndex is less than zero.-or- byteindex and byteCount do not denote a valid range in bytes.-or- charIndex is not a valid index in chars. </exception>
        /// <exception cref="T:System.ArgumentNullException">bytes is null.-or- chars is null. </exception>
        /// <exception cref="T:System.ArgumentException">chars does not have enough capacity from charIndex to the end of the array to accommodate the resulting characters. </exception>
        public override int GetChars(byte[] bytes, int byteIndex, int byteCount, char[] chars, int charIndex)
        {
            return _baseEncoding.GetChars(bytes, byteIndex, byteCount, chars, charIndex);
        }

        /// <summary>
        /// calculates the maximum number of bytes produced by encoding the specified number of characters.
        /// </summary>
        /// <param name="charCount">The number of characters to encode.</param>
        /// <returns>
        /// The maximum number of bytes produced by encoding the specified number of characters.
        /// </returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">charCount is less than zero. </exception>
        public override int GetMaxByteCount(int charCount)
        {
            return _baseEncoding.GetMaxByteCount(charCount);
        }

        /// <summary>
        /// calculates the maximum number of characters produced by decoding the specified number of bytes.
        /// </summary>
        /// <param name="byteCount">The number of bytes to decode.</param>
        /// <returns>
        /// The maximum number of characters produced by decoding the specified number of bytes.
        /// </returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">byteCount is less than zero. </exception>
        public override int GetMaxCharCount(int byteCount)
        {
            return _baseEncoding.GetMaxCharCount(byteCount);
        }
    }
    #endregion

    #region Parallel Enumeration
#if ENABLE_PARALLEL_ENUMERATION
    /// <summary>
    ///
    /// </summary>
    public class BackgroundEnumerable : IEnumerable
    {
        IEnumerable baseEnumerable;

        /// <summary>
        ///
        /// </summary>
        /// <param name="baseEnumerable"></param>
        public BackgroundEnumerable(IEnumerable baseEnumerable)
        {
            this.baseEnumerable = baseEnumerable;
        }

#if EXPLICIT
        public IEnumerator GetEnumerator()
        {
            return new BackgroundEnumerator(baseEnumerable.GetEnumerator());
        }

        private class BackgroundEnumerator : IEnumerator, IDisposable
        {
            IEnumerator baseEnumerator;
            Thread thread;
            AutoResetEvent fillMutex;
            AutoResetEvent drainMutex;
            object current;
            object next;

            public BackgroundEnumerator(IEnumerator baseEnumerator)
            {
                this.baseEnumerator = baseEnumerator;
                fillMutex = new AutoResetEvent(false);
                drainMutex = new AutoResetEvent(true);
                thread = new Thread(new ThreadStart(Enumerate));
                thread.Start();
            }

            private void Enumerate()
            {
            }

            public object Current
            {
                get
                {
                    return current;
                }
            }

            public bool MoveNext()
            {
                fillMutex.WaitOne();
                current = next;
                drainMutex.Set();
            }

            public void Reset()
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
                baseEnumerator.Reset();
                fillMutex = new AutoResetEvent(false);
                drainMutex = new AutoResetEvent(true);
                thread = new Thread(new ThreadStart(Enumerate));
                thread.Start();
            }

            void IDisposable.Dispose()
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
            }
        }
#else
        /// <summary>
        ///
        /// </summary>
        /// <returns></returns>
        public IEnumerator GetEnumerator()
        {
            try
            {
                fillMutex = new AutoResetEvent(false);
                drainMutex = new AutoResetEvent(true);
                thread = new Thread(new ThreadStart(Enumerate));
                thread.Start();

                while (true)
                {
                    fillMutex.WaitOne();
                    current = next;
                    if (current == null) yield break;
                    drainMutex.Set();
                    yield return current;
                }
            }
            finally
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
            }
        }

        Thread thread;
        AutoResetEvent fillMutex;
        AutoResetEvent drainMutex;
        object current;
        object next;

        private void Enumerate()
        {
            foreach (object obj in baseEnumerable)
            {
                drainMutex.WaitOne();
                next = obj;
                fillMutex.Set();
            }
            drainMutex.WaitOne();
            next = null;
            fillMutex.Set();
        }
#endif
    }

    /// <summary>
    ///
    /// </summary>
    public class BackgroundEnumerable<T> : IEnumerable<T>
    {
        IEnumerable<T> baseEnumerable;

        /// <summary>
        ///
        /// </summary>
        /// <param name="baseEnumerable"></param>
        public BackgroundEnumerable(IEnumerable<T> baseEnumerable)
        {
            this.baseEnumerable = baseEnumerable;
        }

#if EXPLICIT
        public IEnumerator GetEnumerator()
        {
            return new BackgroundEnumerator(baseEnumerable.GetEnumerator());
        }

        private class BackgroundEnumerator : IEnumerator, IDisposable
        {
            IEnumerator baseEnumerator;
            Thread thread;
            AutoResetEvent fillMutex;
            AutoResetEvent drainMutex;
            object current;
            object next;

            public BackgroundEnumerator(IEnumerator baseEnumerator)
            {
                this.baseEnumerator = baseEnumerator;
                fillMutex = new AutoResetEvent(false);
                drainMutex = new AutoResetEvent(true);
                thread = new Thread(new ThreadStart(Enumerate));
                thread.Start();
            }

            private void Enumerate()
            {
            }

            public object Current
            {
                get
                {
                    return current;
                }
            }

            public bool MoveNext()
            {
                fillMutex.WaitOne();
                current = next;
                drainMutex.Set();
            }

            public void Reset()
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
                baseEnumerator.Reset();
                fillMutex = new AutoResetEvent(false);
                drainMutex = new AutoResetEvent(true);
                thread = new Thread(new ThreadStart(Enumerate));
                thread.Start();
            }

            void IDisposable.Dispose()
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
            }
        }
#else
        /// <summary>
        ///
        /// </summary>
        /// <returns></returns>
        public IEnumerator<T> GetEnumerator()
        {
            try
            {
                next = new T[3][];
                for (int i = 0; i < next.Length; i++) next[i] = new T[4];
                T[] current = new T[next[0].Length];
                fillMutex = new AutoResetEvent[next.Length];
                drainMutex = new AutoResetEvent[next.Length];
                for (int i = 0; i < fillMutex.Length; i++)
                {
                    fillMutex[i] = new AutoResetEvent(false);
                    drainMutex[i] = new AutoResetEvent(true);
                }
                thread = new Thread(new ThreadStart(Enumerate));
                thread.IsBackground = true;
                thread.Start();

                int currentIndex = 0;
                while (true)
                {
                    fillMutex[currentIndex].WaitOne();
                    T[] tmp = current;
                    current = next[currentIndex];
                    next[currentIndex] = tmp;
                    drainMutex[currentIndex].Set();
                    if (current == null) yield break;
                    for (int i = 0; i < current.Length; i++)
                    {
                        yield return current[i];
                    }
                    currentIndex = (currentIndex + 1) % next.Length;
                }
            }
            finally
            {
                if (thread != null)
                {
                    try
                    {
                        thread.Abort();
                    }
                    catch
                    {
                    }
                    thread = null;
                }
            }
        }

        Thread thread;
        AutoResetEvent[] fillMutex;
        AutoResetEvent[] drainMutex;
        T[][] next;

        private void Enumerate()
        {
            int nextIndex = 0;
            int index = 0;
            foreach (T obj in baseEnumerable)
            {
                next[nextIndex][index] = obj;
                index++;
                if (index == next[nextIndex].Length)
                {
                    fillMutex[nextIndex].Set();
                    nextIndex = (nextIndex + 1) % next.Length;
                    drainMutex[nextIndex].WaitOne();
                    index = 0;
                }
            }
            if (index != 0)
            {
                T[] old = next[nextIndex];
                next[nextIndex] = new T[index];
                Array.Copy(old, next[nextIndex], next[nextIndex].Length);
                fillMutex[nextIndex].Set();
                nextIndex = (nextIndex + 1) % next.Length;
                drainMutex[nextIndex].WaitOne();
                index = 0;
            }
            next[nextIndex] = null;
            fillMutex[nextIndex].Set();
            thread = null;
        }
#endif

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
#endif
    #endregion

#if NDOC
    /// <summary>
    /// <para>
    /// The Microsoft.ML.Runtime.Internal namespace includes classes and functionality developed
    /// by the Text Mining, Search, and Navigation group in Microsoft Research.
    /// All functionality is for internal use only and is copyright
    /// 2007 Microsoft Corporation.
    /// </para>
    /// <para>
    /// Microsoft.ML.Runtime.Internal.IO contains functionality to help with input and output operations.
    /// Using the ZStream classes to open files, support is automatically provided for
    /// compression and decompression, HTTP downloading, Cosmos stream fetching, console
    /// interaction, named streams, SQL table reading, lists of files, and so on.
    /// The concept of filename is extended to allow this to
    /// be simple and automatic. Other classes provide memory-mapped files, unbuffered
    /// I/O, table reading, and other useful functionality.
    /// </para>
    /// <para>
    /// The most essential feature is the ability to open the expanded class of stream names,
    /// easily and efficiently:
    /// </para>
    /// <code>
    /// Stream streamIn  = ZStreamIn.Open("filename");
    /// Stream streamOut = ZStreamOut.Open("filename");
    /// StreamReader reader = ZStreamReader.Open("filename");
    /// StreamWriter writer = ZStreamWriter.Open("filename");
    /// </code>
    /// <para>
    /// See the <c>Open()</c> documentation to see the allowed names for file, console,
    /// null, URL, Cosmos, and compressed input and output streams.
    /// </para>
    /// <para>
    /// Compression support relies on executable utilities to be in the path.
    /// See <see href="http://7-zip.org"/> for 7z.exe and 7za.exe (for many formats -
    /// .7z, .gz, .zip, .rar, .bz2, .cab, .arj), <see href="http://gnuwin32.sourceforge.net/packages/gzip.htm"/> for gzip.exe
    /// (for .gz), or <see href="http://rarsoft.com"/> for unrar.exe (for .rar). Gzip support built-in to .NET
    /// 2.0 can be used, but it has extreme deficiencies in terms of speed, size, and flexibility.
    /// </para>
    /// <para>
    /// The IOUtil static methods are useful functions for file and stream operations. Many duplicate
    /// existing Framework methods, but they are extended to handle all streamnames.
    /// </para>
    /// <para>
    /// BinaryReaderEx and BinaryWriterEx allow for easy and efficient serialization of many data types.
    /// They also make using .NET binary serialization as simple as using Write and Read on serializable objects.
    /// </para>
    /// </summary>
    public class NamespaceDoc
    {
        // this only exists for NDoc
    }
#endif
}
