// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class CommonUtilities
    {
        public static string GetOutputPath(string name, string outDir)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(outDir, name);
        }
        public static string GetOutputPath(string subDir, string name, string outDir)
        {
            if (string.IsNullOrWhiteSpace(subDir))
                return GetOutputPath(name, outDir);
            EnsureOutputDir(subDir, outDir);
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(outDir, subDir, name); // REVIEW: put the path in in braces in case the path has spaces
        }

        public static string GetDataPath(string name, string dataDir)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(dataDir, name));
        }
        public static string GetDataPath(string subDir, string name, string dataDir)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(dataDir, subDir, name));
        }

        public static string DeleteOutputPath(string subDir, string name, string outDir)
        {
            string path = GetOutputPath(subDir, name, outDir);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }
        public static string DeleteOutputPath(string name, string outDir)
        {
            string path = GetOutputPath(name, outDir);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }

        public static string GetRepoRoot()
        {
#if NETFRAMEWORK
            string directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            string directory = AppContext.BaseDirectory;
#endif

            while (!Directory.Exists(Path.Combine(directory, ".git")) && directory != null)
            {
                directory = Directory.GetParent(directory).FullName;
            }

            if (directory == null)
            {
                return null;
            }
            return directory;
        }

        public static bool CheckSameSchemas(DataViewSchema sch1, DataViewSchema sch2, bool exactTypes = true, bool keyNames = true)
        {
            if (sch1.Count != sch2.Count)
            {
                Fail("column count mismatch: {0} vs {1}", sch1.Count, sch2.Count);
                return Failed();
            }

            for (int col = 0; col < sch1.Count; col++)
            {
                string name1 = sch1[col].Name;
                string name2 = sch2[col].Name;
                if (name1 != name2)
                {
                    Fail("column name mismatch at index {0}: {1} vs {2}", col, name1, name2);
                    return Failed();
                }
                var type1 = sch1[col].Type;
                var type2 = sch2[col].Type;
                if (!EqualTypes(type1, type2, exactTypes))
                {
                    Fail("column type mismatch at index {0}", col);
                    return Failed();
                }

                // This ensures that the two schemas map names to the same column indices.
                int col1, col2;
                bool f1 = sch1.TryGetColumnIndex(name1, out col1);
                bool f2 = sch2.TryGetColumnIndex(name2, out col2);
                if (!Check(f1, "TryGetColumnIndex unexpectedly failed"))
                    return Failed();
                if (!Check(f2, "TryGetColumnIndex unexpectedly failed"))
                    return Failed();
                if (col1 != col2)
                {
                    Fail("TryGetColumnIndex on '{0}' produced different results: '{1}' vs '{2}'", name1, col1, col2);
                    return Failed();
                }

                // This checks that an unknown metadata kind does the right thing.
                if (!CheckMetadataNames("PurpleDragonScales", 0, sch1, sch2, col, exactTypes, true))
                    return Failed();

                ulong vsize = type1 is VectorDataViewType vectorType ? (ulong)vectorType.Size : 0;
                if (!CheckMetadataNames("SlotNames", vsize, sch1, sch2, col, exactTypes, true))
                    return Failed();

                if (!keyNames)
                    continue;

                ulong ksize = type1.GetItemType() is KeyDataViewType keyType ? keyType.Count : 0;
                if (!CheckMetadataNames("KeyValues", ksize, sch1, sch2, col, exactTypes, false))
                    return Failed();
            }

            return true;
        }

        public static bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(in v1, in v2, size, (i, x, y) => fn(x, y));
        }

        public static bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Assert(size == 0 || v1.Length == size);
            Assert(size == 0 || v2.Length == size);
            Assert(v1.Length == v2.Length);

            var v1Values = v1.GetValues();
            var v2Values = v2.GetValues();

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1Values[i];
                    var x2 = v2Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            var v1Indices = v1.GetIndices();
            var v2Indices = v2.GetIndices();
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1Indices.Length ? v1Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2Indices.Length ? v2Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1Values[iiv1];
                    x2 = v2Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2Values[iiv2];
                    iv = iv2;
                    iiv2++;
                }
                if (!fn(iv, x1, x2))
                    return false;
            }
        }

        public static bool EqualTypes(DataViewType type1, DataViewType type2, bool exactTypes)
        {
            AssertValue(type1);
            AssertValue(type2);

            return exactTypes ? type1.Equals(type2) : type1.SameSizeAndItemType(type2);
        }

        private static void Fail(string fmt, params object[] args)
        {
            Fail(false, fmt, args);
        }

        private static void Fail(bool relax, string fmt, params object[] args)
        {
            Log("*** Failure: " + fmt, args);
        }

        private static bool Failed()
        {
            return false;
        }

        private static void Log(string message, params object[] args)
        {
            Console.WriteLine(message, args);
        }

        private static bool Check(bool f, string msg)
        {
            if (!f)
                Fail(msg);
            return f;
        }

        [Conditional("DEBUG")]
        private static void AssertValue<T>(T val) where T : class
        {
            if (ReferenceEquals(val, null))
                Debug.Fail("Non - null assertion failure");
        }

        [Conditional("DEBUG")]
        private static void Assert(bool f)
        {
            if (!f)
                Debug.Fail("Assertion Failed");
        }

        [Conditional("DEBUG")]
        public static void Assert(bool f, string message)
        {
            if (!f)
                Debug.Fail(message);
        }

        /// <summary>
        /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
        /// returns true if current and other vector types have the same size and item type.
        /// </summary>
        private static bool SameSizeAndItemType(this DataViewType columnType, DataViewType other)
        {
            if (other == null)
                return false;

            if (columnType.Equals(other))
                return true;

            // For vector types, we don't care about the factoring of the dimensions.
            if (!(columnType is VectorDataViewType vectorType) || !(other is VectorDataViewType otherVectorType))
                return false;
            if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
                return false;
            return vectorType.Size == otherVectorType.Size;
        }

        private static bool TryGetColumnIndex(this DataViewSchema schema, string name, out int col)
        {
            col = schema.GetColumnOrNull(name)?.Index ?? -1;
            return col >= 0;
        }

        private static bool CheckMetadataNames(string kind, ulong size, DataViewSchema sch1, DataViewSchema sch2, int col, bool exactTypes, bool mustBeText)
        {
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var names2 = default(VBuffer<ReadOnlyMemory<char>>);

            var t1 = sch1[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
            var t2 = sch2[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
            if ((t1 == null) != (t2 == null))
            {
                Fail("Different null-ness of {0} metadata types", kind);
                return Failed();
            }
            if (t1 == null)
            {
                if (!CheckMetadataCallFailure(kind, sch1, col, ref names1))
                    return Failed();
                if (!CheckMetadataCallFailure(kind, sch2, col, ref names2))
                    return Failed();
                return true;
            }
            if (size > int.MaxValue)
                Fail(nameof(KeyDataViewType) + "." + nameof(KeyDataViewType.Count) + "is larger than int.MaxValue");
            if (!EqualTypes(t1, t2, exactTypes))
            {
                Fail("Different {0} metadata types: {0} vs {1}", kind, t1, t2);
                return Failed();
            }
            if (!(t1.GetItemType() is TextDataViewType))
            {
                if (!mustBeText)
                {
                    Log("Metadata '{0}' was not text so skipping comparison", kind);
                    return true; // REVIEW: Do something a bit more clever here.
                }
                Fail("Unexpected {0} metadata type", kind);
                return Failed();
            }

            if ((int)size != t1.GetVectorSize())
            {
                Fail("{0} metadata type wrong size: {1} vs {2}", kind, t1.GetVectorSize(), size);
                return Failed();
            }

            sch1[col].Annotations.GetValue(kind, ref names1);
            sch2[col].Annotations.GetValue(kind, ref names2);
            if (!CompareVec(in names1, in names2, (int)size, (a, b) => a.Span.SequenceEqual(b.Span)))
            {
                Fail("Different {0} metadata values", kind);
                return Failed();
            }
            return true;
        }

        private static bool CheckMetadataCallFailure(string kind, DataViewSchema sch, int col, ref VBuffer<ReadOnlyMemory<char>> names)
        {
            try
            {
                sch[col].Annotations.GetValue(kind, ref names);
                Fail("Getting {0} metadata unexpectedly succeeded", kind);
                return Failed();
            }
            catch (InvalidOperationException ex)
            {
                if (ex.Message != "Invalid call to 'GetValue'")
                {
                    Fail("Message from GetValue failed call doesn't match expected message: {0}", ex.Message);
                    return Failed();
                }
            }
            return true;
        }

        private static DataViewType GetItemType(this DataViewType columnType) => (columnType as VectorDataViewType)?.ItemType ?? columnType;

        private static int GetVectorSize(this DataViewType columnType) => (columnType as VectorDataViewType)?.Size ?? 0;

        private static void EnsureOutputDir(string subDir, string outDir)
        {
            Directory.CreateDirectory(Path.Combine(outDir, subDir));
        }
    }
}
