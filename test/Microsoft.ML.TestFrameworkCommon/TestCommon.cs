// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class TestCommon
    {
        public static string GetOutputPath(string outDir, string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(outDir, name);
        }
        public static string GetOutputPath(string outDir, string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(subDir))
                return GetOutputPath(outDir, name);
            EnsureOutputDir(subDir, outDir);
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(outDir, subDir, name); // REVIEW: put the path in in braces in case the path has spaces
        }

        public static string GetDataPath(string dataDir, string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(dataDir, name));
        }
        public static string GetDataPath(string dataDir, string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(dataDir, subDir, name));
        }

        public static string DeleteOutputPath(string outDir, string subDir, string name)
        {
            string path = GetOutputPath(outDir, subDir, name);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }
        public static string DeleteOutputPath(string outDir, string name)
        {
            string path = GetOutputPath(outDir, name);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }

        /// <summary>
        /// Environment variable containing path to the test data and BaseLineOutput folders.
        /// </summary>
        public const string TestDataDirEnvVariable = "ML_TEST_DATADIR";

        public static string GetRepoRoot()
        {
            string directory = Environment.GetEnvironmentVariable(TestDataDirEnvVariable);
            if (directory != null)
            {
                return directory;
            }
#if NETFRAMEWORK
            directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            directory = AppContext.BaseDirectory;
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
            Assert.True(sch1.Count == sch2.Count, $"column count mismatch: {sch1.Count} vs {sch2.Count}");

            for (int col = 0; col < sch1.Count; col++)
            {
                string name1 = sch1[col].Name;
                string name2 = sch2[col].Name;
                Assert.True(name1 == name2, $"column name mismatch at index {col}: {name1} vs {name2}");

                var type1 = sch1[col].Type;
                var type2 = sch2[col].Type;
                Assert.True(EqualTypes(type1, type2, exactTypes), $"column type mismatch at index {col}");

                // This ensures that the two schemas map names to the same column indices.
                int col1;
                int col2;
                bool f1 = sch1.TryGetColumnIndex(name1, out col1);
                bool f2 = sch2.TryGetColumnIndex(name2, out col2);

                Assert.True(f1, "TryGetColumnIndex unexpectedly failed");
                Assert.True(f2, "TryGetColumnIndex unexpectedly failed");
                Assert.True(col1 == col2, $"TryGetColumnIndex on '{name1}' produced different results: '{col1}' vs '{col2}'");

                // This checks that an unknown metadata kind does the right thing.
                if (!CheckMetadataNames("PurpleDragonScales", 0, sch1, sch2, col, exactTypes, true))
                    return false;

                ulong vsize = type1 is VectorDataViewType vectorType ? (ulong)vectorType.Size : 0;
                if (!CheckMetadataNames("SlotNames", vsize, sch1, sch2, col, exactTypes, true))
                    return false;

                if (!keyNames)
                    continue;

                ulong ksize = type1.GetItemType() is KeyDataViewType keyType ? keyType.Count : 0;
                if (!CheckMetadataNames("KeyValues", ksize, sch1, sch2, col, exactTypes, false))
                    return false;
            }

            return true;
        }

        public static bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(in v1, in v2, size, (i, x, y) => fn(x, y));
        }

        public static bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Assert.True(size == 0 || v1.Length == size);
            Assert.True(size == 0 || v2.Length == size);
            Assert.True(v1.Length == v2.Length);

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

            Assert.True(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            var v1Indices = v1.GetIndices();
            var v2Indices = v2.GetIndices();
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1Indices.Length ? v1Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2Indices.Length ? v2Indices[iiv2] : v2.Length;
                T x1;
                T x2;
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
            Assert.NotNull(type1);
            Assert.NotNull(type2);

            return exactTypes ? type1.Equals(type2) : type1.SameSizeAndItemType(type2);
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
            Assert.False((t1 == null) != (t2 == null), $"Different null-ness of {kind} metadata types");

            if (t1 == null)
            {
                Assert.True(CheckMetadataCallFailure(kind, sch1, col, ref names1));
                Assert.True(CheckMetadataCallFailure(kind, sch2, col, ref names2));

                return true;
            }

            Assert.False(size > int.MaxValue, $"{nameof(KeyDataViewType)}.{nameof(KeyDataViewType.Count)} is larger than int.MaxValue");
            Assert.True(EqualTypes(t1, t2, exactTypes), $"Different {kind} metadata types: {t1} vs {t2}");

            if (!(t1.GetItemType() is TextDataViewType))
            {
                if (!mustBeText)
                    return true;

                Assert.False(mustBeText, $"Unexpected {kind} metadata type");
            }

            Assert.True((int)size == t1.GetVectorSize(), $"{kind} metadata type wrong size: {t1.GetVectorSize()} vs {size}");

            sch1[col].Annotations.GetValue(kind, ref names1);
            sch2[col].Annotations.GetValue(kind, ref names2);
            Assert.True(CompareVec(in names1, in names2, (int)size, (a, b) => a.Span.SequenceEqual(b.Span)), $"Different {kind} metadata values");

            return true;
        }

        private static bool CheckMetadataCallFailure(string kind, DataViewSchema sch, int col, ref VBuffer<ReadOnlyMemory<char>> names)
        {
            try
            {
                sch[col].Annotations.GetValue(kind, ref names);

                return false;
            }
            catch (InvalidOperationException ex)
            {
                if (ex.Message != "Invalid call to 'GetValue'")
                {
                    return false;
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
