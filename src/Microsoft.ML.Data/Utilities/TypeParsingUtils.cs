// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML.CommandLine;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Utilities to parse command-line representations of <see cref="IDataView"/> types.
    /// </summary>
    public static class TypeParsingUtils
    {
        /// <summary>
        /// Attempt to parse the string into a data kind and (optionally) a key range. This method does not check whether
        /// the returned <see cref="DataKind"/> can really be made into a key with the specified <paramref name="keyRange"/>.
        /// </summary>
        /// <param name="str">The string to parse.</param>
        /// <param name="dataKind">The parsed data kind.</param>
        /// <param name="keyRange">The parsed key range, or null if there's no key specification.</param>
        /// <returns>Whether the parsing succeeded or not.</returns>
        public static bool TryParseDataKind(string str, out DataKind dataKind, out KeyRange keyRange)
        {
            Contracts.CheckValue(str, nameof(str));
            keyRange = null;
            dataKind = default(DataKind);

            int ich = str.IndexOf('[');
            if (ich >= 0)
            {
                if (str[str.Length - 1] != ']')
                    return false;
                keyRange = KeyRange.Parse(str.Substring(ich + 1, str.Length - ich - 2));
                if (keyRange == null)
                    return false;
                if (ich == 0)
                    return true;
                str = str.Substring(0, ich);
            }

            DataKind kind;
            if (!Enum.TryParse<DataKind>(str, true, out kind))
                return false;
            dataKind = kind;

            return true;
        }

        /// <summary>
        /// Construct a <see cref="KeyType"/> out of the data kind and the key range.
        /// </summary>
        public static KeyType ConstructKeyType(DataKind? type, KeyRange range)
        {
            Contracts.CheckValue(range, nameof(range));

            DataKind kind;
            KeyType keyType;
            kind = type ?? DataKind.U8;
            Contracts.CheckUserArg(KeyType.IsValidDataKind(kind), nameof(TextLoader.Column.Type), "Bad item type for Key");

            if (range.Max == null)
                keyType = new KeyType(kind, 0);
            else
            {
                ulong max = range.Max.GetValueOrDefault();
                Contracts.CheckUserArg(max >= 0, nameof(range.Max), "max must be >= 0");
                Contracts.CheckUserArg(max < ulong.MaxValue, nameof(range.Max), "range is too large");
                ulong count = max;
                Contracts.Assert(count >= 1);
                if (count > kind.ToMaxInt())
                    throw Contracts.ExceptUserArg(nameof(range.Max), "range is too large for type {0}", kind);
                keyType = new KeyType(kind, count);
            }
            return keyType;
        }
    }

    /// <summary>
    /// The key range specification. It is used by <see cref="TextLoader"/> and C# transform.
    /// </summary>
    public sealed class KeyRange
    {
        public KeyRange() { }

        public KeyRange(ulong? max = null)
        {
            Max = max;
        }

        [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
        public ulong? Max;

        public static KeyRange Parse(string str)
        {
            Contracts.AssertValue(str);

            var res = new KeyRange();
            if (res.TryParse(str))
                return res;
            return null;
        }

        private bool TryParse(string str)
        {
            Contracts.AssertValue(str);

            ulong min;
            int ich = str.IndexOf('-');
            if (ich < 0)
            {
                if (!ulong.TryParse(str, out min))
                    return false;
                Contracts.Assert(min == 0);
                return true;
            }

            if (!ulong.TryParse(str.Substring(0, ich), out min))
                return false;
            Contracts.Assert(min == 0);

            string rest = str.Substring(ich + 1);
            if (string.IsNullOrEmpty(rest) || rest == "*")
                return true;

            ulong tmp;
            if (!ulong.TryParse(rest, out tmp))
                return false;
            Max = tmp;
            return true;
        }

        public bool TryUnparse(StringBuilder sb)
        {
            Contracts.AssertValue(sb);

            sb.Append((ulong)0);
            if (Max != null)
                sb.Append('-').Append(Max);
            else
                sb.Append("-*");
            return true;
        }
    }
}
