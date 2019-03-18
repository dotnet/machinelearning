// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Utilities to parse command-line representations of <see cref="IDataView"/> types.
    /// </summary>
    [BestFriend]
    internal static class TypeParsingUtils
    {
        /// <summary>
        /// Attempt to parse the string into a data kind and (optionally) a keyCount. This method does not check whether
        /// the returned <see cref="InternalDataKind"/> can really be made into a key with the specified <paramref name="keyCount"/>.
        /// </summary>
        /// <param name="str">The string to parse.</param>
        /// <param name="dataKind">The parsed data kind.</param>
        /// <param name="keyCount">The parsed key count, or null if there's no key specification.</param>
        /// <returns>Whether the parsing succeeded or not.</returns>
        public static bool TryParseDataKind(string str, out InternalDataKind dataKind, out KeyCount keyCount)
        {
            Contracts.CheckValue(str, nameof(str));
            keyCount = null;
            dataKind = default;

            int ich = str.IndexOf('[');
            if (0 <= ich)
            {
                if (str[str.Length - 1] != ']')
                    return false;
                keyCount = KeyCount.Parse(str.Substring(ich + 1, str.Length - ich - 2));
                if (keyCount == null)
                    return false;
                if (ich == 0)
                    return true;
                str = str.Substring(0, ich);
            }

            if (!Enum.TryParse(str, true, out dataKind))
                return false;

            return true;
        }

        /// <summary>
        /// Construct a <see cref="KeyType"/> out of the data kind and the keyCount.
        /// </summary>
        public static KeyType ConstructKeyType(InternalDataKind? type, KeyCount keyCount)
        {
            Contracts.CheckValue(keyCount, nameof(keyCount));

            KeyType keyType;
            Type rawType = type.HasValue ? type.Value.ToType() : InternalDataKind.U8.ToType();
            Contracts.CheckUserArg(KeyType.IsValidDataType(rawType), nameof(TextLoader.Column.Type), "Bad item type for Key");

            if (keyCount.Count == null)
                keyType = new KeyType(rawType, rawType.ToMaxInt());
            else
                keyType = new KeyType(rawType, keyCount.Count.GetValueOrDefault());

            return keyType;
        }
    }

    /// <summary>
    /// Defines the cardinality, or count, of valid values of a <see cref="KeyType"/> column. This needs to be strictly positive.
    /// It is used by <see cref="TextLoader"/> and <see cref="TypeConvertingEstimator"/>.
    /// </summary>
    public sealed class KeyCount
    {
        /// <summary>
        /// Initializes the cardinality, or count, of valid values of a <see cref="KeyType"/> column to the
        /// largest integer that can be expresed by the underlying datatype of the <see cref="KeyType"/>.
        /// </summary>
        public KeyCount() { }

        /// <summary>
        /// Initializes the cardinality, or count, of valid values of a <see cref="KeyType"/> column to <paramref name="count"/>
        /// </summary>
        public KeyCount(ulong count)
        {
            if (count == 0)
                throw Contracts.ExceptParam(nameof(count), "The cardinality of valid values of a "
                    + nameof(KeyType) + " column has to be strictly positive.");
            Count = count;
        }

        [Argument(ArgumentType.AtMostOnce, HelpText = "Count of valid key values")]
        public ulong? Count;

        /// <summary>
        /// Parses the string format for a KeyCount, also supports the old KeyRange format for backwards compatibility.
        /// </summary>
        internal static KeyCount Parse(string str)
        {
            Contracts.AssertValue(str);

            var res = new KeyCount();
            if (res.TryParse(str))
                return res;
            return null;
        }

        private bool TryParse(string str)
        {
            Contracts.AssertValue(str);

            // This corresponds to the new format `[]`, with no specified Max.
            if (str.Length == 0)
                return true;

            // For backward compatibility we check for the old format that included a Min and looked like: `[Min-Max]`.
            int ich = str.IndexOf('-');
            if (0 <= ich)
            {
                ulong min;
                // Parse Min and the dash, throw if Min is not zero.
                if (!ulong.TryParse(str.Substring(0, ich), out min))
                    return false;
                if (min != 0)
                    throw Contracts.ExceptDecode("The minimum logical value of a " + nameof(KeyType) + " is required to be zero.");

                // The Max could be non defined or it could be an `*`.
                str = str.Substring(ich + 1);
                if (string.IsNullOrEmpty(str) || str == "*")
                    return true;
            }

            // This is the new format `[Max]`.
            ulong tmp;
            if (!ulong.TryParse(str, out tmp))
                return false;

            // The new string format for a key reflects KeyType.Count and expresses the cardinality/count of valid values.
            // The old format was a range with the max of the range equal to keyCount - 1.
            Count = ich == -1 ? tmp : tmp + 1;

            Contracts.CheckDecode(Count == null || Count > 0);
            return true;
        }

        internal bool TryUnparse(StringBuilder sb)
        {
            Contracts.AssertValue(sb);
            Contracts.Assert(Count == null || Count > 0);

            if (Count != null)
                sb.Append(Count);
            return true;
        }
    }
}
