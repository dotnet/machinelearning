// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace GenerateNonSpacingMarkTable
{
    /// <summary>
    /// The parts of the Unicode Character Database needed to decompose and compose text canonically,
    /// read from UnicodeData.txt and DerivedNormalizationProps.txt.
    /// </summary>
    internal sealed class UnicodeData
    {
        public const int MaxCodePoint = 0x10FFFF;

        private readonly bool[] _isNonSpacingMark = new bool[MaxCodePoint + 1];
        private readonly int[] _combiningClasses = new int[MaxCodePoint + 1];
        private readonly Dictionary<int, int[]> _decompositions = new();
        private readonly Dictionary<(int First, int Second), int> _compositions = new();

        public UnicodeData(string unicodeDataPath, string derivedNormalizationPropsPath)
        {
            Version = ReadVersion(derivedNormalizationPropsPath);
            ReadUnicodeData(unicodeDataPath);

            // The primary composites: the canonical decompositions into two code points that aren't excluded from composition.
            HashSet<int> exclusions = ReadFullCompositionExclusions(derivedNormalizationPropsPath);
            foreach (KeyValuePair<int, int[]> decomposition in _decompositions)
            {
                if (decomposition.Value.Length == 2 && !exclusions.Contains(decomposition.Key))
                {
                    _compositions[(decomposition.Value[0], decomposition.Value[1])] = decomposition.Key;
                }
            }
        }

        /// <summary>
        /// Gets the Unicode version, for example "18.0.0".
        /// </summary>
        public string Version { get; }

        public bool IsNonSpacingMark(int codePoint)
        {
            return _isNonSpacingMark[codePoint];
        }

        /// <summary>
        /// Returns the full canonical decomposition of a code point. Hangul syllables aren't decomposed: their algorithmic
        /// decomposition has no non-spacing marks.
        /// </summary>
        public List<int> Decompose(int codePoint)
        {
            if (!_decompositions.TryGetValue(codePoint, out int[]? decomposition))
            {
                return [codePoint];
            }

            var result = new List<int>();
            foreach (int c in decomposition)
            {
                result.AddRange(Decompose(c));
            }

            return result;
        }

        /// <summary>
        /// Puts code points in canonical order, sorting each run of non-starters by canonical combining class.
        /// </summary>
        public List<int> Reorder(IEnumerable<int> codePoints)
        {
            var result = new List<int>(codePoints);
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = 0; i < result.Count - 1; i++)
                {
                    int nextCombiningClass = _combiningClasses[result[i + 1]];
                    if (nextCombiningClass != 0 && _combiningClasses[result[i]] > nextCombiningClass)
                    {
                        (result[i], result[i + 1]) = (result[i + 1], result[i]);
                        changed = true;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Returns the canonical composition (NFC) of code points. Like <see cref="Decompose"/>, it leaves Hangul alone.
        /// </summary>
        public List<int> Compose(IEnumerable<int> codePoints)
        {
            List<int> decomposed = Reorder(codePoints.SelectMany(Decompose));
            var result = new List<int>();
            if (decomposed.Count == 0)
            {
                return result;
            }

            result.Add(decomposed[0]);

            // The index in result of the last starter, or -1 before the first one.
            int starter = _combiningClasses[decomposed[0]] == 0 ? 0 : -1;

            // The combining class of the last code point added after the starter, or 0 when there's none.
            int lastCombiningClass = starter == 0 ? 0 : 256;

            foreach (int c in decomposed.Skip(1))
            {
                int combiningClass = _combiningClasses[c];

                // c combines with the starter unless a code point between them blocks it.
                if (starter >= 0 &&
                    (lastCombiningClass < combiningClass || (lastCombiningClass == 0 && combiningClass == 0 && result.Count - 1 == starter)) &&
                    _compositions.TryGetValue((result[starter], c), out int composite))
                {
                    result[starter] = composite;
                    continue;
                }

                if (combiningClass == 0)
                {
                    starter = result.Count;
                    lastCombiningClass = 0;
                }
                else
                {
                    lastCombiningClass = combiningClass;
                }

                result.Add(c);
            }

            return result;
        }

        private static string ReadVersion(string derivedNormalizationPropsPath)
        {
            // The first line names the file with its version, for example "# DerivedNormalizationProps-18.0.0.txt".
            string firstLine = File.ReadLines(derivedNormalizationPropsPath).FirstOrDefault() ?? string.Empty;
            Match match = Regex.Match(firstLine, @"^# DerivedNormalizationProps-(\d+\.\d+\.\d+)\.txt$");
            if (!match.Success)
            {
                throw new InvalidDataException($"Can't find the Unicode version in the first line of '{derivedNormalizationPropsPath}': '{firstLine}'.");
            }

            return match.Groups[1].Value;
        }

        private void ReadUnicodeData(string unicodeDataPath)
        {
            int previousCodePoint = -1;
            foreach (string line in File.ReadLines(unicodeDataPath))
            {
                // The fields are the code point, name, general category, canonical combining class, bidi class, decomposition, ...
                string[] fields = line.Split(';');
                int codePoint = ParseCodePoint(fields[0]);
                if (fields[1].EndsWith(", Last>", StringComparison.Ordinal))
                {
                    // This line and the previous one ("<..., First>") delimit a range of code points with the same properties.
                    for (int c = previousCodePoint + 1; c <= codePoint; c++)
                    {
                        SetProperties(c, fields);
                    }

                    continue;
                }

                previousCodePoint = codePoint;
                SetProperties(codePoint, fields);

                // Compatibility decompositions start with a tag such as "<compat>".
                if (fields[5].Length > 0 && !fields[5].StartsWith('<'))
                {
                    _decompositions[codePoint] = fields[5].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(ParseCodePoint).ToArray();
                }
            }
        }

        private void SetProperties(int codePoint, string[] fields)
        {
            _isNonSpacingMark[codePoint] = fields[2] == "Mn";
            _combiningClasses[codePoint] = int.Parse(fields[3], CultureInfo.InvariantCulture);
        }

        private static HashSet<int> ReadFullCompositionExclusions(string derivedNormalizationPropsPath)
        {
            var exclusions = new HashSet<int>();
            foreach (string line in File.ReadLines(derivedNormalizationPropsPath))
            {
                // For example "0958..095F    ; Full_Composition_Exclusion # Lo   [8] DEVANAGARI LETTER QA..DEVANAGARI LETTER YYA".
                string data = line.Split('#')[0].Trim();
                if (data.Length == 0)
                {
                    continue;
                }

                string[] fields = data.Split(';', StringSplitOptions.TrimEntries);
                if (fields[1] != "Full_Composition_Exclusion")
                {
                    continue;
                }

                string[] range = fields[0].Split("..");
                for (int c = ParseCodePoint(range[0]); c <= ParseCodePoint(range[^1]); c++)
                {
                    exclusions.Add(c);
                }
            }

            return exclusions;
        }

        private static int ParseCodePoint(string hex)
        {
            return int.Parse(hex, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
        }
    }
}
