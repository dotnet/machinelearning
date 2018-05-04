// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    /// <summary>
    ///This is a test class for TestPredictorMainTest and is intended
    ///to contain all TestPredictorMainTest Unit Tests
    ///</summary>
    public class TestBaselines : BaseTestBaseline
    {
        private static bool[] _forbidNumAfter;

        static TestBaselines()
        {
            _forbidNumAfter = new bool[128];
            for (char ch = '0'; ch <= '9'; ch++)
                _forbidNumAfter[ch] = true;
            for (char ch = 'A'; ch <= 'Z'; ch++)
                _forbidNumAfter[ch] = true;
            for (char ch = 'a'; ch <= 'z'; ch++)
                _forbidNumAfter[ch] = true;
            _forbidNumAfter['+'] = true;
            _forbidNumAfter['-'] = true;
            _forbidNumAfter['.'] = true;
        }

        public TestBaselines(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("BaseLines")]
        public void AAACompareBaselines()
        {
            Compare(GetBaselinePath(@"..\SingleDebug"), GetBaselinePath(@"..\SingleRelease"), @"..\CompareSngDebRel.out");
            Done();
        }

        private void Compare(string root1, string root2, string logName)
        {
            Log("Comparing baselines in {0} with {1}", root1, root2);

            string pathLog = DeleteOutputPath(logName);
            using (var log = OpenWriter(pathLog))
            {
                log.WriteLine("Comparison of baselines {0} to {1}", Path.GetFileName(root1), Path.GetFileName(root2));
                CompareDirs(log, root1, root2, "");
            }
            CheckEquality("", logName);
        }

        private void CompareDirs(TextWriter log, string root1, string root2, string rel)
        {
            string dir1 = Path.Combine(root1, rel);
            string dir2 = Path.Combine(root2, rel);

            var names2 = Directory.EnumerateFiles(dir2, "*.txt")
                .ToDictionary(s => Path.GetFileName(s).ToLowerInvariant(), s => false);
            foreach (string path1 in Directory.EnumerateFiles(dir1, "*.txt"))
            {
                string name = Path.GetFileName(path1);
                string relCur = Path.Combine(rel, name);
                string nameLower = name.ToLowerInvariant();

                if (!names2.ContainsKey(nameLower))
                    log.WriteLine("*** Missing right file: '{0}'", relCur);
                else
                {
                    Contracts.Assert(!names2[nameLower]);
                    names2[nameLower] = true;
                    CompareFiles(log, root1, root2, relCur);
                }
            }
            foreach (var kvp in names2)
            {
                if (!kvp.Value)
                    log.WriteLine("*** Missing left file: '{0}'", Path.Combine(rel, kvp.Key));
            }

            names2 = Directory.EnumerateDirectories(dir2)
                .ToDictionary(s => Path.GetFileName(s).ToLowerInvariant(), s => false);
            foreach (string path1 in Directory.EnumerateDirectories(dir1))
            {
                string name = Path.GetFileName(path1);
                string relCur = Path.Combine(rel, name);
                string nameLower = name.ToLowerInvariant();

                if (!names2.ContainsKey(nameLower))
                    log.WriteLine("*** Missing right directory: '{0}'", relCur);
                else
                {
                    Contracts.Assert(!names2[nameLower]);
                    names2[nameLower] = true;
                    CompareDirs(log, root1, root2, relCur);
                }
            }
            foreach (var kvp in names2)
            {
                if (!kvp.Value)
                    log.WriteLine("*** Missing left directory: '{0}'", Path.Combine(rel, kvp.Key));
            }
        }

        private static readonly Regex _matchIter = new Regex(@"^Iter [0-9]+:", RegexOptions.Compiled);

        private struct Stats
        {
            public double DiffMax;
            public int LineMax;
            public int ColMax;

            public double DiffTot;
            public long DiffCount;
            public long InfCount;

            public void Aggregate(double diff, int line, int col)
            {
                if (diff == 0)
                    return;

                if (!FloatUtils.IsFinite(diff))
                {
                    InfCount++;
                    return;
                }

                if (DiffMax < diff)
                {
                    DiffMax = diff;
                    LineMax = line;
                    ColMax = col;
                }

                DiffTot += diff;
                DiffCount++;
            }

            public void Report(TextWriter log, string path)
            {
                if (InfCount > 0)
                    log.Write("*** Infinite diffs: {0}, ", InfCount);

                if (DiffCount == 0)
                    log.WriteLine("Diffs:     0,                                                                                 File: {0}", path);
                else
                {
                    log.WriteLine("Diffs:{0,6}, Ave: {1:F20}, Max: {2:F20} at line {3,5} col {4,3}, File: {5}",
                        DiffCount, DiffTot / DiffCount, DiffMax, LineMax, ColMax, path);
                }
            }
        }

        private void CompareFiles(TextWriter log, string root1, string root2, string rel)
        {
            using (var rdr1 = OpenReader(Path.Combine(root1, rel)))
            using (var rdr2 = OpenReader(Path.Combine(root2, rel)))
            {
                Stats stats = default(Stats);

                int pos = 0;
                for (; ; )
                {
                    string line1 = rdr1.ReadLine();
                    string line2 = rdr2.ReadLine();
                    pos++;

                LRestart:
                    if (line1 == null && line2 == null)
                        break;
                    if (line1 == null)
                    {
                        log.WriteLine("*** First file is shorter: {0}", rel);
                        break;
                    }
                    if (line2 == null)
                    {
                        log.WriteLine("*** First file is longer: {0}", rel);
                        break;
                    }

                    if (line1 == line2)
                        continue;

                    int ich1 = 0;
                    int ich2 = 0;

                    bool mismatch;
                    for (; ; )
                    {
                        EatSpace(line1, ref ich1);
                        EatSpace(line2, ref ich2);

                        if (ich1 >= line1.Length || ich2 >= line2.Length)
                        {
                            mismatch = ich1 < line1.Length || ich2 < line2.Length;
                            break;
                        }

                        int col = ich1 + 1;

                        double d1;
                        double d2;
                        bool f1 = TryParseNumber(line1, ref ich1, out d1);
                        bool f2 = TryParseNumber(line2, ref ich2, out d2);
                        if (f1 && f2)
                        {
                            stats.Aggregate(Diff(d1, d2), pos, col);
                            continue;
                        }

                        if (f1 || f2 || line1[ich1] != line2[ich2])
                        {
                            mismatch = true;
                            break;
                        }

                        ich1++;
                        ich2++;
                    }

                    if (mismatch)
                    {
                        // Hack alert: if one has extra iterations, skip them.
                        if (_matchIter.IsMatch(line1) != _matchIter.IsMatch(line2))
                        {
                            int skip1 = 0;
                            int skip2 = 0;
                            while (line1 != null && _matchIter.IsMatch(line1))
                            {
                                skip1++;
                                line1 = rdr1.ReadLine();
                            }
                            while (line2 != null && _matchIter.IsMatch(line2))
                            {
                                skip2++;
                                line2 = rdr2.ReadLine();
                            }

                            log.WriteLine("*** Skipped {0} extra iterations in {1}", skip1 + skip2, rel);
                            pos += skip1;
                            goto LRestart;
                        }

                        log.WriteLine("*** Mismatch at line {0} in {1}", pos, rel);
                        log.WriteLine("    col {0}: {1}", ich1 + 1, line1);
                        log.WriteLine("    col {0}: {1}", ich2 + 1, line2);
                    }
                }

                stats.Report(log, rel);
            }
        }

        // This is absolute error near zero and relative error away from zero.
        private static double Diff(double d1, double d2)
        {
            if (d1 == d2)
                return 0;

            if (FloatUtils.IsFinite(d1) && FloatUtils.IsFinite(d2))
                return Math.Abs(d1 - d2) / Math.Max(1, Math.Max(Math.Abs(d1), Math.Abs(d2)));

            if (double.IsNaN(d1) && double.IsNaN(d2))
                return 0;
            return double.PositiveInfinity;
        }

        private static void EatSpace(string line, ref int ich)
        {
            while (ich < line.Length && line[ich] == ' ')
                ich++;
        }

        private bool TryParseNumber(string line, ref int ich, out double d)
        {
            if (ich > 0 && line[ich - 1] < _forbidNumAfter.Length && _forbidNumAfter[line[ich - 1]])
            {
                d = 0;
                return false;
            }

            int ichLim = ich;
            EatSign(line, ref ichLim);

            bool digits = EatDigits(line, ref ichLim);
            if (ichLim < line.Length && line[ichLim] == '.')
            {
                ichLim++;
                digits |= EatDigits(line, ref ichLim);
            }

            if (!digits)
            {
                if (Eq(line, ichLim, "NaN"))
                {
                    d = double.NaN;
                    ich = ichLim + "NaN".Length;
                    return true;
                }
                if (Eq(line, ichLim, "Infinity"))
                {
                    d = line[ich] == '-' ? double.NegativeInfinity : double.PositiveInfinity;
                    ich = ichLim + "Infinity".Length;
                    return true;
                }

                d = 0;
                return false;
            }

            if (ichLim < line.Length && (line[ichLim] == 'e' || line[ichLim] == 'E'))
            {
                int ichLim2 = ichLim + 1;
                EatSign(line, ref ichLim2);
                if (EatDigits(line, ref ichLim2))
                    ichLim = ichLim2;
            }

            string str = line.Substring(ich, ichLim - ich);
            if (!double.TryParse(str, out d))
            {
                Fail("Parsing a double failed!");
                d = 0;
                return false;
            }
            ich = ichLim;
            return true;
        }

        private static bool EatDigits(string line, ref int ich)
        {
            if (ich >= line.Length || !IsDigit(line[ich]))
                return false;

            ich++;
            while (ich < line.Length && IsDigit(line[ich]))
                ich++;

            return true;
        }

        private static bool IsDigit(char ch)
        {
            return '0' <= ch && ch <= '9';
        }

        private static void EatSign(string line, ref int ich)
        {
            if (ich < line.Length && (line[ich] == '-' || line[ich] == '+'))
                ich++;
        }

        private static bool Eq(string line, int ich, string val)
        {
            if (ich > line.Length - val.Length || line[ich] != val[0])
                return false;
            for (int i = 1; i < val.Length; i++)
            {
                if (line[ich + i] != val[i])
                    return false;
            }
            return true;
        }
    }
}