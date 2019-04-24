using System;
using System.Reflection;
using Samples.Dynamic;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll();

        internal static void RunAll()
        {
            DataViewEnumerable.Example();
            FilterRowsByColumn.Example();
            ShuffleRows.Example();
            SkipRows.Example();
            TakeRows.Example();
        }
    }
}
