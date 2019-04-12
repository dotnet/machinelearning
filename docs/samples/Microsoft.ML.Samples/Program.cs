using System.Reflection;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll();

        internal static void RunAll()
        {
            foreach (var type in Assembly.GetExecutingAssembly().GetTypes())
                type.GetMethod("Example", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy)?.Invoke(null, null);
        }
    }
}
