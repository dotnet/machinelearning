namespace Microsoft.ML.Auto
{
    internal interface IDebugLogger
    {
        void Log(DebugStream stream, string message);
    }

    public enum DebugStream
    {
        Exception,
        RunResult
    }
}
