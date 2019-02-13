using System;
using Microsoft.ML;

namespace TestNamespace
{
    // all of the best friend declaration should fail the diagnostic

    [BestFriend]
    public class PublicClass
    {
        [BestFriend]
        public int PublicField;

        [BestFriend]
        public string PublicProperty
        {
            get { return string.Empty; }
        }

        [BestFriend]
        public bool PublicMethod()
        {
            return true;
        }

        [BestFriend]
        public delegate string PublicDelegate();

        [BestFriend]
        public PublicClass()
        {
        }
    }

    [BestFriend]
    public struct PublicStruct
    {
    }

    [BestFriend]
    public enum PublicEnum
    {
        EnumValue1,
        EnumValue2
    }

    [BestFriend]
    public interface PublicInterface
    {
    }

    // these should work

    [BestFriend]
    internal class InternalClass
    {
        [BestFriend]
        internal int InternalField;

        [BestFriend]
        internal string InternalProperty
        {
            get { return string.Empty; }
        }

        [BestFriend]
        internal bool InternalMethod()
        {
            return true;
        }

        [BestFriend]
        internal delegate string InternalDelegate();

        [BestFriend]
        internal InternalClass()
        {
        }
    }

    [BestFriend]
    internal struct InternalStruct
    {
    }

    [BestFriend]
    internal enum InternalEnum
    {
        EnumValue1,
        EnumValue2
    }

    [BestFriend]
    internal interface InternalInterface
    {
    }

    // this should fail the diagnostic
    // a repro for https://github.com/dotnet/machinelearning/pull/2434#discussion_r254770946
    internal class InternalClassWithPublicMember
    {
        [BestFriend]
        public void PublicMethod()
        {
        }
    }
}
