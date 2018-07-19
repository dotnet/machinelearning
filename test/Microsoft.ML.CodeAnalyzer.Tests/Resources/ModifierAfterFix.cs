using System;

namespace TestNamespace
{
    internal class TypeName : IEquatable<TypeName>
    {
        // Let's talk about ice-cream.
        private int Foo;
        private int _bar;


        public static int Hi;

        static TypeName()
        {
            Hi = 2;
        }

        private TypeName() : this(5)
        {
        }

        public TypeName(int value)
        {
            _bar = Foo = value;
        }

        public struct Bubba
        {
            private readonly int A;
            public int B { get; }
            private int C { get; set; }
            private readonly int _d;
        }

        protected internal static void Yo() { }
        // Hello
        protected /* So then I says to Mabel, I says... */ internal static int Dog() { return 1; }

        private enum Enum1 { A, B, C }
        public enum Enum2 { A, B, C }

        bool IEquatable<TypeName>.Equals(TypeName other)
            => other != null && other.Foo == Foo && other._bar == _bar;

        private bool Silly() => false;

        public bool Billy() => true;

        public static TypeName operator +(TypeName a, TypeName b) => new TypeName(1);
        private static TypeName operator -(TypeName a, TypeName b) => new TypeName(2);
    }

    public interface ITest1
    {
        int GetStuff();
    }

    internal interface ITest2
    {
        int GetStuffPlus1(ITest1 other);
    }

    public class Mycroft
    {
        public int Method0() => 0;
        public int Method1() => 1;
        public int Method2() => 2;
        protected internal int Method3() => 3;
    }

    public class Sherlock : Mycroft
    {
        new private int Method0() => 10;
        new public int Method1() => 11;
        new public int Method2() => 12;
        new protected internal int Method3() => 13;
    }
}