using System;

namespace TestNamespace
{
    class TypeName : IEquatable<TypeName>
    {
        // Let's talk about ice-cream.
        int Foo;
        private int _bar;


        static public int Hi;

        static TypeName()
        {
            Hi = 2;
        }

        TypeName() : this(5)
        {
        }

        public TypeName(int value)
        {
            _bar = Foo = value;
        }

        public struct Bubba
        {
            readonly int A;
            public int B { get; }
            int C { get; set; }
            readonly private int _d;
        }

        protected static internal void Yo() { }
        // Hello
        static /* So then I says to Mabel, I says... */ protected internal int Dog() { return 1; }

        enum Enum1 { A, B, C }
        public enum Enum2 { A, B, C }

        bool IEquatable<TypeName>.Equals(TypeName other)
            => other != null && other.Foo == Foo && other._bar == _bar;

        bool Silly() => false;

        public bool Billy() => true;

        public static TypeName operator +(TypeName a, TypeName b) => new TypeName(1);
        static TypeName operator -(TypeName a, TypeName b) => new TypeName(2);
    }

    public interface ITest1
    {
        int GetStuff();
    }

    interface ITest2
    {
        int GetStuffPlus1(ITest1 other);
    }
}