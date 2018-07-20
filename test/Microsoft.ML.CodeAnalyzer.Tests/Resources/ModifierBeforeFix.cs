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

        int this[int i] => i + 2;
    }

    public interface ITest1
    {
        int GetStuff();
    }

    interface ITest2
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
        new int Method0() => 10;
        public new int Method1() => 11;
        new public int Method2() => 12;
        protected new internal int Method3() => 13;
    }

    public class EventBoy
    {
        delegate void AwesomeDelegate();
        event AwesomeDelegate Ev;
        event AwesomeDelegate EvProp { get; }
    }
}