using System;
using Bubba;

namespace McGee
{
    class YoureMyBestFriend
    {
        public void Foo()
        {
            Console.WriteLine(A.Hello);
            Console.WriteLine(A.My);
            Console.WriteLine(B.Friend);
            Console.WriteLine(B.Stay);
            Console.WriteLine(B.Awhile);
            Console.WriteLine(C.And);
            Console.WriteLine(C.Listen);

            var a = new A();
            var b = new B();
            var c = new C(2);
            c = new C(2.0f);
            var d = new D(2);
            d = new D(2.0f);

            var da = (IA)c;
            var db = (IB)d;
        }

        public class CDescend : C
        {
            public CDescend(int a) : base(a) { }
            public CDescend(float a) : base(a) { }
        }

        public class DDescend : D
        {
            public DDescend(int a) : base(a) { }
            public DDescend(float a) : base(a) { }
        }
    }
}
