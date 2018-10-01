using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;

namespace Bubba
{
    class Foo
    {
        public static void Bar()
        {
            IHostEnvironment env = null;
            var text = TextLoader.CreateReader(env, ctx => new
            {
                Label = ctx.LoadBool(0),
                Text = ctx.LoadText(1),
                NumericFeatures = ctx.LoadFloat(2, 5)
            });

            var est = text.MakeNewEstimator();
            // This should work.
            est.Append(r => new { r.Text });

            IDataView view = null;
            view.AssertStatic(env, c => new Class1(c.I4.Scalar, c.Bool.Vector));
            view.AssertStatic(env, c => new Class2 { F1 = c.I4.Scalar, F2 = c.Bool.Vector });
            view.AssertStatic(env, c => new Class3<Class2>
            {
                F1 = new Class1(c.I4.Scalar, c.Bool.Vector),
                F2 = new Class2 { F1 = c.I4.Scalar, F2 = c.Bool.Vector }
            });
            view.AssertStatic(env, c => new Class4 { F1 = c.I4.Scalar });
            view.AssertStatic<Class5>(env, c => null);
            view.AssertStatic(env, c => new Class6(c.I4.Scalar, c.Bool.Vector));
            view.AssertStatic(env, c => new Class7 { F2 = c.Bool.Vector });
            view.AssertStatic(env, c => new Class8(c.I4.Scalar, c.Bool.Vector));
            view.AssertStatic(env, c => new Class9 { F1 = c.I4.Scalar, F2 = c.Bool.Vector });
            view.AssertStatic(env, c => new Class10(c.I4.Scalar, c.Bool.Vector));
            view.AssertStatic(env, c => new Class11(c.I4.Scalar, c.Bool.Vector, c.Bool.Vector));

            // This is wrong but should not fail with our diagnostic since there is a deeper problem that the class
            // simply is not there.
            var text2 = TextLoader.CreateReader(env, ctx => new MissingClass(ctx.LoadText(0)));
        }
    }

    class Class1 // This is good.
    {
        public Class1(Scalar<int> f1, Vector<bool> f2)
        {
            F1 = f1;
            F2 = f2;
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; }
    }

    class Class2 // This is good.
    {
        public Scalar<int> F1 { get; set; }
        public Vector<bool> F2 { get; set; }
    }

    class Class3<[IsShape] T> // This is good.
    {
        public Class1 F1 { get; set; }
        public T F2 { get; set; }
    }

    class Class4 // This is bad, since it has fields, not properties.
    {
        public Scalar<int> F1;
    }

    class Class5 // This is bad since its single constructor is not accessible.
    {
        protected Class5(Scalar<int> f1, Vector<bool> f2)
        {
            F1 = f1;
            F2 = f2;
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; }
    }

    class Class6 // This is bad since it has two public constructors.
    {
        public Class6(Scalar<int> f1, Vector<bool> f2)
        {
            F1 = f1;
            F2 = f2;
        }

        public Class6(Vector<bool> f2, Scalar<int> f1)
            : this(f1, f2)
        {
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; }
    }

    class Class7 // This is bad since it has only an implicit constructor, but only F2 has a set accessor.
    {
        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; set; }
    }

    class Class8 // This is bad since it has a constructor with explicit parameters, but also a set accessor on F2.
    {
        public Class8(Scalar<int> f1, Vector<bool> f2)
        {
            F1 = f1;
            F2 = f2;
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; set; }
    }

    class Class9 // This is bad since F2's get accessor is not publicly accessible.
    {
        public Scalar<int> F1 { get; set; }
        public Vector<bool> F2 { private get; set; }
    }

    class Class10 // This is bad since there are two Vector<bool> columns but only one in the constructor.
    {
        public Class10(Scalar<int> f1, Vector<bool> f2)
        {
            F1 = f1;
            F3 = F2 = f2;
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; }
        public Vector<bool> F3 { get; }
    }

    class Class11 // This is bad since there is one Vector<bool> columns but only one in the constructor.
    {
        public Class11(Scalar<int> f1, Vector<bool> f2, Vector<bool> f3)
        {
            F1 = f1;
            F2 = f2;
        }

        public Scalar<int> F1 { get; }
        public Vector<bool> F2 { get; }
    }
}