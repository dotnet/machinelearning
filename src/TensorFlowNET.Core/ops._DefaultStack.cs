using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class ops
    {
        _DefaultStack _default_session_stack = new _DefaultStack();

        public class _DefaultStack : IPython
        {
            Stack<object> stack;
            bool _enforce_nesting = true;

            public _DefaultStack()
            {
                stack = new Stack<object>();
            }

            public void __enter__()
            {
                throw new NotImplementedException();
            }

            public void __exit__()
            {
                throw new NotImplementedException();
            }

            public void Dispose()
            {
                throw new NotImplementedException();
            }
        }
    }
}
