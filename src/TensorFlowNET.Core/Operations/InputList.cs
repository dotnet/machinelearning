using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class InputList : IEnumerable
    {
        public Tensor[] _inputs;
        public int Length => _inputs.Length;
        public Tensor this[int index]
        {
            get
            {
                if (index == -1)
                    index = _inputs.Length - 1;
                return _inputs[index];
            }
        }

        public InputList(Tensor[] inputs)
        {
            _inputs = inputs;
        }

        public IEnumerator GetEnumerator()
        {
            return _inputs.GetEnumerator();
        }

        public static implicit operator List<Tensor>(InputList input)
        {
            return input._inputs.ToList();
        }

        public static implicit operator Tensor[](InputList input)
        {
            return input._inputs;
        }
    }
}
