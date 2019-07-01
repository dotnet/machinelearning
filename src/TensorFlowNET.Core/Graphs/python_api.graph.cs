using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Graphs
{
    /// <summary>
    /// Lots of other functions required for Operation control flow like AddControlInput, UpdateEdge, RemoveAllControlInputs etc are not exposed via C_API and there is a C implementation of it.
    /// https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/c/python_api.h
    /// https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/c/python_api.cc
    /// 
    /// </summary>
    public class python_api
    {
        public static void UpdateEdge(Graph graph, TF_Output new_src, TF_Input dst, Status status)
        {
            
        }
    }
}
