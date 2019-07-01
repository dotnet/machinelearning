using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Mapping C# functions to Python
    /// </summary>
    public static class Python
    {
        public static void print(object obj)
        {
            Console.WriteLine(obj.ToString());
        }

        //protected int len<T>(IEnumerable<T> a)
        //    => a.Count();

        public static int len(object a)
        {
            switch (a)
            {
                case Array arr:
                    return arr.Length;
                case IList arr:
                    return arr.Count;
                case ICollection arr:
                    return arr.Count;
                case NDArray ndArray:
                    return ndArray.len;
                case IEnumerable enumerable:
                    return enumerable.OfType<object>().Count();
            }
            throw new NotImplementedException("len() not implemented for type: " + a.GetType());
        }

        public static IEnumerable<int> range(int end)
        {
            return Enumerable.Range(0, end);
        }

        public static IEnumerable<int> range(int start, int end)
        {
            return Enumerable.Range(start, end - start);
        }

        public static T New<T>(object args) where T : IPyClass
        {
            var instance = Activator.CreateInstance<T>();

            instance.__init__(instance, args);

            return instance;
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static void with(IPython py, Action<IPython> action)
        {
            try
            {
                py.__enter__();
                action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static void with<T>(T py, Action<T> action) where T : IPython
        {
            try
            {
                py.__enter__();
                action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static TOut with<TIn, TOut>(TIn py, Func<TIn, TOut> action) where TIn : IPython
        {
            try
            {
                py.__enter__();
                return action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
                return default(TOut);
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        public static float time()
        {
            return (float)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds;
        }

        public static IEnumerable<(T, T)> zip<T>(NDArray t1, NDArray t2)
        {
            for (int i = 0; i < t1.size; i++)
                yield return (t1.Data<T>(i), t2.Data<T>(i));
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IList<T1> t1, IList<T2> t2)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i]);
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(NDArray t1, NDArray t2)
        {
            for (int i = 0; i < t1.size; i++)
                yield return (t1.Data<T1>(i), t2.Data<T2>(i));
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IEnumerable<T1> e1, IEnumerable<T2> e2)
        {
            var iter2 = e2.GetEnumerator();
            foreach (var v1 in e1)
            {
                iter2.MoveNext();
                var v2 = iter2.Current;
                yield return (v1, v2);
            }
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(Dictionary<TKey, TValue> values)
        {
            foreach (var item in values)
                yield return (item.Key, item.Value);
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(KeyValuePair<TKey, TValue>[] values)
        {
            foreach (var item in values)
                yield return (item.Key, item.Value);
        }

        public static IEnumerable<(int, T)> enumerate<T>(IList<T> values)
        {
            for (int i = 0; i < values.Count; i++)
                yield return (i, values[i]);
        }

        public static Dictionary<string, object> ConvertToDict(object dyn)
        {
            var dictionary = new Dictionary<string, object>();
            foreach (PropertyDescriptor propertyDescriptor in TypeDescriptor.GetProperties(dyn))
            {
                object obj = propertyDescriptor.GetValue(dyn);
                string name = propertyDescriptor.Name;
                dictionary.Add(name, obj);
            }
            return dictionary;
        }


        public static bool all(IEnumerable enumerable)
        {
            foreach (var e1 in enumerable)
            {
                if (!Convert.ToBoolean(e1))
                    return false;
            }
            return true;
        }

        public static bool any(IEnumerable enumerable)
        {
            foreach (var e1 in enumerable)
            {
                if (Convert.ToBoolean(e1))
                    return true;
            }
            return false;
        }

        public static double sum(IEnumerable enumerable)
        {
            var typedef = new Type[] { typeof(double), typeof(int), typeof(float) };
            var sum = 0.0d;
            foreach (var e1 in enumerable)
            {
                if (!typedef.Contains(e1.GetType()))
                    throw new Exception("Numeric array expected");
                sum += (double)e1;
            }
            return sum;
        }

        public static double sum<TKey, TValue>(Dictionary<TKey, TValue> values)
        {
            return sum(values.Keys);
        }

        public static IEnumerable<double> slice(double start, double end, double step = 1)
        {
            for (double i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<float> slice(float start, float end, float step = 1)
        {
            for (float i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<int> slice(int start, int end, int step = 1)
        {
            for (int i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<int> slice(int range)
        {
            for (int i = 0; i < range; i++)
                yield return i;
        }

        public static bool hasattr(object obj, string key)
        {
            var __type__ = (obj).GetType();

            var __member__ = __type__.GetMembers();
            var __memberobject__ = __type__.GetMember(key);
            return (__memberobject__.Length > 0) ? true : false;
        }
        public delegate object __object__(params object[] args);
        public static __object__ getattr(object obj, string key, params Type[] ___parameter_type__)
        {
            var __dyn_obj__ = obj.GetType().GetMember(key);
            if (__dyn_obj__.Length == 0)
                throw new Exception("The object \"" + nameof(obj) + "\" doesnot have a defination \"" + key + "\"");
            var __type__ = __dyn_obj__[0];
            if (__type__.MemberType == System.Reflection.MemberTypes.Method)
            {
                try
                {
                    var __method__ = (___parameter_type__.Length > 0) ? obj.GetType().GetMethod(key, ___parameter_type__) : obj.GetType().GetMethod(key);
                    return (__object__)((object[] args) => __method__.Invoke(obj, args));
                }
                catch (System.Reflection.AmbiguousMatchException ex)
                {
                    throw new Exception("AmbigousFunctionMatchFound : (Probable cause : Function Overloading) Please add parameter types of the function.");
                }
            }
            else if (__type__.MemberType == System.Reflection.MemberTypes.Field)
            {
                var __field__ = (object)obj.GetType().GetField(key).GetValue(obj);
                return (__object__)((object[] args) => { return __field__; });
            }
            else if (__type__.MemberType == System.Reflection.MemberTypes.Property)
            {
                var __property__ = (object)obj.GetType().GetProperty(key).GetValue(obj);
                return (__object__)((object[] args) => { return __property__; });
            }
            return (__object__)((object[] args) => { return "NaN"; });
        }
    }

    public interface IPython : IDisposable
    {
        void __enter__();

        void __exit__();
    }

    public class PyObject<T> where T : IPyClass
    {
        public T Instance { get; set; }
    }
}
