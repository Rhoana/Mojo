using System;
using System.Collections.Generic;
using System.Linq;

namespace Mojo
{
    public static class LinqUtil
    {
        public static TSrc ArgMin<TSrc, TArg>( this IEnumerable<TSrc> ie, Converter<TSrc, TArg> fn ) where TArg : IComparable<TArg>
        {
            var e = ie.GetEnumerator();
            if ( !e.MoveNext() )
                throw new InvalidOperationException( "Sequence has no elements." );

            TSrc t = e.Current;
            if ( !e.MoveNext() )
                return t;

            TArg minVal = fn( t );
            do
            {
                TSrc tTry;
                TArg v;
                if ( ( v = fn( tTry = e.Current ) ).CompareTo( minVal ) < 0 )
                {
                    t = tTry;
                    minVal = v;
                }
            }
            while ( e.MoveNext() );
            return t;
        }

        public static IEnumerable<TResult> Zip<TFirst, TSecond, TResult>( this IEnumerable<TFirst> first, IEnumerable<TSecond> second, Func<TFirst, TSecond, TResult> func )
        {
            return first.Select((x, i) => new { X = x, I = i }).Join(second.Select((x, i) => new { X = x, I = i }), o => o.I, i => i.I, (o, i) => func(o.X, i.X));
        }
    }
}