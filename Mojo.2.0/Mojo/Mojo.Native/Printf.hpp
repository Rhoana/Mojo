#pragma once

#include "Stl.hpp"
#include "ToString.hpp"

namespace Mojo
{
namespace Native
{
template < typename T00 >
void Printf( T00 t00 );

template < typename T00, typename T01 >
void Printf( T00 t00, T01 t01 );

template < typename T00, typename T01, typename T02 >
void Printf( T00 t00, T01 t01, T02 t02 );

template < typename T00, typename T01, typename T02, typename T03 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03 );

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15 );

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15, T16 t16 );

void PrintfHelper( std::string string );

template < typename T00 >
void Printf( T00 t00 )
{
    PrintfHelper( ToString( t00 ) );
}

template < typename T00, typename T01 >
void Printf( T00 t00, T01 t01 )
{
    PrintfHelper( ToString( t00, t01 ) );
}

template < typename T00, typename T01, typename T02 >
void Printf( T00 t00, T01 t01, T02 t02 )
{
    PrintfHelper( ToString( t00, t01, t02 ) );
}

template < typename T00, typename T01, typename T02, typename T03 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03 )
{
    PrintfHelper( ToString( t00, t01, t02, t03 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13 ) );
}


template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15 ) );
}

template < typename T00, typename T01, typename T02, typename T03, typename T04, typename T05, typename T06, typename T07, typename T08, typename T09, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16 >
void Printf( T00 t00, T01 t01, T02 t02, T03 t03, T04 t04, T05 t05, T06 t06, T07 t07, T08 t08, T09 t09, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15, T16 t16 )
{
    PrintfHelper( ToString( t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15, t16 ) );
}

}
}
