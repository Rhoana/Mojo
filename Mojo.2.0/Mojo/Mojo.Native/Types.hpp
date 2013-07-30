#pragma once

namespace Mojo
{
namespace Native
{

struct Int2
{

    int x;
    int y;

    Int2() : x ( 0 ), y ( 0 ){}

    Int2( int x, int y ) :
        x ( x ), y ( y ){}

};

struct Int3
{

    int x;
    int y;
    int z;

    Int3() : x ( 0 ), y ( 0 ), z ( 0 ){}

    Int3( int x, int y, int z ) :
        x ( x ), y ( y ), z ( z ){}

};

struct Int4
{

    int x;
    int y;
    int z;
    int w;

    Int4() : x ( 0 ), y ( 0 ), z ( 0 ), w ( 0 ){}

    Int4( int x, int y, int z, int w ) :
        x ( x ), y ( y ), z ( z ), w ( w ){}

};

struct Float2
{

    float x;
    float y;

    Float2() : x ( 0 ), y ( 0 ){}

    Float2( float x, float y ) :
        x ( x ), y ( y ){}

};

struct Float3
{

    float x;
    float y;
    float z;

    Float3() : x ( 0 ), y ( 0 ), z ( 0 ){}

    Float3( float x, float y, float z ) :
        x ( x ), y ( y ), z ( z ){}

};

struct Float4
{

    float x;
    float y;
    float z;
    float w;

    Float4() : x ( 0 ), y ( 0 ), z ( 0 ), w ( 0 ){}

    Float4( float x, float y, float z, float w ) :
        x ( x ), y ( y ), z ( z ), w ( w ){}

};

struct Int4Comparator
{
    bool operator() (const Int4 l, Int4 r) const
    {
        return l.w < r.w || ( l.w == r.w && (
               l.z < r.z || ( l.z == r.z && (
               l.y < r.y || ( l.y == r.y && l.x < r.x ) ) ) ) );
    }
};

struct Int3Comparator
{
    bool operator() (const Int3 l, Int3 r) const
    {
        return l.z < r.z || ( l.z == r.z && (
               l.y < r.y || ( l.y == r.y && l.x < r.x ) ) );
    }
};

struct Int2Comparator
{
    bool operator() (const Int2 l, Int2 r) const
    {
        return l.y < r.y || ( l.y == r.y && l.x < r.x );
    }
};

}
}