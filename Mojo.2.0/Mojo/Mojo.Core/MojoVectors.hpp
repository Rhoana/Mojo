#pragma once

namespace Mojo
{
namespace Core
{

struct MojoInt2
{

    int x;
    int y;

    MojoInt2() : x ( 0 ), y ( 0 ){}

    MojoInt2( int x, int y ) :
        x ( x ), y ( y ){}

};

struct MojoInt3
{

    int x;
    int y;
    int z;

    MojoInt3() : x ( 0 ), y ( 0 ), z ( 0 ){}

    MojoInt3( int x, int y, int z ) :
        x ( x ), y ( y ), z ( z ){}

};

struct MojoInt4
{

    int x;
    int y;
    int z;
    int w;

    MojoInt4() : x ( 0 ), y ( 0 ), z ( 0 ), w ( 0 ){}

    MojoInt4( int x, int y, int z, int w ) :
        x ( x ), y ( y ), z ( z ), w ( w ){}

};

struct MojoFloat2
{

    float x;
    float y;

    MojoFloat2() : x ( 0 ), y ( 0 ){}

    MojoFloat2( float x, float y ) :
        x ( x ), y ( y ){}

};

struct MojoFloat3
{

    float x;
    float y;
    float z;

    MojoFloat3() : x ( 0 ), y ( 0 ), z ( 0 ){}

    MojoFloat3( float x, float y, float z ) :
        x ( x ), y ( y ), z ( z ){}

};

struct MojoFloat4
{

    float x;
    float y;
    float z;
    float w;

    MojoFloat4() : x ( 0 ), y ( 0 ), z ( 0 ), w ( 0 ){}

    MojoFloat4( float x, float y, float z, float w ) :
        x ( x ), y ( y ), z ( z ), w ( w ){}

};

}
}