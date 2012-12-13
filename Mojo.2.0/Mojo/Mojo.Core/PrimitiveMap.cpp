#include "PrimitiveMap.hpp"

namespace Mojo
{
namespace Core
{

void PrimitiveMap::Set( std::string key, float x, float y, float z, float w )
{
    mFloat4[ key ] = make_float4( x, y, z, w );
}

void PrimitiveMap::Set( std::string key, float x, float y )
{
    mFloat2[ key ] = make_float2( x, y );
}

void PrimitiveMap::Set( std::string key, unsigned char x, unsigned char y, unsigned char z, unsigned char w )
{
    mUChar4[ key ] = make_uchar4( x, y, z, w );
}

void PrimitiveMap::Set( std::string key, float x )
{
    mFloat[ key ] = x;
}

void PrimitiveMap::Set( std::string key, int x )
{
    mInt[ key ] = x;
}

void PrimitiveMap::Set( std::string key, bool x )
{
    mBool[ key ] = x;
}

void PrimitiveMap::Clear()
{
    mFloat4.clear();
    mFloat2.clear();
    mUChar4.clear();
    mFloat.clear();
    mInt.clear();
    mBool.clear();
}

}
}