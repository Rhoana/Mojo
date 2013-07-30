#include "PrimitiveMap.hpp"

namespace Mojo
{
namespace Native
{

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
    mFloat.clear();
    mInt.clear();
    mBool.clear();
}

}
}