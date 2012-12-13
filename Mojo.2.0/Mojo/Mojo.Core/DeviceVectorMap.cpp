#include "DeviceVectorMap.hpp"

namespace Mojo
{
namespace Core
{

void DeviceVectorMap::Set( std::string key, thrust::device_vector< float4 >& value )
{
    mFloat4[ key ] = value;
}

void DeviceVectorMap::Set( std::string key, thrust::device_vector< float2 >& value )
{
    mFloat2[ key ] = value;
}

void DeviceVectorMap::Set( std::string key, thrust::device_vector< uchar4 >& value )
{
    mUChar4[ key ] = value;
}

void DeviceVectorMap::Set( std::string key, thrust::device_vector< float >& value )
{
    mFloat[ key ] = value;
}

void DeviceVectorMap::Set( std::string key, thrust::device_vector< int >& value )
{
    mInt[ key ] = value;
}

void DeviceVectorMap::Set( std::string key, thrust::device_vector< char >& value )
{
    mChar[ key ] = value;
}

void DeviceVectorMap::EraseAll( std::string key )
{
    mFloat4.erase( key );
    mFloat2.erase( key );
    mUChar4.erase( key );
    mFloat.erase( key );
    mInt.erase( key );
    mChar.erase( key );
}

void DeviceVectorMap::Clear()
{
    mFloat4.clear();
    mFloat2.clear();
    mUChar4.clear();
    mFloat.clear();
    mInt.clear();
    mChar.clear();
}

}
}