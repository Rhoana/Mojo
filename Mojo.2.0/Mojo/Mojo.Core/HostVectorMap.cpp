#include "HostVectorMap.hpp"

namespace Mojo
{
namespace Core
{

void HostVectorMap::Set( std::string key, thrust::host_vector< float4 >& value )
{
    mFloat4[ key ] = value;
}

void HostVectorMap::Set( std::string key, thrust::host_vector< float2 >& value )
{
    mFloat2[ key ] = value;
}

void HostVectorMap::Set( std::string key, thrust::host_vector< uchar4 >& value )
{
    mUChar4[ key ] = value;
}

void HostVectorMap::Set( std::string key, thrust::host_vector< float >& value )
{
    mFloat[ key ] = value;
}

void HostVectorMap::Set( std::string key, thrust::host_vector< int >& value )
{
    mInt[ key ] = value;
}

void HostVectorMap::Set( std::string key, thrust::host_vector< char >& value )
{
    mChar[ key ] = value;
}

void HostVectorMap::Clear()
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