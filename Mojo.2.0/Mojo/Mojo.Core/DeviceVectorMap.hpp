#pragma once

#include "Stl.hpp"

#include "Thrust.hpp"
#include "Assert.hpp"

namespace Mojo
{
namespace Core
{

class DeviceVectorMap
{
public:
    template< typename TCudaType >
    thrust::device_vector< TCudaType >& Get( std::string key );

    void Set( std::string key, thrust::device_vector< float4 >& value );
    void Set( std::string key, thrust::device_vector< float2 >& value );
    void Set( std::string key, thrust::device_vector< uchar4 >& value );
    void Set( std::string key, thrust::device_vector< float >&  value );
    void Set( std::string key, thrust::device_vector< int >&    value );
    void Set( std::string key, thrust::device_vector< char >&   value );

	void EraseAll ( std::string key );

	template< typename TCudaType >
    stdext::hash_map< std::string, thrust::device_vector< TCudaType > >& GetHashMap();

    void Clear();

private:
    stdext::hash_map< std::string, thrust::device_vector< float4 > > mFloat4;
    stdext::hash_map< std::string, thrust::device_vector< float2 > > mFloat2;
    stdext::hash_map< std::string, thrust::device_vector< uchar4 > > mUChar4;
    stdext::hash_map< std::string, thrust::device_vector< float > >  mFloat;
    stdext::hash_map< std::string, thrust::device_vector< int > >    mInt;
    stdext::hash_map< std::string, thrust::device_vector< char > >   mChar;
};

template< typename TCudaType >
inline thrust::device_vector< TCudaType >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( 0 );
    thrust::device_vector< TCudaType > dummy;
    return dummy;
}

template<>
inline thrust::device_vector< float4 >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mFloat4.find( key ) != mFloat4.end() );
    return mFloat4[ key ];
}

template<>
inline thrust::device_vector< float2 >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mFloat2.find( key ) != mFloat2.end() );
    return mFloat2[ key ];
}

template<>
inline thrust::device_vector< uchar4 >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mUChar4.find( key ) != mUChar4.end() );
    return mUChar4[ key ];
}

template<>
inline thrust::device_vector< float >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mFloat.find( key ) != mFloat.end() );
    return mFloat[ key ];
}

template<>
inline thrust::device_vector< int >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mInt.find( key ) != mInt.end() );
    return mInt[ key ];
}

template<>
inline thrust::device_vector< char >& DeviceVectorMap::Get( std::string key )
{
    RELEASE_ASSERT( mChar.find( key ) != mChar.end() );
    return mChar[ key ];
}

template< typename TCudaType >
inline stdext::hash_map< std::string, thrust::device_vector< TCudaType > >& DeviceVectorMap::GetHashMap()
{
    RELEASE_ASSERT( 0 );
    stdext::hash_map< std::string, thrust::device_vector< TCudaType > > dummy;
    return dummy;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< float4 > >& DeviceVectorMap::GetHashMap()
{
    return mFloat4;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< float2 > >& DeviceVectorMap::GetHashMap()
{
    return mFloat2;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< uchar4 > >& DeviceVectorMap::GetHashMap()
{
    return mUChar4;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< float > >& DeviceVectorMap::GetHashMap()
{
    return mFloat;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< int > >& DeviceVectorMap::GetHashMap()
{
    return mInt;
}

template<>
inline stdext::hash_map< std::string, thrust::device_vector< char > >& DeviceVectorMap::GetHashMap()
{
    return mChar;
}

}
}