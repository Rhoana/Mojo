#pragma once

#include "Stl.hpp"
#include "D3D11.hpp"
#include "Assert.hpp"

namespace Mojo
{
namespace Native
{

class PrimitiveMap
{
public:
    template< typename T >
    T Get( std::string key );

    void Set( std::string key, float x );
    void Set( std::string key, int   x );
    void Set( std::string key, bool  x );

    template< typename T >
    stdext::hash_map< std::string, T >& GetHashMap();
    
    void Clear();

private:
    stdext::hash_map< std::string, float > mFloat;
    stdext::hash_map< std::string, int >   mInt;
    stdext::hash_map< std::string, bool >  mBool;
};

template< typename T >
inline T PrimitiveMap::Get( std::string key )
{
    RELEASE_ASSERT( 0 );
    T dummy;
    return dummy;
};

template<>
inline float PrimitiveMap::Get( std::string key )
{
    RELEASE_ASSERT( mFloat.find( key ) != mFloat.end() );
    return mFloat[ key ];
};

template<>
inline int PrimitiveMap::Get( std::string key )
{
    RELEASE_ASSERT( mInt.find( key ) != mInt.end() );
    return mInt[ key ];
};

template<>
inline bool  PrimitiveMap::Get( std::string key )
{
    RELEASE_ASSERT( mBool.find( key ) != mBool.end() );
    return mBool[ key ];
};

template< typename T >
inline stdext::hash_map< std::string, T >& PrimitiveMap::GetHashMap()
{
    RELEASE_ASSERT( 0 );
    stdext::hash_map< std::string, T > dummy;
    return dummy;
};

template<>
inline stdext::hash_map< std::string, float >& PrimitiveMap::GetHashMap()
{
    return mFloat;
};

template<>
inline stdext::hash_map< std::string, int >& PrimitiveMap::GetHashMap()
{
    return mInt;
};

template<>
inline stdext::hash_map< std::string, bool >& PrimitiveMap::GetHashMap()
{
    return mBool;
};

}
}