#pragma once

//#include "vld.h"

#include "Stl.hpp"

#include "Assert.hpp"

namespace Mojo
{
namespace Core
{

template< typename K, typename V >
class HashMap
{
public:
    V&                        Get( const K& key );
    void                      Set( const K& key, const V& value );

    stdext::hash_map< K, V >& GetHashMap();

private:
    stdext::hash_map< K, V > mHashMap;
};

template< typename K, typename V >
inline V& HashMap< K, V >::Get( const K& key )
{
    RELEASE_ASSERT( mHashMap.find( key ) != mHashMap.end() );
    return mHashMap[ key ];
};

template< typename K, typename V >
inline void HashMap< K, V >::Set( const K& key, const V& value )
{
    mHashMap[ key ] = value;
};

template< typename K, typename V >
inline stdext::hash_map< K, V >& HashMap< K, V >::GetHashMap()
{
    return mHashMap;
};

}
}