#pragma once

#include "Stl.hpp"

#include "Assert.hpp"

namespace Mojo
{
namespace Native
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
    //
    // CODE QUALITY ISSUE:
    // This is absolutely the wrong approach. The whole point of HashMap is
    // to provide a layer of error checking on top of stdext::hash_map. This
    // assert has probably caught more bugs than any other line of code in the entire
    // Mojo codebase. Seeing this commented out is quite dissapointing. If you don't wan't
    // this kind of error checking, don't use a HashMap. -MR
    //
    //RELEASE_ASSERT( mHashMap.find( key ) != mHashMap.end() );

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