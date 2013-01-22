#pragma once

#include "Stl.hpp"
#include "HashMap.hpp"
#include "Comparator.hpp"

#include <boost/pool/pool_alloc.hpp>

namespace Mojo
{
namespace Core
{

typedef std::set< int4, Mojo::Core::Int4Comparator, boost::fast_pool_allocator< int4 > >   MojoTileSet;
typedef Core::HashMap< unsigned int, MojoTileSet >     IdTileMap;

}
}