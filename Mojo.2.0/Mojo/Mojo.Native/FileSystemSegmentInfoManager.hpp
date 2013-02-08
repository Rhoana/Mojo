#pragma once

#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/Comparator.hpp"

#include "SegmentInfo.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>
#include <boost/pool/pool_alloc.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include "sqlite3.h"

using boost::multi_index_container;
using namespace boost::multi_index;

namespace Mojo
{
namespace Native
{

typedef std::set< int4, Mojo::Core::Int4Comparator, boost::fast_pool_allocator< int4 > >   FileSystemTileSet;
typedef Core::HashMap< unsigned int, FileSystemTileSet >                                   FileSystemIdTileMap;
typedef stdext::hash_map< unsigned int, FileSystemTileSet >                                FileSystemIdTileMapDirect;

//
// Tags for accesing indicies
//
struct id{};
struct name{};
struct size{};
struct confidence{};

//
// multi_index_container to store all segment info and access quickly
//

typedef boost::multi_index_container<
  SegmentInfo,
  indexed_by<
        // random access index for sorting and index lookup
        random_access<>,
        // unique index by id
        ordered_unique< tag<id>, member< SegmentInfo, unsigned int, &SegmentInfo::id > >,
        // index by name
        ordered_non_unique< tag<name>, member< SegmentInfo, std::string, &SegmentInfo::name > >,
	    // index by size
	    ordered_non_unique< tag<size>, member< SegmentInfo, long, &SegmentInfo::size > >,
	    // index by locked
	    ordered_non_unique< tag<confidence>, member< SegmentInfo, char, &SegmentInfo::confidence > >
  > 
> SegmentMultiIndex;

typedef SegmentMultiIndex::index<id>::type SegmentMultiIndexById;
typedef SegmentMultiIndex::index<name>::type SegmentMultiIndexByName;
typedef SegmentMultiIndex::index<size>::type SegmentMultiIndexBySize;
typedef SegmentMultiIndex::index<confidence>::type SegmentMultiIndexByConfidence;

class FileSystemSegmentInfoManager
{

public:
	FileSystemSegmentInfoManager();
	FileSystemSegmentInfoManager( std::string idInfoFilePath, std::string idTileIndexDBFilePath );

	void                                              Save();
    void                                              OpenDB();
	void                                              CloseDB();

    marray::Marray< unsigned char >                   GetIdColorMap();
	FileSystemTileSet                                 GetTiles( unsigned int segid );
	unsigned int                                      GetTileCount ( unsigned int segid );
	unsigned int                                      GetVoxelCount ( unsigned int segid );
                                               
	unsigned int                                      GetMaxId();
	unsigned int                                      AddNewId();
                                               
	void                                              SetTiles( unsigned int segid, FileSystemTileSet tiles );
	void                                              SetVoxelCount ( unsigned int segid, long voxelCount );

    void                                              SortById( bool reverse );
    void                                              SortByName( bool reverse );
    void                                              SortBySize( bool reverse );
    void                                              SortByConfidence( bool reverse );

    std::list< SegmentInfo >                          GetSegmentInfoRange( unsigned int startIndex, unsigned int endIndex );
                                               
private:                                       

	FileSystemTileSet                                 LoadTileSet( unsigned int segid );

	std::string                                       mColorMapPath;
	hid_t                                             mColorMapHdf5FileHandle;

    std::string                                       mIdTileIndexDBPath;
    sqlite3                                           *mIdTileIndexDB;
    bool                                              mIsDBOpen;

    int                                               mCurrentSortIndex;

	unsigned int                                      mIdMax;
    marray::Marray< unsigned char >                   mIdColorMap;
	SegmentMultiIndex                                 mSegmentMultiIndex;
	FileSystemIdTileMap                               mCacheIdTileMap;


};

}
}