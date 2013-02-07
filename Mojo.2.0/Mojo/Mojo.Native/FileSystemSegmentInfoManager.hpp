#pragma once

#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/Comparator.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>
#include <boost/pool/pool_alloc.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include "sqlite3.h"

namespace Mojo
{
namespace Native
{

typedef std::set< int4, Mojo::Core::Int4Comparator, boost::fast_pool_allocator< int4 > >   FileSystemTileSet;
typedef Core::HashMap< unsigned int, FileSystemTileSet >                                   FileSystemIdTileMap;
typedef stdext::hash_map< unsigned int, FileSystemTileSet >                                FileSystemIdTileMapDirect;

struct segmentInfo
{
  int         id;
  std::string name;
  int         size;

  segmentInfo( int id,const std::string& name, int size )
	  : id( id ), name( name ), size ( size ){}

  bool operator< (const segmentInfo& e) const
  {
	  return id < e.id;
  }
};

typedef boost::multi_index_container<
  segmentInfo,
  boost::multi_index::indexed_by<
    // index by id ( using ::operator< )
    boost::multi_index::ordered_unique< boost::multi_index::identity< segmentInfo > >,
    // index by name
    boost::multi_index::ordered_non_unique< boost::multi_index::member< segmentInfo, std::string, &segmentInfo::name > >,
	// index by size
	boost::multi_index::ordered_non_unique< boost::multi_index::member< segmentInfo, int, &segmentInfo::size > >
  > 
> SegmentMultiIndex;

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
	void                                              SetVoxelCount ( unsigned int segid, unsigned long voxelCount );
                                               
private:                                       

	FileSystemTileSet                                 LoadTileSet( unsigned int segid );

	std::string                                       mColorMapPath;
	hid_t                                             mColorMapHdf5FileHandle;

    std::string                                       mIdTileIndexDBPath;
    sqlite3                                           *mIdTileIndexDB;
    bool                                              mIsDBOpen;

	marray::Marray< unsigned int >                    mIdMax;
    marray::Marray< unsigned char >                   mIdColorMap;
	marray::Marray< unsigned int >                    mIdVoxelCountMap;

	FileSystemIdTileMap                               mCacheIdTileMap;

	SegmentMultiIndex                                 mSegmentMultiIndex;

};

}
}