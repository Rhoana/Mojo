#pragma once

#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/Comparator.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>
#include <boost/pool/pool_alloc.hpp>
#include "sqlite3.h"

namespace Mojo
{
namespace Native
{

typedef std::set< int4, Mojo::Core::Int4Comparator, boost::fast_pool_allocator< int4 > >   FileSystemTileSet;
typedef Core::HashMap< unsigned int, FileSystemTileSet >                                   FileSystemIdTileMap;
typedef stdext::hash_map< unsigned int, FileSystemTileSet >                                FileSystemIdTileMapDirect;

class FileSystemIdIndex
{

public:
	FileSystemIdIndex();
	FileSystemIdIndex( std::string idInfoFilePath, std::string idTileIndexDBFilePath );

	void                                              Save();
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

    void                                              OpenDB();
	FileSystemTileSet                                 LoadTiles( unsigned int segid );

	std::string                                       mIdInfoPath;
	hid_t                                             mIdIndexHdf5FileHandle;
	hid_t                                             mIdTileMapGroupHandle;

    std::string                                       mIdTileIndexDBPath;
    sqlite3                                           *mIdTileIndexDB;
    bool                                              mIsDBOpen;

	marray::Marray< unsigned int >                    mIdMax;
    marray::Marray< unsigned char >                   mIdColorMap;
	marray::Marray< unsigned int >                    mIdVoxelCountMap;
	FileSystemIdTileMap                               mCacheIdTileMap;

};

}
}