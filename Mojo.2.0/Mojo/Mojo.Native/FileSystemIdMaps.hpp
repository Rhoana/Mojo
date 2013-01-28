#pragma once

#include "Stl.hpp"
#include "HashMap.hpp"
#include "Comparator.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>
#include <boost/pool/pool_alloc.hpp>

namespace Mojo
{
namespace Native
{

typedef std::set< int4, Mojo::Core::Int4Comparator, boost::fast_pool_allocator< int4 > >   FileSystemTileSet;

class FileSystemIdMaps
{

public:
	FileSystemIdMaps( std::string idMapsPath );
	~FileSystemIdMaps();

	void                                           Save();
	void                                           SaveAs( std::string newIdMapsPath );
	void                                           Close();

	FileSystemTileSet                              GetTiles( int segid );
	int                                            GetTileCount ( int segid );
	int                                            GetVoxelCount ( int segid );


	int                                            GetMaxId();
	int                                            AddNewId();

	void                                           SetTiles( int segid, FileSystemTileSet tiles );
	void                                           SetTileCount ( int segid, int tileCount );
	void                                           SetVoxelCount ( int segid, int voxelCount );

private:
	std::string                                    mIdMapsPath;
	hid_t                                          mIdMapsHdf5FileHandle;
	int                                            mMaxId;
    marray::Marray< unsigned char >                mIdColorMap;
	marray::Marray< int >                          mIdVoxelMap;
	Core::HashMap< unsigned int, MojoTileSet >     mCacheIdTileMap;


}



}
}