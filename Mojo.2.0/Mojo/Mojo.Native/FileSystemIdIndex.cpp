#include "FileSystemIdIndex.hpp"

#include "Mojo.Core/Boost.hpp"

namespace Mojo
{
namespace Native
{

FileSystemIdIndex::FileSystemIdIndex()
{
}

FileSystemIdIndex::FileSystemIdIndex( std::string idIndexFilePath )
{
    mIdIndexPath = idIndexFilePath;
    mIdIndexHdf5FileHandle = marray::hdf5::openFile( mIdIndexPath );

    marray::hdf5::load( mIdIndexHdf5FileHandle, "idMax", mIdMax );
    marray::hdf5::load( mIdIndexHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::load( mIdIndexHdf5FileHandle, "idVoxelCountMap", mIdVoxelCountMap );

    marray::hdf5::closeFile( mIdIndexHdf5FileHandle );
}

FileSystemIdIndex::~FileSystemIdIndex()
{
    Close();
}

void FileSystemIdIndex::Close()
{
    //if ( mIdTileMapGroupHandle != 0 )
    //{
    //    marray::hdf5::closeGroup( mIdTileMapGroupHandle );
    //    mIdTileMapGroupHandle = 0;
    //}
    //if ( mIdIndexHdf5FileHandle != 0 )
    //{
    //    marray::hdf5::closeFile( mIdIndexHdf5FileHandle );
    //    mIdIndexHdf5FileHandle = 0;
    //}
}

void FileSystemIdIndex::SaveAs( std::string newPath )
{
    boost::filesystem::path tempIdIndexPath = boost::filesystem::path( newPath );
    if ( !boost::filesystem::exists( tempIdIndexPath ) )
    {
        boost::filesystem::create_directories( tempIdIndexPath.parent_path() );
    }

    mIdIndexHdf5FileHandle = marray::hdf5::openFile( mIdIndexPath );
    hid_t newIdMapsFile = marray::hdf5::createFile( newPath );

    Core::Printf("Saving id max / colors / voxels.");

    marray::hdf5::save( newIdMapsFile, "idMax", mIdMax );
    marray::hdf5::save( newIdMapsFile, "idColorMap", mIdColorMap );
    marray::hdf5::save( newIdMapsFile, "idVoxelCountMap", mIdVoxelCountMap );

    mIdTileMapGroupHandle = marray::hdf5::openGroup( mIdIndexHdf5FileHandle, "idTileMap" );
    hid_t newIdMapsGroup = marray::hdf5::createGroup( newIdMapsFile, "idTileMap" );

    std::string idString;
    std::stringstream converter;
    marray::Marray< unsigned int > rawIdTileMap;

    Core::Printf( "Saving id tile maps." );

    for ( unsigned int segidi = 0; segidi <= mIdMax(0); ++segidi )
    {
        FileSystemIdTileMapDirect::iterator idIt = mCacheIdTileMap.GetHashMap().find( segidi );

        if ( idIt == mCacheIdTileMap.GetHashMap().end() )
        {

            converter.str("");
            converter << segidi;
            idString = converter.str();

            if ( H5Lexists( mIdTileMapGroupHandle, idString.c_str(), H5P_DEFAULT ) )
            {
                marray::hdf5::load( mIdTileMapGroupHandle, idString, rawIdTileMap );

                marray::hdf5::save( newIdMapsGroup, idString, rawIdTileMap );

                //Core::Printf( "Saved ", idString, " from existing index." );
            }

        }
        else if ( idIt->second.size() > 0 )
        {

            size_t shape[] = { idIt->second.size(), 4 };
            rawIdTileMap = marray::Marray< unsigned int >( shape, shape + 2, 0, marray::FirstMajorOrder );

            unsigned int i = 0;
            for ( FileSystemTileSet::iterator tileIt = idIt->second.begin(); tileIt != idIt->second.end(); ++tileIt )
            {
                rawIdTileMap( i, 0 ) = tileIt->w;
                rawIdTileMap( i, 1 ) = tileIt->z;
                rawIdTileMap( i, 2 ) = tileIt->y;
                rawIdTileMap( i, 3 ) = tileIt->x;
                ++i;
            }

            converter.str("");
            converter << segidi;
            idString = converter.str();

            marray::hdf5::save( newIdMapsGroup, idString, rawIdTileMap );

            //Core::Printf( "Saved ", idString, " from cache." );

        }
    }

    Core::Printf("Finished saving id maps.");

    marray::hdf5::closeGroup( newIdMapsGroup );
    marray::hdf5::closeFile( newIdMapsFile );

    marray::hdf5::closeGroup( mIdTileMapGroupHandle );
    marray::hdf5::closeFile( mIdIndexHdf5FileHandle );

}

marray::Marray< unsigned char > FileSystemIdIndex::GetIdColorMap()
{
    //marray::Marray< unsigned char > idColorMap;
    //marray::hdf5::load( mIdIndexHdf5FileHandle, "idColorMap", idColorMap );
    //return idColorMap;
    return mIdColorMap;
}
                                               
FileSystemTileSet FileSystemIdIndex::GetTiles( unsigned int segid )
{

    if ( mCacheIdTileMap.GetHashMap().find( segid ) == mCacheIdTileMap.GetHashMap().end() )
    {
        //
        // Load the tile id map from the hdf5
        //
        std::string idString;
        std::stringstream converter;
        marray::Marray< unsigned int > rawIdTileMap;

        converter << segid;
        idString = converter.str();

        mIdIndexHdf5FileHandle = marray::hdf5::openFile( mIdIndexPath );
        mIdTileMapGroupHandle = marray::hdf5::openGroup( mIdIndexHdf5FileHandle, "idTileMap" );

        marray::hdf5::load( mIdTileMapGroupHandle, idString, rawIdTileMap );

        marray::hdf5::closeGroup( mIdTileMapGroupHandle );
        marray::hdf5::closeFile( mIdIndexHdf5FileHandle );

        unsigned int i;
        for ( i = 0; i < rawIdTileMap.shape( 0 ); i++ )
        {
            mCacheIdTileMap.GetHashMap()[ segid ].insert( make_int4( rawIdTileMap( i, 3 ), rawIdTileMap( i, 2 ), rawIdTileMap( i, 1 ), rawIdTileMap( i, 0 ) ) );
        }
    }

    return mCacheIdTileMap.Get( segid );

}

unsigned int FileSystemIdIndex::GetTileCount ( unsigned int segid )
{
    return GetTiles( segid ).size();
}

unsigned int FileSystemIdIndex::GetVoxelCount ( unsigned int segid )
{
    return mIdVoxelCountMap( segid );
}
                                               
unsigned int FileSystemIdIndex::GetMaxId()
{
    return mIdMax( 0 );
}

unsigned int FileSystemIdIndex::AddNewId()
{
    mIdMax( 0 ) = mIdMax( 0 ) + 1;

    size_t shape[] = { mIdMax( 0 ) + 1 };
    mIdVoxelCountMap.resize( shape, shape + 1 );
    mIdVoxelCountMap( mIdMax(0) ) = 0;

    return mIdMax( 0 );
}
                                               
void FileSystemIdIndex::SetTiles( unsigned int segid, FileSystemTileSet tiles )
{
    mCacheIdTileMap.GetHashMap()[ segid ] = tiles;
}

void FileSystemIdIndex::SetVoxelCount ( unsigned int segid, unsigned long voxelCount )
{
    mIdVoxelCountMap( segid ) = voxelCount;
}


}
}
