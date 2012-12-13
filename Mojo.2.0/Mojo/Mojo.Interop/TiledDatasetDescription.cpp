#include "TiledDatasetDescription.hpp"

#include "Mojo.Core/Stl.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Core/ForEach.hpp"

namespace Mojo
{
namespace Interop
{


Native::TiledDatasetDescription TiledDatasetDescription::ToNative()
{
    Native::TiledDatasetDescription tiledDatasetDescription;

    for each ( Collections::Generic::KeyValuePair< String^, TiledVolumeDescription^ > keyValuePair in TiledVolumeDescriptions )
    {
        //tiledDatasetDescription.tiledVolumeDescriptions.Set( msclr::interop::marshal_as < std::string >( keyValuePair.Key ), keyValuePair.Value->ToNative() );
        tiledDatasetDescription.tiledVolumeDescriptions.GetHashMap()[msclr::interop::marshal_as < std::string >( keyValuePair.Key )] = keyValuePair.Value->ToNative();
    }

	if ( IdTileMap )
	{
		for each ( Collections::Generic::KeyValuePair< unsigned int, Collections::Generic::IList< Vector4 >^ > keyValuePair in IdTileMap )
		{
			std::set< int4, Mojo::Core::Int4Comparator > tiles;

			for each ( Vector4 tile in keyValuePair.Value )
			{
				tiles.insert( make_int4( (int)tile.X, (int)tile.Y, (int)tile.Z, (int)tile.W ) );
			}

			//tiledDatasetDescription.idTileMap.Set( keyValuePair.Key, tiles );
			tiledDatasetDescription.idTileMap.GetHashMap()[keyValuePair.Key] = tiles;
		}
	}

    for each ( Collections::Generic::KeyValuePair< String^, String^ > keyValuePair in Paths )
    {
        //tiledDatasetDescription.paths.Set( msclr::interop::marshal_as < std::string >( keyValuePair.Key ), msclr::interop::marshal_as < std::string >( keyValuePair.Value ) );
        tiledDatasetDescription.paths.GetHashMap()[msclr::interop::marshal_as < std::string >( keyValuePair.Key )] = msclr::interop::marshal_as < std::string >( keyValuePair.Value );
    }

	tiledDatasetDescription.maxLabelId = MaxLabelId;

    return tiledDatasetDescription;
}

}
}