#include "TiledDatasetDescription.hpp"

#include "Mojo.Native/Stl.hpp"

#include <msclr/marshal_cppstd.h>

namespace Mojo
{
namespace Interop
{

Native::TiledDatasetDescription TiledDatasetDescription::ToNative()
{
    Native::TiledDatasetDescription tiledDatasetDescription;

    for each ( Collections::Generic::KeyValuePair< String^, TiledVolumeDescription^ > keyValuePair in TiledVolumeDescriptions )
    {
        tiledDatasetDescription.tiledVolumeDescriptions.Set( msclr::interop::marshal_as < std::string >( keyValuePair.Key ), keyValuePair.Value->ToNative() );
    }

    for each ( Collections::Generic::KeyValuePair< String^, String^ > keyValuePair in Paths )
    {
        tiledDatasetDescription.paths.Set( msclr::interop::marshal_as < std::string >( keyValuePair.Key ), msclr::interop::marshal_as < std::string >( keyValuePair.Value ) );
    }

    tiledDatasetDescription.maxLabelId = MaxLabelId;

    return tiledDatasetDescription;
}

}
}