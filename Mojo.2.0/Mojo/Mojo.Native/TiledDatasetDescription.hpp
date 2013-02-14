#pragma once

#include "Mojo.Core/Stl.hpp"
//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/Comparator.hpp"

#include "TiledVolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

class TiledDatasetDescription
{
public:
    Core::HashMap< std::string, TiledVolumeDescription >                            tiledVolumeDescriptions;
    Core::HashMap< std::string, std::string >                                       paths;
	unsigned int                                                                    maxLabelId;
};

}
}