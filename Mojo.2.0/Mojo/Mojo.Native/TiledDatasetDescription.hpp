#pragma once

#include "Stl.hpp"
#include "HashMap.hpp"
#include "Types.hpp"

#include "TiledVolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

class TiledDatasetDescription
{
public:
    HashMap< std::string, TiledVolumeDescription > tiledVolumeDescriptions;
    HashMap< std::string, std::string >            paths;
    unsigned int                                   maxLabelId;
};

}
}