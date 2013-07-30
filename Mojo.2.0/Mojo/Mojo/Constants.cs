using Mojo.Interop;
using SlimDX;

namespace Mojo
{
    public static class Constants
    {
        public static readonly PrimitiveMap ConstParameters =
            new PrimitiveMap
            {
                { PrimitiveType.Int, "TILE_PIXELS_X", 512 },
                { PrimitiveType.Int, "TILE_PIXELS_Y", 512 },
                { PrimitiveType.Int, "TILE_PIXELS_Z", 1 },

                { PrimitiveType.Int, "TILE_SIZE_X", 1 },
                { PrimitiveType.Int, "TILE_SIZE_Y", 1 },
                { PrimitiveType.Int, "TILE_SIZE_Z", 1 },

                { PrimitiveType.Int, "ID_MAP_NUM_BYTES_PER_VOXEL", 4 },
                { PrimitiveType.Int, "OVERLAY_MAP_NUM_BYTES_PER_VOXEL", 4 },
            };

        public const double MAIN_WINDOW_WIDTH = 1200;

        public static Color4 CLEAR_COLOR = new Color4( 0.5f, 0.5f, 1.0f );

        public const double MAGNIFICATION_STEP = 1.1;
        public const float ARROW_KEY_STEP = 100;
        public const int NUM_DETENTS_PER_WHEEL_MOVE = 120;

        public const int MAX_NUM_TINY_TEXT_CHARACTERS = 1024;

        public static bool DEBUG_D3D11_DEVICE = false;

        public static readonly string MAIN_WINDOW_BASE_TITLE = "Mojo v2.0";

        public static readonly string SOURCE_IMAGES_FILE_NAME_EXTENSION = "mojoimg";
        public static readonly string SEGMENTATION_FILE_NAME_EXTENSION = "mojoseg";

        public static readonly string SOURCE_IMAGES_ROOT_DIRECTORY_NAME_SUFFIX = "_mojoimg";
        public static readonly string SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX = "_mojoseg";

        public static readonly string SOURCE_MAP_ROOT_DIRECTORY_NAME = @"images\tiles";
        public static readonly string SOURCE_MAP_TILED_VOLUME_DESCRIPTION_NAME = @"images\tiledVolumeDescription.xml";

        public static readonly string ID_MAP_ROOT_DIRECTORY_NAME = @"ids\tiles";
        public static readonly string ID_MAP_TILED_VOLUME_DESCRIPTION_NAME = @"ids\tiledVolumeDescription.xml";
        public static readonly string COLOR_MAP_PATH = @"ids\colorMap.hdf5";
        public static readonly string SEGMENT_INFO_PATH = @"ids\segmentInfo.db";
        public static readonly string LOG_PATH = @"ids\changelog.txt";

        public static readonly string TEMP_ROOT_DIRECTORY_NAME = @"temp";
        public static readonly string TEMP_ID_MAP_ROOT_DIRECTORY_NAME = @"temp\ids\tiles";
        public static readonly string TEMP_SEGMENT_INFO_PATH = @"temp\ids\segmentInfo.db";
        public static readonly string TEMP_COLOR_MAP_PATH = @"temp\ids\colorMap.hdf5";

        public static readonly string AUTOSAVE_ID_MAP_ROOT_DIRECTORY_NAME = @"autosave\ids\tiles";
    }
}
