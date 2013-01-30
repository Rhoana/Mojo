using Mojo.Interop;
using SlimDX;

namespace Mojo
{
    public enum RecordingMode
    {
        NotRecording = 0,
        RecordingWithSoftConstraintsVisible = 1,
        RecordingWithSoftConstraintsInvisible = 2
    }

    public static class Constants
    {
        public static readonly PrimitiveMap ConstParameters =
            new PrimitiveMap
            {
                { PrimitiveType.Float4, "DUAL_MAP_INITIAL_VALUE", new Vector4( 0f, 0f, 0f, 0f ) },                 
                                                                                                                   
                { PrimitiveType.Float2, "OPTICAL_FLOW_MAP_INITIAL_VALUE", new Vector2( -1f, -1f ) },               
                                                                                                                   
                { PrimitiveType.UChar4, "COLOR_MAP_INITIAL_VALUE", new Vector4( 0f, 0f, 0f, 255f ) },       

                { PrimitiveType.Float, "MAX_CONVERGENCE_GAP", 999999f },                                           
                { PrimitiveType.Float, "MAX_CONVERGENCE_GAP_DELTA", 999999f },                                     

                { PrimitiveType.Float, "PRIMAL_MAP_INITIAL_VALUE", 0f },                                           
                { PrimitiveType.Float, "PRIMAL_MAP_FOREGROUND", 1f },                                              
                { PrimitiveType.Float, "PRIMAL_MAP_BACKGROUND", 0f },                                              
                { PrimitiveType.Float, "PRIMAL_MAP_THRESHOLD", 0.4f },                                             
                { PrimitiveType.Float, "PRIMAL_MAP_ERODE_NUM_PASSES", 5f },

                { PrimitiveType.Float, "OLD_PRIMAL_MAP_INITIAL_VALUE", 0f },                                       

                { PrimitiveType.Float, "EDGE_MAP_INITIAL_VALUE", 0f },                                             
                { PrimitiveType.Float, "EDGE_POWER_XY", 0.45f },                                                   
                { PrimitiveType.Float, "EDGE_MULTIPLIER", 10f },                                                   
                { PrimitiveType.Float, "EDGE_MAX_BEFORE_SATURATE", 0.4f },
                { PrimitiveType.Float, "EDGE_SPLIT_BOOST", 0.9f },

                { PrimitiveType.Float, "EDGE_STRENGTH_Z", 0.03f },                                                 
                { PrimitiveType.Float, "EDGE_POWER_Z", 1f },                                                       

                { PrimitiveType.Float, "STENCIL_MAP_INITIAL_VALUE", 0f },                                         
                { PrimitiveType.Float, "STENCIL_MAP_BACKGROUND_VALUE", 1f },                                         
                { PrimitiveType.Float, "STENCIL_MAP_STRONGEST_EDGE_VALUE", 2f },                                         
                { PrimitiveType.Float, "STENCIL_MAP_WEAKEST_EDGE_VALUE", 3f },

                { PrimitiveType.Float, "CONSTRAINT_MAP_INITIAL_VALUE", 0f },                                       
                { PrimitiveType.Float, "CONSTRAINT_MAP_HARD_FOREGROUND_USER", -100000f },                               
                { PrimitiveType.Float, "CONSTRAINT_MAP_HARD_BACKGROUND_USER", 100000f },
                { PrimitiveType.Float, "CONSTRAINT_MAP_HARD_FOREGROUND_AUTO", -99999f },                               
                { PrimitiveType.Float, "CONSTRAINT_MAP_HARD_BACKGROUND_AUTO", 99999f },        
                { PrimitiveType.Float, "CONSTRAINT_MAP_FALLOFF_GAUSSIAN_SIGMA", 0.12f },                           
                { PrimitiveType.Float, "CONSTRAINT_MAP_MIN_UPDATE_THRESHOLD", 1000f },                             
                { PrimitiveType.Float, "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_FOREGROUND", -10f }, 
                { PrimitiveType.Float, "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_BACKGROUND", 2000f },
                { PrimitiveType.Float, "CONSTRAINT_MAP_INITIALIZE_FROM_COST_MAP_MIN_FOREGROUND", -100f },          
                { PrimitiveType.Float, "CONSTRAINT_MAP_INITIALIZE_FROM_COST_MAP_MAX_BACKGROUND", 100f },           
                { PrimitiveType.Float, "CONSTRAINT_MAP_SPLIT_BACKGROUND_CONTRACT_NUM_PASSES", 1f },                                                                                 

                { PrimitiveType.Float, "COST_MAP_INITIAL_VALUE", 999999f },                                        
                { PrimitiveType.Float, "COST_MAP_MAX_VALUE", 999999f },                                            
                { PrimitiveType.Float, "COST_MAP_OPTICAL_FLOW_IMPORTANCE_FACTOR", 10f },                           
                { PrimitiveType.Float, "COST_MAP_INITIAL_MAX_FOREGROUND_COST_DELTA", 50f },                        

                { PrimitiveType.Float, "SCRATCHPAD_MAP_INITIAL_VALUE", 0f },                                       

                { PrimitiveType.Int, "ID_MAP_INITIAL_VALUE", 0 },
                { PrimitiveType.Int, "TILE_SIZE_X", 1 },
                { PrimitiveType.Int, "TILE_SIZE_Y", 1 },
                { PrimitiveType.Int, "TILE_SIZE_Z", 1 },

                { PrimitiveType.Bool, "SWAP_COLOR_CHANNELS", true },
                { PrimitiveType.Bool, "DIRECT_SCRIBBLE_PROPAGATION", false },
                { PrimitiveType.Bool, "DIRECT_ANISOTROPIC_TV", false },
                { PrimitiveType.Bool, "DUMP_EDGE_XY_MAP", false },
                { PrimitiveType.Bool, "DUMP_EDGE_Z_MAP", false },
                { PrimitiveType.Bool, "DUMP_CONSTRAINT_MAP", false },
                { PrimitiveType.Bool, "DUMP_PRIMAL_MAP", false },
                { PrimitiveType.Bool, "DUMP_COLOR_MAP", false },
                { PrimitiveType.Bool, "DUMP_ID_MAP", false }
            };

        public static readonly SegmentationLabelDescription DEFAULT_SEGMENTATION_LABEL =
            new SegmentationLabelDescription( 9999 )
            {
                Name = "Default Segmentation Label",
                Color = new Vector3( 0f, 255f, 255f )
            };

        public static readonly SegmentationLabelDescription NULL_SEGMENTATION_LABEL =
            new SegmentationLabelDescription( 0 )
            {
                Name = "NULL Segmentation Label",
                Color = new Vector3( 0f, 0f, 0f )
            };

        public static RecordingMode RECORDING_MODE = RecordingMode.RecordingWithSoftConstraintsInvisible;

        public static Color4 CLEAR_COLOR = new Color4( 0.5f, 0.5f, 1.0f );

        public const double MAGNIFICATION_STEP = 1.1;

        public const float ARROW_KEY_STEP = 100;
        public const float CONVERGENCE_GAP_THRESHOLD = 2.0f;
        public const float CONVERGENCE_DELTA_THRESHOLD = 0.0001f;

        public const int MAX_NUM_TINY_TEXT_CHARACTERS = 1024;

        public const int NUM_ITERATIONS_PER_VISUAL_UPDATE_HIGH_LATENCY_2D = 100;             
        public const int NUM_ITERATIONS_PER_VISUAL_UPDATE_LOW_LATENCY_2D = 20;               
                                                                                             
        public const int NUM_ITERATIONS_PER_VISUAL_UPDATE_HIGH_LATENCY_3D = 50;
        public const int NUM_ITERATIONS_PER_VISUAL_UPDATE_LOW_LATENCY_3D = 10;

        public const int MAX_BRUSH_WIDTH = 50;                                               
        public const int MIN_BRUSH_WIDTH = 8;                                  
        public const int NUM_DETENTS_PER_WHEEL_MOVE = 120;                                   
        public const int NUM_CONSTRAINT_MAP_DILATION_PASSES_INITIALIZE_NEW_PROCESS = 0;

        public static bool DEBUG_D3D11_DEVICE = false;
        public static bool DIRECT_SCRIBBLE_PROPAGATION = false;

        public static readonly string DATASET_ROOT_DIRECTORY_NAME = "mojo";
        public static readonly string SOURCE_MAP_ROOT_DIRECTORY_NAME = @"images\tiles";
        public static readonly string SOURCE_MAP_TILED_VOLUME_DESCRIPTION_NAME = @"images\tiledVolumeDescription.xml";
        public static readonly string ID_MAP_ROOT_DIRECTORY_NAME = @"ids\tiles";
        public static readonly string ID_MAP_TILED_VOLUME_DESCRIPTION_NAME = @"ids\tiledVolumeDescription.xml";
        public static readonly string FILTERED_SOURCE_MAP_ROOT_DIRECTORY_NAME = @"probabilities\tiles";
        public static readonly string OPTICAL_FLOW_FORWARD_MAP_ROOT_DIRECTORY_NAME = @"opticalflow\forward\tiles";
        public static readonly string OPTICAL_FLOW_BACKWARD_MAP_ROOT_DIRECTORY_NAME = @"opticalflow\backward\tiles";

        //public static readonly string ID_TILE_MAP_PATH = @"idTileMap\idTileMap.hdf5";
        //public static readonly string ID_COLOR_MAP_PATH = @"idColorMap\idColorMap.hdf5";
        public static readonly string ID_INDEX_PATH = @"ids\idIndex.hdf5";
        public static readonly string TEMP_ID_INDEX_PATH = @"temp\idIndex.hdf5";

        public static readonly string TEMP_ID_MAP_ROOT_DIRECTORY_NAME = @"temp\ids\tiles";
        //public static readonly string TEMP_ID_TILE_MAP_PATH = @"temp\idTileMap\idTileMap.hdf5";

        public static readonly string AUTOSAVE_ID_MAP_ROOT_DIRECTORY_NAME = @"autosave\ids\tiles";
        //public static readonly string AUTOSAVE_ID_TILE_MAP_PATH = @"autosave\idTileMap\idTileMap.hdf5";

    }
}
