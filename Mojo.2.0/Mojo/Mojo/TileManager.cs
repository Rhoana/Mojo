using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Xml.Serialization.GeneratedAssembly;
using Mojo.Interop;
using Mojo.Xml;
using SlimDX;
using SlimDX.DXGI;

namespace Mojo
{
    public class TileManager : NotifyPropertyChanged, IDisposable
    {
        public Interop.TileManager Internal { get; private set; }

        public bool TiledDatasetLoaded
        {
            get
            {
                if ( Internal != null )
                {
                    return Internal.IsTiledDatasetLoaded();
                }
                else
                {
                    return false;
                }
            }
        }

        public bool SegmentationLoaded
        {
            get
            {
                if ( Internal != null )
                {
                    return Internal.IsSegmentationLoaded();
                }
                else
                {
                    return false;
                }
            }
        }

        private TiledDatasetDescription mTiledDatasetDescription;
        public TiledDatasetDescription TiledDatasetDescription
        {
            get
            {
                return mTiledDatasetDescription;
            }
            set
            {
                mTiledDatasetDescription = value;
                OnPropertyChanged( "TiledDatasetDescription" );
            }
        }

        private TiledDatasetView mTiledDatasetView;
        public TiledDatasetView TiledDatasetView
        {
            get
            {
                return mTiledDatasetView;
            }
            set
            {
                mTiledDatasetView = value;
                OnPropertyChanged( "TiledDatasetView" );
            }
        }

        private bool mShowSegmentation = true;
        public bool ShowSegmentation
        {
            get
            {
                return SegmentationLoaded && mShowSegmentation;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mShowSegmentation = value;
                    OnPropertyChanged( "ShowSegmentation" );
                    OnPropertyChanged( "SegmentationVisibilityRatio" );
                }
            }
        }

        public void ToggleShowSegmentation()
        {
            if ( SegmentationLoaded )
            {
                ShowSegmentation = !ShowSegmentation;
            }
        }

        private float mSegmentationVisibilityRatio = 0.5f;
        public float SegmentationVisibilityRatio
        {
            get
            {
                return ( SegmentationLoaded && mShowSegmentation ) ? mSegmentationVisibilityRatio : 0f;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mSegmentationVisibilityRatio = value;
                    OnPropertyChanged( "SegmentationVisibilityRatio" );

                    //Someone is trying to change the segmentation visibility - make sure they can see it
                    if ( !mShowSegmentation )
                    {
                        mShowSegmentation = true;
                        OnPropertyChanged( "ShowSegmentation" );
                    }
                }
            }
        }

        private int mSelectedSegmentId = 0;
        public int SelectedSegmentId
        {
            get
            {
                return mSelectedSegmentId;
            }
            set
            {
                mSelectedSegmentId = value;
                OnPropertyChanged( "SelectedSegmentId" );
            }
        }

        private int mMouseOverSegmentId = 0;
        public int MouseOverSegmentId
        {
            get
            {
                return mMouseOverSegmentId;
            }
            set
            {
                mMouseOverSegmentId = value;
                OnPropertyChanged( "MouseOverSegmentId" );
            }
        }

        private float mMouseOverX = 0;
        public float MouseOverX
        {
            get
            {
                return mMouseOverX;
            }
            set
            {
                mMouseOverX = value;
                OnPropertyChanged( "MouseOverX" );
            }
        }

        private float mMouseOverY = 0;
        public float MouseOverY
        {
            get
            {
                return mMouseOverY;
            }
            set
            {
                mMouseOverY = value;
                OnPropertyChanged( "MouseOverY" );
            }
        }

        private bool mConstrainSegmentationMergeToCurrentSlice = true;
        public bool ConstrainSegmentationMergeToCurrentSlice
        {
            get
            {
                return mConstrainSegmentationMergeToCurrentSlice;
            }
            set
            {
                mConstrainSegmentationMergeToCurrentSlice = value;
                OnPropertyChanged( "ConstrainSegmentationMergeToCurrentSlice" );
            }
        }

        public void ToggleConstrainSegmentationMergeToCurrentSlice()
        {
            ConstrainSegmentationMergeToCurrentSlice = !ConstrainSegmentationMergeToCurrentSlice;
        }

        private bool mConstrainSegmentationMergeToConnectedComponent = true;
        public bool ConstrainSegmentationMergeToConnectedComponent
        {
            get
            {
                return mConstrainSegmentationMergeToConnectedComponent;
            }
            set
            {
                mConstrainSegmentationMergeToConnectedComponent = value;
                OnPropertyChanged( "ConstrainSegmentationMergeToConnectedComponent" );
            }
        }

        public void ToggleConstrainSegmentationMergeToConnectedComponent()
        {
            ConstrainSegmentationMergeToConnectedComponent = !ConstrainSegmentationMergeToConnectedComponent;
        }

        public TileManager( Interop.TileManager tileManager )
        {
            Internal = tileManager;
            TiledDatasetView = new TiledDatasetView();
        }

        public void Dispose()
        {
            UnloadTiledDataset();

            if ( Internal != null )
            {
                Internal.SaveAndClearFileSystemTileCache( );
                Internal.Dispose();
                Internal = null;
            }
        }

        public void Update()
        {
            if ( TiledDatasetLoaded )
            {
                Internal.LoadTiles( TiledDatasetView );
                Internal.Update();
            }
        }

        public void UpdateView()
        {
            OnPropertyChanged( "TiledDatasetView" );
        }

        public void LoadTiledDataset( string datasetRootDirectory )
        {
            if ( TiledDatasetLoaded )
            {
                UnloadTiledDataset();
            }

            Release.Assert( Directory.Exists( datasetRootDirectory ) );

            Release.Assert(
                datasetRootDirectory.EndsWith( Constants.DATASET_ROOT_DIRECTORY_NAME ) ||
                datasetRootDirectory.EndsWith( Constants.DATASET_ROOT_DIRECTORY_NAME + "\\" ) ||
                datasetRootDirectory.EndsWith( Constants.DATASET_ROOT_DIRECTORY_NAME + "/" ) );

            var sourceMapRootDirectory = Path.Combine( datasetRootDirectory, Constants.SOURCE_MAP_ROOT_DIRECTORY_NAME );

            var sourceMapTiledVolumeDescriptionPath = Path.Combine( datasetRootDirectory, Constants.SOURCE_MAP_TILED_VOLUME_DESCRIPTION_NAME );

            Release.Assert( Directory.Exists( sourceMapRootDirectory ) );

            var sourceMapTiledVolumeDescription = GetTiledVolumeDescription( sourceMapRootDirectory, sourceMapTiledVolumeDescriptionPath );

            //
            // Read in the default idMap settings
            // Required before LoadSegmentation so that D3D11_TEXTURE3D_DESC is set correctly in TileManager.hpp
            //
            var idMapRootDirectory = Path.Combine( datasetRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME );
            var idMapTiledVolumeDescriptionPath = Path.Combine( datasetRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME );

            Release.Assert( Directory.Exists( idMapRootDirectory ) );

            var idMapTiledVolumeDescription = GetTiledVolumeDescription( idMapRootDirectory, idMapTiledVolumeDescriptionPath );



            //var tempIdMapRootDirectory = Path.Combine( datasetRootDirectory, Constants.TEMP_ID_MAP_ROOT_DIRECTORY_NAME );

            //var idTileMapPath = Path.Combine( datasetRootDirectory, Constants.ID_TILE_MAP_PATH );
            //var idColorMapPath = Path.Combine( datasetRootDirectory, Constants.ID_COLOR_MAP_PATH );

            //var tempIdMapTiledVolumeDescription = GetTiledVolumeDescription( tempIdMapRootDirectory, idMapTiledVolumeDescriptionPath );

            //var idTileMapXml = XmlReader.ReadFromFile<idTileMap, idTileMapSerializer>( idTileMapPath );

            //var idTileMap = idTileMapXml.idTileMapEntry.ToDictionary(
            //                    idTileMapEntry => idTileMapEntry.id,
            //                    idTileMapEntry => (IList<Vector4>)idTileMapEntry.tiles.Select( tile => new Vector4( tile.x, tile.y, tile.z, tile.w ) ).ToList() );



            var tiledDatasetDescription = new TiledDatasetDescription
                                          {
                                              TiledVolumeDescriptions =
                                                  new ObservableDictionary<string, TiledVolumeDescription>
                                                  {
                                                      { "SourceMap", sourceMapTiledVolumeDescription },
                                                      { "IdMap", idMapTiledVolumeDescription },
                                                      { "SplitMap", idMapTiledVolumeDescription }
                                                      //,
                                                      //{ "TempIdMap", tempIdMapTiledVolumeDescription }
                                                  },
                                              Paths =
                                                  new ObservableDictionary<string, string>
                                                  {
                                                      //{ "IdColorMap", idColorMapPath },
                                                  },
                                              IdTileMap = null,
                                              MaxLabelId = 0
                                          };

            LoadTiledDataset( tiledDatasetDescription );

            UpdateView();

        }

        public void LoadSegmentation( string segmentationRootDirectory )
        {
            if ( SegmentationLoaded )
            {
                UnloadSegmentation();
            }

            Release.Assert( Directory.Exists( segmentationRootDirectory ) );

            var idMapRootDirectory = Path.Combine( segmentationRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME );
            var tempIdMapRootDirectory = Path.Combine( segmentationRootDirectory, Constants.TEMP_ID_MAP_ROOT_DIRECTORY_NAME );

            var idMapTiledVolumeDescriptionPath = Path.Combine( segmentationRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME );

            var idTileMapPath = Path.Combine( segmentationRootDirectory, Constants.ID_TILE_MAP_PATH );
            var idColorMapPath = Path.Combine( segmentationRootDirectory, Constants.ID_COLOR_MAP_PATH );

            Release.Assert( Directory.Exists( idMapRootDirectory ) );

            var idMapTiledVolumeDescription = GetTiledVolumeDescription( idMapRootDirectory, idMapTiledVolumeDescriptionPath );
            var tempIdMapTiledVolumeDescription = GetTiledVolumeDescription( tempIdMapRootDirectory, idMapTiledVolumeDescriptionPath );

            var idTileMapXml = XmlReader.ReadFromFile<idTileMap, idTileMapSerializer>( idTileMapPath );

            var idTileMap = idTileMapXml.idTileMapEntry.ToDictionary(
                                idTileMapEntry => idTileMapEntry.id,
                                idTileMapEntry => (IList<Vector4>)idTileMapEntry.tiles.Select( tile => new Vector4( tile.x, tile.y, tile.z, tile.w ) ).ToList() );

            TiledDatasetDescription.TiledVolumeDescriptions.Set( "IdMap", idMapTiledVolumeDescription );
            TiledDatasetDescription.TiledVolumeDescriptions.Set( "TempIdMap", tempIdMapTiledVolumeDescription );
            TiledDatasetDescription.Paths.Set( "IdColorMap", idColorMapPath );
            TiledDatasetDescription.IdTileMap = idTileMap;
            TiledDatasetDescription.MaxLabelId = idTileMap.Keys.Max();

            LoadSegmentation( TiledDatasetDescription );

            UpdateView();

        }

        private static TiledVolumeDescription GetTiledVolumeDescription( string mapRootDirectory, string tiledVolumeDescriptionPath )
        {
            var tiledVolumeDescriptionXml = XmlReader.ReadFromFile<tiledVolumeDescription, tiledVolumeDescriptionSerializer>( tiledVolumeDescriptionPath );
            var tiledVolumeDescription = new TiledVolumeDescription
                                         {
                                             ImageDataDirectory = mapRootDirectory,
                                             FileExtension = tiledVolumeDescriptionXml.fileExtension,
                                             NumTilesX = tiledVolumeDescriptionXml.numTilesX,
                                             NumTilesY = tiledVolumeDescriptionXml.numTilesY,
                                             NumTilesZ = tiledVolumeDescriptionXml.numTilesZ,
                                             NumTilesW = tiledVolumeDescriptionXml.numTilesW,
                                             NumVoxelsPerTileX = tiledVolumeDescriptionXml.numVoxelsPerTileX,
                                             NumVoxelsPerTileY = tiledVolumeDescriptionXml.numVoxelsPerTileY,
                                             NumVoxelsPerTileZ = tiledVolumeDescriptionXml.numVoxelsPerTileZ,
                                             NumVoxelsX = tiledVolumeDescriptionXml.numVoxelsX,
                                             NumVoxelsY = tiledVolumeDescriptionXml.numVoxelsY,
                                             NumVoxelsZ = tiledVolumeDescriptionXml.numVoxelsZ,
                                             DxgiFormat = (Format)Enum.Parse( typeof( Format ), tiledVolumeDescriptionXml.dxgiFormat ),
                                             NumBytesPerVoxel = tiledVolumeDescriptionXml.numBytesPerVoxel,
                                             IsSigned = tiledVolumeDescriptionXml.isSigned
                                         };
            return tiledVolumeDescription;
        }

        public void LoadTiledDataset( TiledDatasetDescription tiledDatasetDescription )
        {
            Internal.LoadTiledDataset( tiledDatasetDescription );

            if ( TiledDatasetLoaded )
            {
                TiledDatasetDescription = tiledDatasetDescription;
            }
        }

        public void LoadSegmentation( TiledDatasetDescription tiledDatasetDescription )
        {
            if ( TiledDatasetLoaded )
            {
                Internal.LoadSegmentation( tiledDatasetDescription );
                if ( SegmentationLoaded )
                {
                    TiledDatasetDescription = tiledDatasetDescription;                    
                }
            }            
        }

        public void UnloadTiledDataset()
        {
            if ( Internal != null )
            {
                if ( SegmentationLoaded )
                {
                    UnloadSegmentation();
                }

                Internal.UnloadTiledDataset();

                if ( !TiledDatasetLoaded )
                {
                    TiledDatasetDescription = new TiledDatasetDescription();
                }
            }
        }

        public void UnloadSegmentation()
        {
            if ( Internal != null )
            {
                Internal.UnloadSegmentation();
            }
        }

        public IList<TileCacheEntry> GetTileCache()
        {
            return Internal.GetTileCache().Where( tileCacheEntry => tileCacheEntry.Active ).OrderBy( tileCacheEntry => tileCacheEntry.CenterDataSpace.Z ).ToList();
        }

    }
}
