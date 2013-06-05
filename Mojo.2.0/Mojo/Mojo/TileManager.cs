using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Mojo.Interop;
using SlimDX;
using SlimDX.DXGI;
using System.Xml.Serialization;
using System.Xml;

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

        private bool mSegmentationChangeInProgress = false;
        public bool SegmentationChangeInProgress
        {
            get
            {
                return mSegmentationChangeInProgress;
            }
            set
            {
                mSegmentationChangeInProgress = value;
            }
        }


        private bool mSegmentationControlsEnabled = false;
        public bool SegmentationControlsEnabled
        {
            get
            {
                return mSegmentationControlsEnabled;
            }
            set
            {
                mSegmentationControlsEnabled = value;
                OnPropertyChanged( "SegmentationControlsEnabled" );
            }
        }

        private bool mNavigationControlsEnabled = false;
        public bool NavigationControlsEnabled
        {
            get
            {
                return mNavigationControlsEnabled;
            }
            set
            {
                mNavigationControlsEnabled = value;
                OnPropertyChanged( "NavigationControlsEnabled" );
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

        public void IncreaseSegmentationVisibility()
        {
            SegmentationVisibilityRatio = System.Math.Min( SegmentationVisibilityRatio + 0.1f, 1.0f );
        }

        public void DecreaseSegmentationVisibility()
        {
            SegmentationVisibilityRatio = System.Math.Max( SegmentationVisibilityRatio - 0.1f, 0f );
        }

        private bool mShowBoundaryLines = true;
        public bool ShowBoundaryLines
        {
            get
            {
                return SegmentationLoaded && mShowBoundaryLines;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mShowBoundaryLines = value;
                    OnPropertyChanged( "ShowBoundaryLines" );
                }
            }
        }

        public void ToggleShowBoundaryLines()
        {
            if ( SegmentationLoaded )
            {
                ShowBoundaryLines = !ShowBoundaryLines;
            }
        }

        private bool mJoinSplits3D = true;
        public bool JoinSplits3D
        {
            get
            {
                return SegmentationLoaded && mJoinSplits3D;
            }
            set
            {
                mJoinSplits3D = value;
                OnPropertyChanged( "JoinSplits3D" );
            }
        }

        public void ToggleJoinSplits3D()
        {
            if ( SegmentationLoaded )
            {
                JoinSplits3D = !JoinSplits3D;
            }
        }

        private float mSplitStartZ = 0;
        public float SplitStartZ
        {
            get
            {
                return mSplitStartZ;
            }
            set
            {
                mSplitStartZ = value;
                OnPropertyChanged( "SplitStartZ" );
            }
        }

        private uint mSelectedSegmentId = 0;
        public uint SelectedSegmentId
        {
            get
            {
                return mSelectedSegmentId;
            }
            set
            {
                if ( mSelectedSegmentId != value )
                {
                    SplitStartZ = TiledDatasetView.CenterDataSpace.Z;
                }
                mSelectedSegmentId = value;
                OnPropertyChanged( "SelectedSegmentId" );
            }
        }

        private uint mMouseOverSegmentId = 0;
        public uint MouseOverSegmentId
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

        private float mBrushSize = 10;
        public float BrushSize
        {
            get
            {
                return mBrushSize;
            }
            set
            {
                mBrushSize = value;
                OnPropertyChanged( "BrushSize" );
            }
        }

        public void IncreaseBrushSize()
        {
            if ( BrushSize < 32 )
            {
                BrushSize += 2;
            }
        }

        public void DecreaseBrushSize()
        {
            if ( BrushSize > 2 )
            {
                BrushSize -= 2;
            }
        }

        private float mMergeBrushSize = 20;
        public float MergeBrushSize
        {
            get
            {
                return mMergeBrushSize;
            }
            set
            {
                mMergeBrushSize = value;
                OnPropertyChanged( "MergeBrushSize" );
            }
        }

        public void IncreaseMergeBrushSize()
        {
            if ( MergeBrushSize < 64 )
            {
                MergeBrushSize += 4;
            }
        }

        public void DecreaseMergeBrushSize()
        {
            if ( MergeBrushSize > 4 )
            {
                MergeBrushSize -= 4;
            }
        }

        private MergeMode mCurrentMergeMode = MergeMode.GlobalReplace;
        public MergeMode CurrentMergeMode
        {
            get { return mCurrentMergeMode; }
            set
            {
                if ( mCurrentMergeMode != value )
                {
                    mCurrentMergeMode = value;
                    OnPropertyChanged( "CurrentMergeMode" );
                }
            }
        }

        private SplitMode mCurrentSplitMode = SplitMode.DrawSplit;
        public SplitMode CurrentSplitMode
        {
            get { return mCurrentSplitMode; }
            set
            {
                if ( mCurrentSplitMode != value )
                {
                    mCurrentSplitMode = value;
                    OnPropertyChanged( "CurrentSplitMode" );
                }
            }
        }

        private bool mAutoChangesMade = false;
        public bool AutoChangesMade
        {
            get { return mAutoChangesMade; }
            set
            {
                if ( mAutoChangesMade != value )
                {
                    mAutoChangesMade = value;
                    OnPropertyChanged( "AutoChangesMade" );
                }
            }
        }

        private bool mChangesMade = false;
        public bool ChangesMade
        {
            get { return mChangesMade; }
            set
            {
                if ( mChangesMade != value )
                {
                    mChangesMade = value;
                    OnPropertyChanged( "ChangesMade" );
                }
                AutoChangesMade = value;
            }
        }

        public void CommmitSplitChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                if ( CurrentSplitMode == SplitMode.JoinPoints )
                {
                    SegmentationChangeProgress = 10;
                    Internal.CompletePointSplit( SelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                    SegmentationChangeProgress = 100;
                }
                else
                {
                    SegmentationChangeProgress = 10;
                    Internal.CompleteDrawSplit( SelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ), mJoinSplits3D, (int)mSplitStartZ );
                    SegmentationChangeProgress = 100;
                }
                SplitStartZ = TiledDatasetView.CenterDataSpace.Z;
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CancelSplitChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.ResetSplitState( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );

                SegmentationChangeInProgress = false;
            }
        }

        public void CommmitAdjustChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                Internal.CommitAdjustChange( SelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                SegmentationChangeProgress = 100;

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CancelAdjustChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.ResetAdjustState( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );

                SegmentationChangeInProgress = false;
            }
        }

        public uint CommitDrawMerge()
        {
            uint result = 0;
            if ( SegmentationChangeInProgress ) return result;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return result;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                result = Internal.CommitDrawMerge( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                SegmentationChangeProgress = 100;

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
            return result;
        }

        public uint CommitDrawMergeCurrentSlice()
        {
            uint result = 0;
            if ( SegmentationChangeInProgress ) return result;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return result;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                result = Internal.CommitDrawMergeCurrentSlice( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                SegmentationChangeProgress = 100;

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
            return result;
        }

        public uint CommitDrawMergeCurrentConnectedComponent()
        {
            uint result = 0;
            if ( SegmentationChangeInProgress ) return result;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return result;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                result = Internal.CommitDrawMergeCurrentConnectedComponent( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                SegmentationChangeProgress = 100;

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
            return result;
        }

        public void UndoChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                Internal.UndoChange();
                SegmentationChangeProgress = 100;
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void RedoChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                SegmentationChangeProgress = 10;
                Internal.RedoChange();
                SegmentationChangeProgress = 100;
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void SelectNewId()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                SelectedSegmentId = Internal.GetNewId();

                SegmentationChangeInProgress = false;
            }
        }

        public void LockSegmentLabel( uint segId )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.LockSegmentLabel( segId );
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void UnlockSegmentLabel( uint segId )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.UnlockSegmentLabel( segId );
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        private double mSegmentationChangeProgress = 100;
        public double SegmentationChangeProgress
        {
            get
            {
                return mSegmentationChangeProgress;
            }
            set
            {
                mSegmentationChangeProgress = value;
                OnPropertyChanged( "SegmentationChangeProgress" );
            }
        }


        public TileManager( Interop.TileManager tileManager )
        {
            Internal = tileManager;
            TiledDatasetView = new TiledDatasetView();
        }

        public void Dispose()
        {
            UnloadTiledDataset();

            //
            // TileManager does not have any files to close or output to write
            // So we can just exit without calling Dispose here
            // This will save time on exit when deallocating memory is not necessary
            //

            //if ( Internal != null )
            //{
            //    Internal.Dispose();
            //    Internal = null;
            //}
        }

        public void Update()
        {
            if ( TiledDatasetLoaded )
            {
                Internal.LoadTiles( TiledDatasetView );
            }
        }

        public void UpdateOneTile()
        {
            if ( TiledDatasetLoaded )
            {
                Internal.LoadOverTile( TiledDatasetView );
            }
        }

        public void UpdateView()
        {
            OnPropertyChanged( "TiledDatasetView" );
            OnPropertyChanged( "SegmentationControlsEnabled" );
            OnPropertyChanged( "NavigationControlsEnabled" );
            OnPropertyChanged( "ShowSegmentation" );
            OnPropertyChanged( "SegmentationVisibilityRatio" );
            OnPropertyChanged( "ShowBoundaryLines" );
            OnPropertyChanged( "SelectedSegmentId" );
            OnPropertyChanged( "MouseOverSegmentId" );
            OnPropertyChanged( "CurrentMergeMode" );
            OnPropertyChanged( "CurrentSplitMode" );
            OnPropertyChanged( "JoinSplits3D" );
        }

        public void UpdateZ()
        {
            OnPropertyChanged( "TiledDatasetView" );
        }

        public void UpdateXYZ()
        {
            OnPropertyChanged( "TiledDatasetView" );
        }

        public void UpdateSegmentListFocus()
        {
            OnPropertyChanged( "SegmentListFocus" );
            OnPropertyChanged( "SelectedSegmentId" );
        }

        public void LoadTiledDataset( string datasetRootDirectory )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                if ( TiledDatasetLoaded )
                {
                    UnloadTiledDataset();
                }

                if ( !Directory.Exists( datasetRootDirectory ) )
                {
                    throw new Exception( "Dataset directory not found." );
                }

                var sourceMapRootDirectory = Path.Combine( datasetRootDirectory, Constants.SOURCE_MAP_ROOT_DIRECTORY_NAME );

                var sourceMapTiledVolumeDescriptionPath = Path.Combine( datasetRootDirectory, Constants.SOURCE_MAP_TILED_VOLUME_DESCRIPTION_NAME );

                if ( !Directory.Exists( sourceMapRootDirectory ) )
                {
                    throw new Exception( "Image subdirectory not found." );
                }

                var sourceMapTiledVolumeDescription = GetTiledVolumeDescription( sourceMapRootDirectory, sourceMapTiledVolumeDescriptionPath );

                //
                // Read in the default idMap settings
                // Required before LoadSegmentation so that D3D11_TEXTURE3D_DESC is set correctly in TileManager.hpp
                //
                var idMapRootDirectory = Path.Combine( datasetRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME );
                var idMapTiledVolumeDescriptionPath = Path.Combine( datasetRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME );

                if ( !Directory.Exists( idMapRootDirectory ) )
                {
                    throw new Exception( "Id subdirectory not found." );
                }

                var idMapTiledVolumeDescription = GetTiledVolumeDescription( idMapRootDirectory, idMapTiledVolumeDescriptionPath );

                var tiledDatasetDescription = new TiledDatasetDescription
                                              {
                                                  TiledVolumeDescriptions =
                                                      new ObservableDictionary<string, TiledVolumeDescription>
                                                      {
                                                          { "SourceMap", sourceMapTiledVolumeDescription },
                                                          { "IdMap", idMapTiledVolumeDescription },
                                                          { "OverlayMap", idMapTiledVolumeDescription }
                                                          //,
                                                          //{ "TempIdMap", tempIdMapTiledVolumeDescription }
                                                      },
                                                  Paths =
                                                      new ObservableDictionary<string, string>
                                                      {
                                                          //{ "IdColorMap", idColorMapPath },
                                                      },
                                                  MaxLabelId = 0
                                              };

                LoadTiledDataset( tiledDatasetDescription );

                NavigationControlsEnabled = true;

                UpdateView();

                ChangesMade = false;

                SegmentationChangeInProgress = false;
            }
        }

        public void LoadSegmentation( string segmentationRootDirectory )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                if ( SegmentationLoaded )
                {
                    UnloadSegmentation();
                }

                if ( !Directory.Exists( segmentationRootDirectory ) )
                {
                    throw new Exception( "Dataset directory not found." );
                }

                var idMapRootDirectory = Path.Combine( segmentationRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME );

                if ( !Directory.Exists( idMapRootDirectory ) )
                {
                    throw new Exception( "Id subdirectory not found." );
                }

                var tempIdMapRootDirectory = Path.Combine( segmentationRootDirectory, Constants.TEMP_ID_MAP_ROOT_DIRECTORY_NAME );
                var autosaveIdMapRootDirectory = Path.Combine( segmentationRootDirectory, Constants.AUTOSAVE_ID_MAP_ROOT_DIRECTORY_NAME );

                var idMapTiledVolumeDescriptionPath = Path.Combine( segmentationRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME );

                var idMapTiledVolumeDescription = GetTiledVolumeDescription( idMapRootDirectory, idMapTiledVolumeDescriptionPath );
                var tempIdMapTiledVolumeDescription = GetTiledVolumeDescription( tempIdMapRootDirectory, idMapTiledVolumeDescriptionPath );
                var autosaveIdMapTiledVolumeDescription = GetTiledVolumeDescription( autosaveIdMapRootDirectory, idMapTiledVolumeDescriptionPath );

                TiledDatasetDescription.TiledVolumeDescriptions.Set( "IdMap", idMapTiledVolumeDescription );
                TiledDatasetDescription.TiledVolumeDescriptions.Set( "TempIdMap", tempIdMapTiledVolumeDescription );
                TiledDatasetDescription.TiledVolumeDescriptions.Set( "AutosaveIdMap", autosaveIdMapTiledVolumeDescription );

                var colorMapPath = Path.Combine( segmentationRootDirectory, Constants.COLOR_MAP_PATH );
                TiledDatasetDescription.Paths.Set( "ColorMap", colorMapPath );

                var idTileIndexPath = Path.Combine( segmentationRootDirectory, Constants.SEGMENT_INFO_PATH );
                TiledDatasetDescription.Paths.Set( "SegmentInfo", idTileIndexPath );

                var logPath = Path.Combine( segmentationRootDirectory, Constants.LOG_PATH );
                TiledDatasetDescription.Paths.Set( "Log", logPath );

                LoadSegmentation( TiledDatasetDescription );

                SegmentationControlsEnabled = true;

                UpdateView();

                ChangesMade = false;

                SegmentationChangeInProgress = false;
            }
        }

        private static TiledVolumeDescription GetTiledVolumeDescription( string mapRootDirectory, string tiledVolumeDescriptionPath )
        {
            tiledVolumeDescription tiledVolumeDescriptionXml;

            try
            {
                XmlSerializer tiledVolumeDescriptionSerializer = new XmlSerializer( typeof( tiledVolumeDescription ) );

                // A FileStream is needed to read the XML document.
                using ( var fileStream = new FileStream( tiledVolumeDescriptionPath, FileMode.Open ) )
                {
                    System.Xml.XmlReader xmlTextReader = new XmlTextReader( fileStream );

                    // Declare an object variable of the type to be deserialized.
                    tiledVolumeDescriptionXml = (tiledVolumeDescription)tiledVolumeDescriptionSerializer.Deserialize( xmlTextReader );
                }
            }
            catch ( Exception e )
            {
                throw new Exception( "Error reading XML from " + tiledVolumeDescriptionPath + ":\n" + e.Message );
            }

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
            ChangesMade = false;
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
            ChangesMade = false;
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
            NavigationControlsEnabled = false;
            ChangesMade = false;
        }

        public void UnloadSegmentation()
        {
            if ( Internal != null )
            {
                Internal.UnloadSegmentation();
            }

            SegmentationControlsEnabled = false;
            ChangesMade = false;
        }

        public void SaveSegmentation()
        {
            lock ( this )
            {
                if ( Internal != null && SegmentationLoaded )
                {
                    SegmentationChangeProgress = 10;
                    Internal.SaveSegmentation();
                    SegmentationChangeProgress = 100;
                }
                ChangesMade = false;
            }
        }

        public void SaveSegmentationAs( string savePath )
        {
            lock ( this )
            {
                if ( Internal != null && SegmentationLoaded )
                {
                    SegmentationChangeProgress = 10;
                    Internal.SaveSegmentationAs( savePath );
                    SegmentationChangeProgress = 100;
                }
                ChangesMade = false;
            }
        }

        public void AutoSaveSegmentation()
        {
            lock ( this )
            {
                if ( Internal != null && SegmentationLoaded )
                {
                    SegmentationChangeProgress = 10;
                    Internal.AutosaveSegmentation();
                    SegmentationChangeProgress = 100;
                }
            }
        }

        public void DiscardChanges()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                if ( Internal != null && SegmentationLoaded )
                {
                    Internal.DeleteTempFiles();
                    //UnloadSegmentation();
                    //LoadSegmentation( TiledDatasetDescription );
                }

                SegmentationChangeInProgress = false;
            }
        }

        public IList<TileCacheEntry> GetTileCache()
        {
            return Internal.GetTileCache().Where( tileCacheEntry => tileCacheEntry.Active ).OrderBy( tileCacheEntry => tileCacheEntry.IndexTileSpace.W ).ToList();
        }

    }
}
