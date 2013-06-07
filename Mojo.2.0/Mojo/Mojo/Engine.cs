using System;
using System.Linq;
using System.Windows;
using Mojo.Interop;
using SlimDX.DXGI;

namespace Mojo
{
    public enum ViewerMode
    {
        TileManager2D
    }

    public enum ToolMode
    {
        Null,
        AdjustSegmentation,
        MergeSegmentation,
        DrawMergeSegmentation,
        SplitSegmentation,
    }

    public enum MergeMode
    {
        Fill2D,
        Fill3D,
        GlobalReplace
    }

    public enum MergeControlMode
    {
        Draw,
        Click
    }

    public enum SplitMode
    {
        JoinPoints,
        DrawSplit,
        DrawRegions
    }

    public class Engine : NotifyPropertyChanged, IDisposable
    {
        private Factory mDxgiFactory;
        private SlimDX.Direct3D11.Device mD3D11Device;

        public TileManager TileManager { get; private set; }

        public ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, ITool>> Tools { get; private set; }
        public ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, IRenderingStrategy>> RenderingStrategies { get; private set; }
        public ObservableDictionary<ViewerMode, Viewer> Viewers { get; private set; }

        private ToolMode mCurrentToolMode;
        private bool mToolModeChanging = false;

        public ToolMode CurrentToolMode
        {
            get { return mCurrentToolMode; }
            set
            {

                /*
                 * TODO: remove mToolModeChanging (Currently required because a call loop can be created on keyboard tool change).
                 * On change from Split to Merge mode using keyboard shortcut this method is called twice and ends up back in Split mode.
                 */

                if ( value == ToolMode.MergeSegmentation && CurrentMergeControlMode == MergeControlMode.Draw )
                {
                    value = ToolMode.DrawMergeSegmentation;
                }

                if ( value != mCurrentToolMode && !mToolModeChanging )
                {
                    mToolModeChanging = true;
                    mCurrentToolMode = value;

                    Tools.Internal.ToList().ForEach( viewerModeToolsMap => viewerModeToolsMap.Value.Internal[mCurrentToolMode].Select() );

                    Viewers.Internal.ToList()
                           .ForEach( viewer => viewer.Value.D3D11RenderingPane.RenderingStrategy = RenderingStrategies.Internal[viewer.Key].Internal[mCurrentToolMode] );
                    Viewers.Internal.ToList().ForEach( viewer => viewer.Value.UserInputHandler = Tools.Internal[viewer.Key].Internal[mCurrentToolMode] );

                    OnPropertyChanged( "CurrentToolMode" );
                    mToolModeChanging = false;
                }
            }
        }

        private MergeControlMode mCurrentMergeControlMode = MergeControlMode.Draw;
        public MergeControlMode CurrentMergeControlMode
        {
            get { return mCurrentMergeControlMode; }
            set
            {
                if ( mCurrentMergeControlMode != value )
                {
                    mCurrentMergeControlMode = value;
                    if ( CurrentToolMode == ToolMode.DrawMergeSegmentation || CurrentToolMode == ToolMode.MergeSegmentation )
                    {
                        CurrentToolMode = ToolMode.MergeSegmentation;
                    }
                    OnPropertyChanged( "CurrentMergeControlMode" );
                }
            }
        }

        public Engine( ObservableDictionary<string, D3D11HwndDescription> d3d11HwndDescriptions )
        {
            Console.WriteLine( "\nMojo initializing...\n" );

            try
            {

                D3D11.Initialize( out mDxgiFactory, out mD3D11Device );
                //Cuda.Initialize( mD3D11Device );
                //Thrust.Initialize();

                TileManager = new TileManager( new Interop.TileManager( mD3D11Device, mD3D11Device.ImmediateContext, Constants.ConstParameters ) );

                Tools = new ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, ITool>>
                        {
                            {
                                ViewerMode.TileManager2D,
                                new ObservableDictionary< ToolMode, ITool >
                                {
                                    { ToolMode.Null, new NullTool() },
                                    { ToolMode.AdjustSegmentation, new AdjustSegmentationTool( TileManager, this ) },
                                    { ToolMode.MergeSegmentation, new MergeSegmentationTool( TileManager, this ) },
                                    { ToolMode.DrawMergeSegmentation, new DrawMergeSegmentationTool( TileManager, this ) },
                                    { ToolMode.SplitSegmentation, new SplitSegmentationTool( TileManager, this ) }
                                }
                                }
                        };

                RenderingStrategies = new ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, IRenderingStrategy>>
                                      {
                                          {
                                              ViewerMode.TileManager2D,
                                              new ObservableDictionary< ToolMode, IRenderingStrategy >
                                              {
                                                  { ToolMode.Null, new NullRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext ) },
                                                  { ToolMode.AdjustSegmentation, new AdjustSegmentationRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) },
                                                  { ToolMode.MergeSegmentation, new MergeSegmentationRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) },
                                                  { ToolMode.DrawMergeSegmentation, new DrawMergeSegmentationRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) },
                                                  { ToolMode.SplitSegmentation, new SplitSegmentationRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) }
                                              }
                                              }
                                      };

                Viewers = new ObservableDictionary<ViewerMode, Viewer>
                          {
                              {
                                  ViewerMode.TileManager2D,
                                  new Viewer
                                  {
                                      D3D11RenderingPane = new D3D11RenderingPane( mDxgiFactory,
                                                                                   mD3D11Device,
                                                                                   mD3D11Device.ImmediateContext,
                                                                                   d3d11HwndDescriptions.Get( "TileManager2D" ) )
                                                           {
                                                               RenderingStrategy = RenderingStrategies.Internal[ ViewerMode.TileManager2D ].Internal[ ToolMode.Null ]
                                                           },
                                      UserInputHandler = Tools.Internal[ ViewerMode.TileManager2D ].Internal[ ToolMode.Null ]
                                  }
                                  }
                          };
            }
            catch ( Exception e )
            {
                String errorMessage = "Error opening main window:\n\n" + e.Message + "\n\nYou might want to try one of the following:\n - Install the \"DirectX End-User Runtime\" (from the Microsoft website).\n - Install the latest graphics drivers for your graphics card.\n - Reinstall the latest Mojo release.";
                MessageBox.Show( errorMessage, "Initialization Error", MessageBoxButton.OK, MessageBoxImage.Error );
                Console.WriteLine( errorMessage );
                Application.Current.Shutdown( 1 );
            }

        }

        public void Dispose()
        {
            if ( Viewers != null )
            {
                Viewers.Internal.Values.ToList().ForEach( viewer => viewer.Dispose() );
                Viewers.Internal.Clear();
            }

            if ( RenderingStrategies != null )
            {
                RenderingStrategies.Internal.Values.ToList().ForEach( renderingStrategies => renderingStrategies.Internal.Values.ToList().ForEach( renderingStrategy => renderingStrategy.Dispose() ) );
                RenderingStrategies.Internal.Clear();
            }

            if ( TileManager != null )
            {
                TileManager.Dispose();
                TileManager = null;
            }

            //Thrust.Terminate();
            //Cuda.Terminate();
            D3D11.Terminate( ref mDxgiFactory, ref mD3D11Device );

            Console.WriteLine( "\nMojo terminating...\n" );
        }

        public void NextImage()
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
            if ( centerDataSpace.Z < TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsZ - 1 )
            {
                if ( CurrentToolMode == ToolMode.SplitSegmentation && TileManager.JoinSplits3D )
                {
                    //
                    // Record split state for 3D splitting
                    //
                    TileManager.Internal.RecordSplitState( TileManager.SelectedSegmentId, centerDataSpace );
                }
                centerDataSpace.Z += 1f;

                TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;

                CurrentToolMoveZ();
                TileManager.UpdateZ();
                UpdateOneTile();
            }
        }

        public void PreviousImage()
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
            if ( centerDataSpace.Z > 0 )
            {
                if ( CurrentToolMode == ToolMode.SplitSegmentation && TileManager.JoinSplits3D )
                {
                    //
                    // Record split state for 3D splitting
                    //
                    TileManager.Internal.RecordSplitState( TileManager.SelectedSegmentId, centerDataSpace );
                }
                centerDataSpace.Z -= 1f;

                TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;

                CurrentToolMoveZ();
                TileManager.UpdateZ();
                UpdateOneTile();
            }
        }

        public void UpdateLocationFromText( object parameter )
        {
            var parameterString = parameter as String;
            if ( parameterString != null )
            {
                Console.WriteLine( parameterString );
                string[] coordinateStrings = parameterString.Split( ',' );
                if ( coordinateStrings.Length >= 3 )
                {
                    try
                    {
                        var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
                        centerDataSpace.X = float.Parse( coordinateStrings[ 0 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_X" );
                        centerDataSpace.Y = float.Parse( coordinateStrings[ 1 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_Y" );
                        centerDataSpace.Z = float.Parse( coordinateStrings[ 2 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_Z" );
                        TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "Couldn't parse location string: " + e.Message );
                    }
                }
            }

            CheckBounds();
            CurrentToolMoveZ();
            TileManager.UpdateXYZ();
            Update();

        }

        public void CheckBounds()
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;

            var tiledVolumeDescription = TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" );

            if ( centerDataSpace.X < 0 )
            {
                centerDataSpace.X = 0;

            }
            else if ( centerDataSpace.X > tiledVolumeDescription.NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" ) - 1 )
            {
                centerDataSpace.X = tiledVolumeDescription.NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" ) - 1;
            }

            if ( centerDataSpace.Y < 0 )
            {
                centerDataSpace.Y = 0;
            }
            else if ( centerDataSpace.Y > tiledVolumeDescription.NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" ) - 1 )
            {
                centerDataSpace.Y = tiledVolumeDescription.NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" ) - 1;
            }

            if ( centerDataSpace.Z < 0 )
            {
                centerDataSpace.Z = 0;
            }
            else if ( centerDataSpace.Z > tiledVolumeDescription.NumTilesZ * Constants.ConstParameters.GetInt( "TILE_SIZE_Z" ) - 1 )
            {
                centerDataSpace.Z = tiledVolumeDescription.NumTilesZ * Constants.ConstParameters.GetInt( "TILE_SIZE_Z" ) - 1;
            }

            TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;

        }

        public void CurrentToolMoveZ()
        {
            Tools.Get( ViewerMode.TileManager2D ).Get( CurrentToolMode ).MoveZ();
        }

        public void ZoomIn()
        {
            var extentDataSpace = TileManager.TiledDatasetView.ExtentDataSpace;
            if ( extentDataSpace.X > 1e-3 && extentDataSpace.Y > 1e-3 )
            {
                var changeBy = (float) Constants.MAGNIFICATION_STEP;

                //
                // Decrease the view extent
                //
                extentDataSpace.X /= changeBy;
                extentDataSpace.Y /= changeBy;

                TileManager.TiledDatasetView.ExtentDataSpace = extentDataSpace;

                CheckBounds();

                QuickRender();

                TileManager.UpdateXYZ();

            }
        }

        public void ZoomOut()
        {
            var extentDataSpace = TileManager.TiledDatasetView.ExtentDataSpace;
            var tiledVolumeDescription = TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" );

            if ( extentDataSpace.X < tiledVolumeDescription.NumTilesX * 10 && extentDataSpace.Y < tiledVolumeDescription.NumTilesY * 10 )
            {
                var changeBy = (float) Constants.MAGNIFICATION_STEP;

                //
                // Increase the view extent
                //
                extentDataSpace.X *= changeBy;
                extentDataSpace.Y *= changeBy;

                TileManager.TiledDatasetView.ExtentDataSpace = extentDataSpace;

                CheckBounds();

                QuickRender();

                TileManager.UpdateXYZ();

            }
        }

        public void PanToSegmentCentroid3D( uint segId )
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
            var pointTileSpace = TileManager.Internal.GetSegmentCentralTileLocation( segId );

            if ( centerDataSpace.Z != pointTileSpace.Z && CurrentToolMode == ToolMode.SplitSegmentation && TileManager.JoinSplits3D )
            {
                //
                // Record split state for 3D splitting
                //
                TileManager.Internal.RecordSplitState( TileManager.SelectedSegmentId, centerDataSpace );
            }

            centerDataSpace.X = 0.5f + pointTileSpace.X;
            centerDataSpace.Y = 0.5f + pointTileSpace.Y;
            centerDataSpace.Z = pointTileSpace.Z;

            TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;

            Tools.Get( ViewerMode.TileManager2D ).Get( CurrentToolMode ).Select();

            TileManager.UpdateXYZ();
            Update();
        }

        public void CenterAndZoomToSegment2D( uint segId )
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
            var extentDataSpace = TileManager.TiledDatasetView.ExtentDataSpace;

            var boundsTileSpace = TileManager.Internal.GetSegmentZTileBounds( segId, (int)centerDataSpace.Z );
            centerDataSpace.X = 0.5f + boundsTileSpace.X + ( boundsTileSpace.Z - boundsTileSpace.X ) / 2;
            centerDataSpace.Y = 0.5f + boundsTileSpace.Y + ( boundsTileSpace.W - boundsTileSpace.Y ) / 2;

            var targetExtentX = 1 + boundsTileSpace.Z - boundsTileSpace.X;
            var targetExtentY = 1 + boundsTileSpace.W - boundsTileSpace.Y;

            var changeBy = (float)Math.Max( 1.1, (float)Constants.MAGNIFICATION_STEP );

            while ( extentDataSpace.X > targetExtentX || extentDataSpace.Y > targetExtentY )
            {
                extentDataSpace.X /= changeBy;
                extentDataSpace.Y /= changeBy;
            }

            while ( extentDataSpace.X < targetExtentX || extentDataSpace.Y < targetExtentY )
            {
                extentDataSpace.X *= changeBy;
                extentDataSpace.Y *= changeBy;
            }

            TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
            TileManager.TiledDatasetView.ExtentDataSpace = extentDataSpace;

            QuickRender();

            TileManager.UpdateXYZ();
            TileManager.UpdateSegmentListFocus();

        }

        public void CenterAndZoomToSegmentCentroid3D( uint segId )
        {
            PanToSegmentCentroid3D( segId );
            CenterAndZoomToSegment2D( segId );
        }

        private bool mSkipUpdate = false;
        public void Update()
        {
            if ( mSkipUpdate )
            {
                mSkipUpdate = false;
            }
            else if ( !TileManager.SegmentationChangeInProgress )
            {
                TileManager.Update();
                Viewers.Internal.ToList().ForEach( viewer => viewer.Value.D3D11RenderingPane.Render() );
            }
        }

        public void UpdateOneTile()
        {
            mSkipUpdate = true;
            TileManager.UpdateOneTile();
            Viewers.Internal.ToList().ForEach( viewer => viewer.Value.D3D11RenderingPane.Render() );
        }

        public void QuickRender()
        {
            //
            // Render the tiles we have now without loading anything else
            //
            mSkipUpdate = true;
            Viewers.Internal.ToList().ForEach( viewer => viewer.Value.D3D11RenderingPane.Render() );
        }

        public void CommitChange()
        {
            if ( CurrentToolMode == ToolMode.AdjustSegmentation )
            {
                TileManager.CommmitAdjustChange();
            }
            else if ( CurrentToolMode == ToolMode.SplitSegmentation )
            {
                TileManager.CommmitSplitChange();
            }
        }

        public void CancelChange()
        {
            if ( CurrentToolMode == ToolMode.AdjustSegmentation )
            {
                TileManager.CancelAdjustChange();
            }
            else if ( CurrentToolMode == ToolMode.SplitSegmentation )
            {
                TileManager.CancelSplitChange();
            }
        }

        public void SelectSegment( uint segId )
        {
            Tools.Get( ViewerMode.TileManager2D ).Get( CurrentToolMode ).SelectSegment( segId );
        }

    }
}
