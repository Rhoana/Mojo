using System;
using System.Linq;
using System.Windows;
using System.ComponentModel;
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

        private String mExternalViewerPath = null;
        private String mExternalViewerArgs = null;
        private int mWOffset = 0;

        public Engine( ObservableDictionary<string, D3D11HwndDescription> d3d11HwndDescriptions, String externalViewerPath, int wOffset )
        {
            Console.WriteLine( "\nMojo initializing...\n" );

            try
            {
                mExternalViewerPath = externalViewerPath;
                mWOffset = wOffset;

                String[] pathSplit = mExternalViewerPath.Split( ' ' );
                if (pathSplit.Length > 1)
                {
                    bool foundExe = false;
                    mExternalViewerPath = "";
                    mExternalViewerArgs = "";
                    for ( int i = 0; i < pathSplit.Length; ++i )
                    {
                        if ( !foundExe )
                        {
                            // Append to the executable path
                            if (mExternalViewerPath.Length > 0)
                            {
                                mExternalViewerPath += " ";
                            }
                            mExternalViewerPath += pathSplit[ i ];

                            // Check if the exe path is complete
                            if (pathSplit[i].ToLower().EndsWith(".exe"))
                            {
                                foundExe = true;
                            }
                        }
                        else
                        {
                            // Append to the arguments list
                            if ( mExternalViewerArgs.Length > 0 )
                            {
                                mExternalViewerArgs += " ";
                            }
                            mExternalViewerArgs += pathSplit[ i ];
                        }
                    }
                }

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

        public void NextImage( bool isRepeat )
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
                UpdateZ();

                if ( isRepeat )
                    UpdateFast();
                else
                    Update();
            }
        }

        public void PreviousImage( bool isRepeat )
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
                UpdateZ();

                if ( isRepeat )
                    UpdateFast();
                else
                    Update();
            }
        }

        public void UpdateZ()
        {
            TileManager.UpdateZ();
            UpdateExternalViewerLocation();
        }

        public void UpdateXYZ()
        {
            TileManager.UpdateXYZ();
            UpdateExternalViewerLocation();
        }

        public void UpdateLocationFromText( object parameter )
        {
            var parameterString = parameter as String;
            if ( parameterString != null )
            {
                Console.WriteLine( parameterString );
                string[] coordinateStrings = parameterString.Replace("(", "").Replace(")", "").Split( ',' );
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
            UpdateXYZ();
            Update();

        }

        public void CheckBounds()
        {
            var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;

            var tiledVolumeDescription = TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" );

            var xMax = (float)( tiledVolumeDescription.NumVoxelsX - 1 ) / (float)tiledVolumeDescription.NumVoxelsPerTileX;

            if ( centerDataSpace.X < 0 )
            {
                centerDataSpace.X = 0;

            }
            else if ( centerDataSpace.X > xMax )
            {
                centerDataSpace.X = xMax;
            }
            
            var yMax = (float)( tiledVolumeDescription.NumVoxelsY - 1 ) / (float)tiledVolumeDescription.NumVoxelsPerTileY;

            if ( centerDataSpace.Y < 0 )
            {
                centerDataSpace.Y = 0;
            }
            else if ( centerDataSpace.Y > yMax )
            {
                centerDataSpace.Y = yMax;
            }

            var zMax = (float)( tiledVolumeDescription.NumVoxelsZ - 1 ) / (float)tiledVolumeDescription.NumVoxelsPerTileZ;

            if ( centerDataSpace.Z < 0 )
            {
                centerDataSpace.Z = 0;
            }
            else if ( centerDataSpace.Z > zMax )
            {
                centerDataSpace.Z = zMax;
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

                UpdateXYZ();

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

                UpdateXYZ();

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

            UpdateXYZ();
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

        public void UpdateFast()
        {
            mSkipUpdate = true;
            if ( mWOffset <= 0 )
            {
                TileManager.UpdateOneTile();
            }
            else
            {
                TileManager.UpdateWOffset( mWOffset );
            }
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

        System.Diagnostics.Process mExternalViewerProcess = null;

        private decimal mZSpacing = 2.5M;
        public decimal ZSpacing
        {
            get
            {
                return mZSpacing;
            }
            set
            {
                mZSpacing = value;
            }
        }

        public void Open3DViewer()
        {
            if ( TileManager.SelectedSegmentId != 0 )
            {
                if ( TileManager.ChangesMade )
                {
                    var result = MessageBox.Show( "Changes were made to this segmentation. Do you want to save the changes before viewing?", "Save Changes?", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning );
                    switch ( result )
                    {
                        case MessageBoxResult.Yes:
                            TileManager.SaveSegmentation();
                            break;
                        case MessageBoxResult.No:
                            TileManager.DiscardChanges();
                            break;
                        default:
                            return;
                    }
                }

                StartExternalViewer( TileManager.SelectedSegmentId );
                UpdateExternalViewerLocation();

            }

        }

        public void StartExternalViewer( uint viewId )
        {
            lock ( this )
            {
                if ( mExternalViewerProcess == null )
                {
                    try
                    {
                        String viewerArguments = "";
                        if ( mExternalViewerArgs == null || mExternalViewerArgs.Length == 0 )
                        {
                            viewerArguments +=
                                TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).ImageDataDirectory.Replace( "\\ids\\tiles", "" ) + " " +
                                Math.Round( TileManager.TiledDatasetView.CenterDataSpace.X * Constants.ConstParameters.GetInt( "TILE_PIXELS_X" ) ) + " " +
                                Math.Round( TileManager.TiledDatasetView.CenterDataSpace.Y * Constants.ConstParameters.GetInt( "TILE_PIXELS_Y" ) ) + " " +
                                ( TileManager.TiledDatasetView.CenterDataSpace.Z * Constants.ConstParameters.GetInt( "TILE_PIXELS_Z" ) ) + " " +
                                TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).NumVoxelsX + " " +
                                TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).NumVoxelsY + " " +
                                viewId + ":";
                        }
                        else
                        {
                            viewerArguments += mExternalViewerArgs
                                .Replace( "$DIRECTORY", TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).ImageDataDirectory.Replace( "\\ids\\tiles", "" ) )
                                .Replace( "$LOCATION", Math.Round( TileManager.TiledDatasetView.CenterDataSpace.X * Constants.ConstParameters.GetInt( "TILE_PIXELS_X" ) ) + "," +
                                    Math.Round( TileManager.TiledDatasetView.CenterDataSpace.Y * Constants.ConstParameters.GetInt( "TILE_PIXELS_Y" ) ) + "," +
                                    ( TileManager.TiledDatasetView.CenterDataSpace.Z * Constants.ConstParameters.GetInt( "TILE_PIXELS_Z" ) ) )
                                .Replace( "$MAX_X", TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).NumVoxelsX.ToString() )
                                .Replace( "$MAX_Y", TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).NumVoxelsY.ToString() )
                                .Replace( "$Z_SPACING", mZSpacing.ToString() ) +
                                " " + viewId + ":";
                        }

                        System.Collections.Generic.IList<uint> tileIds = TileManager.GetRemappedChildren( viewId );

                        viewerArguments += String.Join( ",", tileIds );


                        Console.WriteLine( "Running Viewer:" );
                        Console.WriteLine( mExternalViewerPath );
                        Console.WriteLine( viewerArguments );

                        mExternalViewerProcess = new System.Diagnostics.Process();
                        mExternalViewerProcess.StartInfo.UseShellExecute = false;
                        mExternalViewerProcess.StartInfo.RedirectStandardOutput = true;
                        mExternalViewerProcess.StartInfo.RedirectStandardInput = true;
                        mExternalViewerProcess.StartInfo.WorkingDirectory = System.IO.Path.GetDirectoryName( mExternalViewerPath );
                        mExternalViewerProcess.StartInfo.FileName = mExternalViewerPath;
                        mExternalViewerProcess.StartInfo.Arguments = viewerArguments;
                        mExternalViewerProcess.Start();

                        //
                        // Start a backgroundWorker to monitor navigation events
                        //

                        DoWorkEventHandler externalViewerNavigationDelegate =
                            delegate( object s, DoWorkEventArgs args )
                            {
                                string line = mExternalViewerProcess.StandardOutput.ReadLine();
                                while ( line != null )
                                {
                                    Console.WriteLine( "Got external viewer line:" );
                                    Console.WriteLine( line );
                                    if ( line.StartsWith( "location " ) )
                                    {
                                        line = line.Replace( "location ", "" );
                                        Application.Current.Dispatcher.Invoke( new System.Action( () => UpdateLocationFromExternalViewer( line, TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "IdMap" ).NumTilesW - 1 ) ) );
                                    }
                                    line = mExternalViewerProcess.StandardOutput.ReadLine();
                                }
                                lock ( this )
                                {
                                    mExternalViewerProcess.WaitForExit();
                                    mExternalViewerProcess = null;
                                }
                            };

                        BackgroundWorker worker = new BackgroundWorker();
                        worker.DoWork += externalViewerNavigationDelegate;
                        worker.RunWorkerAsync();
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "WARNING: Could not start external viewer." );
                        Console.WriteLine( e.Message );
                        Console.WriteLine( e.StackTrace );
                    }
                }
                else
                {
                    UpdateExternalViewerId( viewId );
                }
            }
        }

        public void UpdateExternalViewerSelectedId()
        {
            UpdateExternalViewerId( TileManager.SelectedSegmentId );
        }

        public void UpdateExternalViewerId( uint viewId )
        {
            lock ( this )
            {
                if ( mExternalViewerProcess != null )
                {
                    try
                    {
                        //
                        // Render given id
                        //
                        string viewerArguments = "ids " +
                            viewId + ":";

                        System.Collections.Generic.IList<uint> tileIds = TileManager.GetRemappedChildren( viewId );

                        viewerArguments += String.Join( ",", tileIds );

                        Console.WriteLine( "Updating Viewer:" );
                        Console.WriteLine( viewerArguments );

                        mExternalViewerProcess.StandardInput.WriteLine( viewerArguments );
                        mExternalViewerProcess.StandardInput.Flush();
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "WARNING: Could not update id in external viewer." );
                        Console.WriteLine( e.Message );
                        Console.WriteLine( e.StackTrace );
                    }
                }
            }
        }

        public void UpdateExternalViewerLocation()
        {
            UpdateExternalViewerLocation( TileManager.TiledDatasetView.CenterDataSpace );
        }

        public void UpdateExternalViewerLocation( SlimDX.Vector3 p )
        {
            lock ( this )
            {
                if ( mExternalViewerProcess != null )
                {
                    try
                    {
                        //
                        // Update marker location
                        //
                        string viewerArguments = "marker " +
                            Math.Round( TileManager.TiledDatasetView.CenterDataSpace.X * Constants.ConstParameters.GetInt( "TILE_PIXELS_X" ) ) + " " +
                            Math.Round( TileManager.TiledDatasetView.CenterDataSpace.Y * Constants.ConstParameters.GetInt( "TILE_PIXELS_Y" ) ) + " " +
                            ( TileManager.TiledDatasetView.CenterDataSpace.Z * Constants.ConstParameters.GetInt( "TILE_PIXELS_Z" ) );

                        Console.WriteLine( "Updating Viewer:" );
                        Console.WriteLine( viewerArguments );

                        mExternalViewerProcess.StandardInput.WriteLine( viewerArguments );
                        mExternalViewerProcess.StandardInput.Flush();
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "WARNING: Could not navigate in external viewer." );
                        Console.WriteLine( e.Message );
                        Console.WriteLine( e.StackTrace );
                    }
                }
            }
        }

        public void UpdateLocationFromExternalViewer( String location, int w )
        {
            if ( location != null )
            {
                Console.WriteLine( location );
                string[] locationSplit = location.Split( ' ' );
                if ( locationSplit.Length >= 3 )
                {
                    try
                    {
                        var centerDataSpace = TileManager.TiledDatasetView.CenterDataSpace;
                        centerDataSpace.X = float.Parse( locationSplit[ 0 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_X" ) * (float)Math.Pow( 2, w );
                        centerDataSpace.Y = float.Parse( locationSplit[ 1 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_Y" ) * (float)Math.Pow( 2, w );
                        centerDataSpace.Z = float.Parse( locationSplit[ 2 ] ) / Constants.ConstParameters.GetInt( "TILE_PIXELS_Z" );
                        TileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "Couldn't parse location string: " + e.Message );
                    }
                }
                if ( locationSplit.Length >= 4 )
                {
                    try
                    {
                        TileManager.SelectedSegmentId = uint.Parse( locationSplit[3] );
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( "Couldn't selection string: " + e.Message );
                    }
                }
            }

            CheckBounds();
            CurrentToolMoveZ();
            TileManager.UpdateXYZ();
            Update();
        }

    }
}
