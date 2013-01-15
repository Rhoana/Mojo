using System;
using System.IO;
using System.Collections.Generic;
using Mojo.Interop;
using SlimDX;

namespace Mojo.Wpf.ViewModel
{
    public class EngineDataContext : NotifyPropertyChanged, IDisposable
    {
        public Engine Engine { get; private set; }

        public TileManagerDataContext TileManagerDataContext { get; private set; }
        public SegmenterDataContext SegmenterDataContext { get; private set; }

        public RelayCommand LoadDatasetCommand { get; private set; }
        public RelayCommand LoadSegmentationCommand { get; private set; }

        public class MergeModeItem
        {
            public MergeMode MergeMode { get; set; }
            public String DisplayName { get; set; }
        }

        public class SplitModeItem
        {
            public SplitMode SplitMode { get; set; }
            public String DisplayName { get; set; }
        }

        public List<MergeModeItem> MergeModes { get; private set; }
        public List<SplitModeItem> SplitModes { get; private set; }

        public EngineDataContext( Engine engine, TileManagerDataContext tileManagerDataContext, SegmenterDataContext segmenterDataContext )
        {
            Engine = engine;

            TileManagerDataContext = tileManagerDataContext;
            SegmenterDataContext = segmenterDataContext;

            LoadDatasetCommand = new RelayCommand( param => LoadDataset() );
            LoadSegmentationCommand = new RelayCommand( param => LoadSegmentation(), param => Engine.TileManager.TiledDatasetLoaded );

            TileManagerDataContext.StateChanged += StateChangedHandler;
            SegmenterDataContext.StateChanged += StateChangedHandler;

            MergeModes = new List<MergeModeItem>
            {
              new MergeModeItem() { MergeMode = MergeMode.Fill2D, DisplayName = "2D Region Fill" },
              new MergeModeItem() { MergeMode = MergeMode.Fill3D, DisplayName = "3D Region Fill" },
              new MergeModeItem() { MergeMode = MergeMode.GlobalReplace, DisplayName = "Global Replace" }
            };

            SplitModes = new List<SplitModeItem>
            {
                new SplitModeItem() { SplitMode = SplitMode.JoinPoints, DisplayName = "Points" },
                new SplitModeItem() { SplitMode = SplitMode.DrawSplit, DisplayName = "Draw Split Line" },
                new SplitModeItem() { SplitMode = SplitMode.DrawRegions, DisplayName = "Draw Regions" }
            };

            OnPropertyChanged( "MergeModes" );
            OnPropertyChanged( "SplitModes" );
        }

        public void Dispose()
        {
            if ( SegmenterDataContext != null )
            {
                SegmenterDataContext.StateChanged -= StateChangedHandler;
                SegmenterDataContext.Dispose();
                SegmenterDataContext = null;
            }

            if ( TileManagerDataContext != null )
            {
                TileManagerDataContext.StateChanged -= StateChangedHandler;
                TileManagerDataContext.Dispose();
                TileManagerDataContext = null;
            }

            if ( Engine != null )
            {
                Engine.Dispose();
                Engine = null;
            }
        }

        public void Refresh()
        {
            TileManagerDataContext.Refresh();
            SegmenterDataContext.Refresh();

            LoadSegmentationCommand.RaiseCanExecuteChanged();

            OnPropertyChanged( "TileManagerDataContext" );
            OnPropertyChanged( "SegmenterDataContext" );
        }

        private void StateChangedHandler( object sender, EventArgs e )
        {
            Refresh();
        }

        private void LoadDataset()
        {
            var initialPath = Settings.Default.LoadDatasetPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var folderBrowserDialog = new Ookii.Dialogs.Wpf.VistaFolderBrowserDialog
            {
                Description = "Select Mojo Dataset Folder (the folder you select should be called \"" + Constants.DATASET_ROOT_DIRECTORY_NAME + "\")",
                UseDescriptionForTitle = true,
                ShowNewFolderButton = false,
                SelectedPath = initialPath
            };

            var result = folderBrowserDialog.ShowDialog();

            if ( result != null && result == true )
            {
                Settings.Default.LoadDatasetPath = folderBrowserDialog.SelectedPath;
                Settings.Default.Save();

                Engine.TileManager.LoadTiledDataset( folderBrowserDialog.SelectedPath );

                if ( Engine.TileManager.TiledDatasetLoaded )
                {
                    //Set the initial view
                    var viewportDataSpaceX = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Width / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileX;
                    var viewportDataSpaceY = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Height / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileY;
                    var maxExtentDataSpaceX = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
                    var maxExtentDataSpaceY = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

                    var zoomLevel = Math.Min( viewportDataSpaceX / maxExtentDataSpaceX, viewportDataSpaceY / maxExtentDataSpaceY );

                    Engine.TileManager.TiledDatasetView.CenterDataSpace = new Vector3( maxExtentDataSpaceX / 2f, maxExtentDataSpaceY / 2f, 0f );
                    Engine.TileManager.TiledDatasetView.ExtentDataSpace = new Vector3( viewportDataSpaceX / zoomLevel, viewportDataSpaceY / zoomLevel, 0f );

                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;
                    //Engine.CurrentToolMode = ToolMode.MergeSegmentation;
                    //SegmenterDataContext.MergeSegmentationToolRadioButtonIsChecked = true;

                }
            }
        }

        private void LoadSegmentation()
        {
            var initialPath = Settings.Default.LoadSegmentationPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var folderBrowserDialog = new Ookii.Dialogs.Wpf.VistaFolderBrowserDialog
            {
                Description = "Select Mojo Segmentation Folder (the folder you select should be called \"" + Constants.DATASET_ROOT_DIRECTORY_NAME + "\")",
                UseDescriptionForTitle = true,
                ShowNewFolderButton = false,
                SelectedPath = initialPath
            };

            var result = folderBrowserDialog.ShowDialog();

            if ( result != null && result == true )
            {
                Settings.Default.LoadSegmentationPath = folderBrowserDialog.SelectedPath;
                Settings.Default.Save();

                Engine.TileManager.LoadSegmentation( folderBrowserDialog.SelectedPath );

                if ( Engine.TileManager.SegmentationLoaded )
                {
                    //Set the initial view
                    //var viewportDataSpaceX = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Width / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileX;
                    //var viewportDataSpaceY = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Height / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileY;
                    //var maxExtentDataSpaceX = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
                    //var maxExtentDataSpaceY = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

                    //var zoomLevel = Math.Min( viewportDataSpaceX / maxExtentDataSpaceX, viewportDataSpaceY / maxExtentDataSpaceY );

                    //Engine.TileManager.TiledDatasetView.CenterDataSpace = new Vector3( maxExtentDataSpaceX / 2f, maxExtentDataSpaceY / 2f, 0f );
                    //Engine.TileManager.TiledDatasetView.ExtentDataSpace = new Vector3( viewportDataSpaceX / zoomLevel, viewportDataSpaceY / zoomLevel, 0f );

                    //
                    // Enable controls and set detault values
                    //
                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;
                    //Engine.CurrentToolMode = ToolMode.MergeSegmentation;
                    SegmenterDataContext.MergeSegmentationToolRadioButtonIsChecked = true;
                    Engine.TileManager.SegmentationVisibilityRatio = 0.5f;

                }
            }
        }
    }
}