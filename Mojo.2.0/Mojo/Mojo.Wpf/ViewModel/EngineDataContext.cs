using System;
using System.IO;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Threading;
using Mojo.Interop;
using SlimDX;

namespace Mojo.Wpf.ViewModel
{
    public class EngineDataContext : NotifyPropertyChanged, IDisposable
    {
        public Engine Engine { get; private set; }

        public TileManagerDataContext TileManagerDataContext { get; private set; }

        //
        // File menu commands
        //
        public RelayCommand LoadDatasetCommand { get; private set; }
        public RelayCommand LoadSegmentationCommand { get; private set; }
        public RelayCommand SaveSegmentationCommand { get; private set; }
        public RelayCommand SaveSegmentationAsCommand { get; private set; }
        public RelayCommand ExitCommand { get; private set; }

        //
        // Edit menu commands
        //
        public RelayCommand UndoChangeCommand { get; private set; }
        public RelayCommand RedoChangeCommand { get; private set; }
        public RelayCommand CommitChangeCommand { get; private set; }
        public RelayCommand CancelChangeCommand { get; private set; }
        public RelayCommand IncreaseBrushSizeCommand { get; private set; }
        public RelayCommand DecreaseBrushSizeCommand { get; private set; }

        //
        // View menu commands
        //
        public RelayCommand NextImageCommand { get; private set; }
        public RelayCommand PreviousImageCommand { get; private set; }
        public RelayCommand ToggleShowSegmentationCommand { get; private set; }
        public RelayCommand ToggleShowBoundaryLinesCommand { get; private set; }
        public RelayCommand IncreaseSegmentationVisibilityCommand { get; private set; }
        public RelayCommand DecreaseSegmentationVisibilityCommand { get; private set; }

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

        public EngineDataContext( Engine engine, TileManagerDataContext tileManagerDataContext )
        {
            Engine = engine;

            TileManagerDataContext = tileManagerDataContext;

            //
            // File menu commands
            //
            LoadDatasetCommand = new RelayCommand( param => LoadDataset() );
            LoadSegmentationCommand = new RelayCommand( param => LoadSegmentation(), param => Engine.TileManager.TiledDatasetLoaded );
            SaveSegmentationCommand = new RelayCommand( param => SaveSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            SaveSegmentationAsCommand = new RelayCommand( param => SaveSegmentationAs(), param => Engine.TileManager.SegmentationLoaded );
            ExitCommand = new RelayCommand( param => Application.Current.MainWindow.Close() );

            //
            // Edit menu commands
            //
            UndoChangeCommand = new RelayCommand( param => Engine.TileManager.UndoChange(), param => Engine.TileManager.SegmentationLoaded );
            RedoChangeCommand = new RelayCommand( param => Engine.TileManager.RedoChange(), param => Engine.TileManager.SegmentationLoaded );
            CommitChangeCommand = new RelayCommand( param => Engine.TileManager.CommmitChange(), param => Engine.TileManager.SegmentationLoaded );
            CancelChangeCommand = new RelayCommand( param => Engine.TileManager.CancelChange(), param => Engine.TileManager.SegmentationLoaded );
            IncreaseBrushSizeCommand = new RelayCommand( param => Engine.TileManager.IncreaseBrushSize(), param => Engine.TileManager.SegmentationLoaded );
            DecreaseBrushSizeCommand = new RelayCommand( param => Engine.TileManager.DecreaseBrushSize(), param => Engine.TileManager.SegmentationLoaded );

            //
            // View menu commands
            //
            NextImageCommand = new RelayCommand( param => Engine.NextImage(), param => Engine.TileManager.TiledDatasetLoaded );
            PreviousImageCommand = new RelayCommand( param => Engine.PreviousImage(), param => Engine.TileManager.TiledDatasetLoaded );
            ToggleShowSegmentationCommand = new RelayCommand( param => Engine.TileManager.ToggleShowSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            ToggleShowBoundaryLinesCommand = new RelayCommand( param => Engine.TileManager.ToggleShowBoundaryLines(), param => Engine.TileManager.SegmentationLoaded );
            IncreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.IncreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );
            DecreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.DecreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );

            TileManagerDataContext.StateChanged += StateChangedHandler;

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

            mAutoSaveTimer.Tick += AutoSave;

        }

        public void Dispose()
        {
            mAutoSaveTimer.Tick -= AutoSave;

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

            LoadSegmentationCommand.RaiseCanExecuteChanged();

            OnPropertyChanged( "TileManagerDataContext" );
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
                    //
                    // Set the initial view
                    //
                    var viewportDataSpaceX = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Width / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileX;
                    var viewportDataSpaceY = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Height / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileY;
                    var maxExtentDataSpaceX = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
                    var maxExtentDataSpaceY = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

                    var zoomLevel = Math.Min( viewportDataSpaceX / maxExtentDataSpaceX, viewportDataSpaceY / maxExtentDataSpaceY );

                    Engine.TileManager.TiledDatasetView.CenterDataSpace = new Vector3( maxExtentDataSpaceX / 2f, maxExtentDataSpaceY / 2f, 0f );
                    Engine.TileManager.TiledDatasetView.ExtentDataSpace = new Vector3( viewportDataSpaceX / zoomLevel, viewportDataSpaceY / zoomLevel, 0f );

                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;

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
                    //
                    // Set the initial view
                    //
                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;
                    Engine.TileManager.SegmentationVisibilityRatio = 0.5f;
                }
            }
        }

        private void SaveSegmentation()
        {
            Engine.TileManager.SaveSegmentation();
        }

        private void SaveSegmentationAs()
        {
            var initialPath = Settings.Default.LoadSegmentationPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var folderBrowserDialog = new Ookii.Dialogs.Wpf.VistaFolderBrowserDialog
            {
                Description = "Select New Save Folder (the folder you select should be called \"" + Constants.DATASET_ROOT_DIRECTORY_NAME + "\")",
                UseDescriptionForTitle = true,
                ShowNewFolderButton = false,
                SelectedPath = initialPath
            };

            var result = folderBrowserDialog.ShowDialog();

            if ( result != null && result == true )
            {
                Settings.Default.LoadSegmentationPath = folderBrowserDialog.SelectedPath;
                Settings.Default.Save();

                Engine.TileManager.SaveSegmentationAs( folderBrowserDialog.SelectedPath );
            }
        }

        private string mAutoSavePath;
        private readonly DispatcherTimer mAutoSaveTimer = new DispatcherTimer( DispatcherPriority.Input );

        public void EnableAutoSave( int autoSaveSegmentationFrequencySeconds, string autoSaveSegmentationPath )
        {
            mAutoSavePath = autoSaveSegmentationPath;
            mAutoSaveTimer.Interval = TimeSpan.FromSeconds( autoSaveSegmentationFrequencySeconds );

            mAutoSaveTimer.Start();

            Console.WriteLine( "Auto-saving turned on. Auto-saving every {0} seconds.", autoSaveSegmentationFrequencySeconds );
        }

        public void DisableAutoSave()
        {
            mAutoSaveTimer.Stop();

            Console.WriteLine( "Auto-saving turned off." );
        }

        public void AutoSave( object sender, EventArgs eventArgs )
        {
            if ( Engine.TileManager.ChangesMade )
            {
                Engine.TileManager.AutoSaveSegmentation();
                Engine.TileManager.AutoChangesMade = false;
            }
        }

    }
}