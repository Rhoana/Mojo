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
        public RelayCommand SelectNewIdCommand { get; private set; }
        public RelayCommand ToggleJoinSplits3DCommand { get; private set; }
        public RelayCommand CommitChangeCommand { get; private set; }
        public RelayCommand CancelChangeCommand { get; private set; }
        public RelayCommand IncreaseBrushSizeCommand { get; private set; }
        public RelayCommand DecreaseBrushSizeCommand { get; private set; }

        //
        // View menu commands
        //
        public RelayCommand NextImageCommand { get; private set; }
        public RelayCommand PreviousImageCommand { get; private set; }
        public RelayCommand ZoomInCommand { get; private set; }
        public RelayCommand ZoomOutCommand { get; private set; }
        public RelayCommand UpdateLocationFromTextCommand { get; private set; }
        public RelayCommand ToggleShowSegmentationCommand { get; private set; }
        public RelayCommand ToggleShowBoundaryLinesCommand { get; private set; }
        public RelayCommand IncreaseSegmentationVisibilityCommand { get; private set; }
        public RelayCommand DecreaseSegmentationVisibilityCommand { get; private set; }
        public RelayCommand Open3DViewerCommand { get; private set; }
        public RelayCommand Set3DZSpacingCommand { get; private set; }
        public RelayCommand Multi3DViewCommand { get; private set; }

        //
        // Segment list commands
        //
        public RelayCommand FirstSegmentPageCommand { get; private set; }
        public RelayCommand PreviousSegmentPageCommand { get; private set; }
        public RelayCommand NextSegmentPageCommand { get; private set; }
        public RelayCommand LastSegmentPageCommand { get; private set; }
        public RelayCommand LockSegmentLabelCommand { get; private set; }
        public RelayCommand ToggleSelectedSegmentLockCommand { get; private set; }
        public RelayCommand SetSelectedSegmentTypeCommand { get; private set; }
        public RelayCommand SetSelectedSegmentSubTypeCommand { get; private set; }

        //
        // Recent dataset details
        //
        public static String RecentRecentDataset1 { get; private set; }
        public static String RecentRecentDataset2 { get; private set; }
        public static String RecentRecentDataset3 { get; private set; }
        public static String RecentRecentDataset4 { get; private set; }
        public static String RecentRecentDataset5 { get; private set; }
        public static Visibility RecentRecentDataset1Visibility { get; private set; }
        public static Visibility RecentRecentDataset2Visibility { get; private set; }
        public static Visibility RecentRecentDataset3Visibility { get; private set; }
        public static Visibility RecentRecentDataset4Visibility { get; private set; }
        public static Visibility RecentRecentDataset5Visibility { get; private set; }

        //
        // Recent segmentation details
        //
        public static String RecentRecentSegmentation1 { get; private set; }
        public static String RecentRecentSegmentation2 { get; private set; }
        public static String RecentRecentSegmentation3 { get; private set; }
        public static String RecentRecentSegmentation4 { get; private set; }
        public static String RecentRecentSegmentation5 { get; private set; }
        public static Visibility RecentRecentSegmentation1Visibility { get; private set; }
        public static Visibility RecentRecentSegmentation2Visibility { get; private set; }
        public static Visibility RecentRecentSegmentation3Visibility { get; private set; }
        public static Visibility RecentRecentSegmentation4Visibility { get; private set; }
        public static Visibility RecentRecentSegmentation5Visibility { get; private set; }

        private const int NumRecentItems = 5;

        public class MergeModeItem
        {
            public MergeMode MergeMode { get; set; }
            public String DisplayName { get; set; }
        }

        public class MergeControlModeItem
        {
            public MergeControlMode MergeControlMode { get; set; }
            public String DisplayName { get; set; }
        }

        public class SplitModeItem
        {
            public SplitMode SplitMode { get; set; }
            public String DisplayName { get; set; }
        }

        public List<MergeModeItem> MergeModes { get; private set; }
        public List<MergeControlModeItem> MergeControlModes { get; private set; }
        public List<SplitModeItem> SplitModes { get; private set; }

        public EngineDataContext( Engine engine, TileManagerDataContext tileManagerDataContext )
        {
            Engine = engine;
            Engine.ZSpacing = Settings.Default.ViewerZSpacing;

            TileManagerDataContext = tileManagerDataContext;

            //
            // File menu commands
            //
            LoadDatasetCommand = new RelayCommand( param => LoadDataset( null ) );
            LoadSegmentationCommand = new RelayCommand( param => LoadSegmentation( null ), param => Engine.TileManager.TiledDatasetLoaded );
            SaveSegmentationCommand = new RelayCommand( param => SaveSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            SaveSegmentationAsCommand = new RelayCommand( param => SaveSegmentationAs(), param => Engine.TileManager.SegmentationLoaded );
            ExitCommand = new RelayCommand( param => Application.Current.MainWindow.Close() );

            ResetRecentDatasets();
            ResetRecentSegmentations();

            //
            // Edit menu commands
            //
            UndoChangeCommand = new RelayCommand( param => Engine.TileManager.UndoChange(), param => Engine.TileManager.SegmentationLoaded );
            RedoChangeCommand = new RelayCommand( param => Engine.TileManager.RedoChange(), param => Engine.TileManager.SegmentationLoaded );
            SelectNewIdCommand = new RelayCommand( param => Engine.TileManager.SelectNewId(), param => Engine.TileManager.SegmentationLoaded );
            CommitChangeCommand = new RelayCommand( param => Engine.CommitChange(), param => Engine.TileManager.SegmentationLoaded );
            CancelChangeCommand = new RelayCommand( param => Engine.CancelChange(), param => Engine.TileManager.SegmentationLoaded );
            IncreaseBrushSizeCommand = new RelayCommand( param => Engine.TileManager.IncreaseBrushSize(), param => Engine.TileManager.SegmentationLoaded );
            DecreaseBrushSizeCommand = new RelayCommand( param => Engine.TileManager.DecreaseBrushSize(), param => Engine.TileManager.SegmentationLoaded );

            //
            // View menu commands
            //
            NextImageCommand = new RelayCommand( param => Engine.NextImage( false ), param => Engine.TileManager.TiledDatasetLoaded );
            PreviousImageCommand = new RelayCommand( param => Engine.PreviousImage( false ), param => Engine.TileManager.TiledDatasetLoaded );
            ZoomInCommand = new RelayCommand( param => Engine.ZoomIn(), param => Engine.TileManager.TiledDatasetLoaded );
            ZoomOutCommand = new RelayCommand( param => Engine.ZoomOut(), param => Engine.TileManager.TiledDatasetLoaded );
            UpdateLocationFromTextCommand = new RelayCommand( param => Engine.UpdateLocationFromText( param ), param => Engine.TileManager.TiledDatasetLoaded );
            ToggleShowSegmentationCommand = new RelayCommand( param => Engine.TileManager.ToggleShowSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            ToggleShowBoundaryLinesCommand = new RelayCommand( param => Engine.TileManager.ToggleShowBoundaryLines(), param => Engine.TileManager.SegmentationLoaded );
            ToggleJoinSplits3DCommand = new RelayCommand( param => Engine.TileManager.ToggleJoinSplits3D(), param => Engine.TileManager.SegmentationLoaded );
            IncreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.IncreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );
            DecreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.DecreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );
            Open3DViewerCommand = new RelayCommand( param => Engine.Open3DViewer(), param => Engine.TileManager.SegmentationLoaded );
            Set3DZSpacingCommand = new RelayCommand( param => Set3DZSpacing(), param => Engine.TileManager.SegmentationLoaded );
            Multi3DViewCommand = new RelayCommand( param => Multi3DView(), param => Engine.TileManager.SegmentationLoaded );

            //
            // Segment list commands
            //
            FirstSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToFirstSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            PreviousSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToPreviousSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            NextSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToNextSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            LastSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToLastSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            ToggleSelectedSegmentLockCommand = new RelayCommand( param => TileManagerDataContext.ToggleSelectedSegmentLock(), param => Engine.TileManager.SegmentationLoaded );
            SetSelectedSegmentTypeCommand = new RelayCommand( param => TileManagerDataContext.SetSelectedSegmentTypeCommand( param ), param => Engine.TileManager.SegmentationLoaded );
            SetSelectedSegmentSubTypeCommand = new RelayCommand( param => TileManagerDataContext.SetSelectedSegmentSubTypeCommand( param ), param => Engine.TileManager.SegmentationLoaded );

            TileManagerDataContext.StateChanged += StateChangedHandler;

            MergeModes = new List<MergeModeItem>
            {
                new MergeModeItem() {MergeMode = MergeMode.GlobalReplace, DisplayName = "Global Replace"},
                new MergeModeItem() {MergeMode = MergeMode.Fill2D, DisplayName = "2D Region Fill"},
                new MergeModeItem() {MergeMode = MergeMode.Fill3D, DisplayName = "3D Region Fill (slow)"},
            };

            MergeControlModes = new List<MergeControlModeItem>
            {
                new MergeControlModeItem() {MergeControlMode = MergeControlMode.Draw, DisplayName = "Draw"},
                new MergeControlModeItem() {MergeControlMode = MergeControlMode.Click, DisplayName = "Click"},
            };

            SplitModes = new List<SplitModeItem>
            {
                new SplitModeItem() {SplitMode = SplitMode.DrawSplit, DisplayName = "Draw Split Line"},
                new SplitModeItem() {SplitMode = SplitMode.DrawRegions, DisplayName = "Draw Regions"},
                new SplitModeItem() {SplitMode = SplitMode.JoinPoints, DisplayName = "Points"}
            };

            OnPropertyChanged( "MergeModes" );
            OnPropertyChanged( "MergeControlModes" );
            OnPropertyChanged( "SplitModes" );

        }

        public class RecentMenuItem
        {
            public RelayCommand Command { get; set; }
            public string Header { get; set; }
        }

        private IList<RecentMenuItem> mRecentDatasetList = new List<RecentMenuItem>();
        public IList<RecentMenuItem> RecentDatasetList
        {
            get
            {
                return mRecentDatasetList;
            }
            set
            {
                mRecentDatasetList = value;
                OnPropertyChanged( "RecentDatasetList" );
            }
        }
        
        public void ResetRecentDatasets()
        {
            IList<RecentMenuItem> newRecentDatasetList = new List<RecentMenuItem>();
            for ( int menuIndex = 0; menuIndex < NumRecentItems; ++menuIndex )
            {
                if ( Settings.Default.RecentDatasetPaths != null && Settings.Default.RecentDatasetPaths.Count - menuIndex > 0 )
                {
                    String datasetPath = Settings.Default.RecentDatasetPaths[ Settings.Default.RecentDatasetPaths.Count - menuIndex - 1 ];
                    newRecentDatasetList.Add( new RecentMenuItem
                    {
                        Command = new RelayCommand( param => LoadDataset( datasetPath ) ),
                        Header = ( menuIndex + 1 ) + " " + datasetPath,
                    } );
                }
                else
                {
                    break;
                }
            }
            RecentDatasetList = newRecentDatasetList;
        }

        private IList<RecentMenuItem> mRecentSegmentationList = new List<RecentMenuItem>();
        public IList<RecentMenuItem> RecentSegmentationList
        {
            get
            {
                return mRecentSegmentationList;
            }
            set
            {
                mRecentSegmentationList = value;
                OnPropertyChanged( "RecentSegmentationList" );
            }
        }

        public void ResetRecentSegmentations()
        {
            IList<RecentMenuItem> newRecentSegmentationList = new List<RecentMenuItem>();
            for ( int menuIndex = 0; menuIndex < NumRecentItems; ++menuIndex )
            {
                if ( Settings.Default.RecentSegmentationPaths != null && Settings.Default.RecentSegmentationPaths.Count - menuIndex > 0 )
                {
                    String segmentationPath = Settings.Default.RecentSegmentationPaths[ Settings.Default.RecentSegmentationPaths.Count - menuIndex - 1 ];
                    newRecentSegmentationList.Add( new RecentMenuItem
                    {
                        Command = new RelayCommand( param => LoadSegmentation( segmentationPath ) ),
                        Header = ( menuIndex + 1 ) + " " + segmentationPath,
                    } );
                }
                else
                {
                    break;
                }
            }
            RecentSegmentationList = newRecentSegmentationList;
        }

        public void Dispose()
        {
            if ( TileManagerDataContext != null )
            {
                TileManagerDataContext.StateChanged -= StateChangedHandler;
                TileManagerDataContext.Dispose();
                TileManagerDataContext = null;
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

        public void Multi3DView()
        {
            //
            // Open a dialog to accept multiple ids
            //
            var textInputDialog = new Ookii.Dialogs.InputDialog
            {
                WindowTitle = "3D View Multiple Objects",
                MainInstruction = "Please input ids you would like to display in the 3D viewer, seperated by commas.\nFor example: 13, 14, 16"
            };

            var result = textInputDialog.ShowDialog();

            if ( result == System.Windows.Forms.DialogResult.OK )
            {
                bool first_round = true;

                foreach ( String idString in textInputDialog.Input.Split(',') )
                {
                    Console.WriteLine( idString );
                    try
                    {
                        uint viewId = uint.Parse( idString );
                        Console.WriteLine( "" + viewId );
                        if ( viewId > 0 )
                        {
                            if ( first_round )
                            {
                                Engine.StartExternalViewer( viewId );
                                first_round = false;
                            }
                            else
                            {
                                Engine.UpdateExternalViewerId( viewId );
                            }
                        }
                    }
                    catch ( Exception e )
                    {
                        Console.WriteLine( e.Message );
                    }
                }

                Engine.UpdateExternalViewerLocation();
            }

        }

        public void Set3DZSpacing()
        {
            //
            // Open a dialog to set z-spacing
            //
            var textInputDialog = new Ookii.Dialogs.InputDialog
            {
                WindowTitle = "3D Z-Spacing",
                MainInstruction = "Please enter the new z-spacing.",
                Input = Engine.ZSpacing.ToString()
            };

            var result = textInputDialog.ShowDialog();

            if ( result == System.Windows.Forms.DialogResult.OK )
            {

                decimal newSpacing = decimal.Parse( textInputDialog.Input );
                Settings.Default.ViewerZSpacing = newSpacing;
                Settings.Default.Save();
                Engine.ZSpacing = newSpacing;
            }
        }

        private void RememberDatasetPath( String datasetPath )
        {
            if ( Settings.Default.RecentDatasetPaths == null )
            {
                Settings.Default.RecentDatasetPaths = new System.Collections.Specialized.StringCollection();
            }
            var newDataset = true;
            while ( Settings.Default.RecentDatasetPaths.Contains( datasetPath ) )
            {
                Settings.Default.RecentDatasetPaths.Remove( datasetPath );
                newDataset = false;
            }
            Settings.Default.RecentDatasetPaths.Add( datasetPath );
            while ( Settings.Default.RecentDatasetPaths.Count > NumRecentItems )
            {
                Settings.Default.RecentDatasetPaths.RemoveAt( 0 );
            }
            Settings.Default.Save();
            ResetRecentDatasets();
            if ( newDataset )
            {
                RememberSegmentationPath( datasetPath );
            }
        }

        private void RememberSegmentationPath( String segmentationPath )
        {
            if ( Settings.Default.RecentSegmentationPaths == null )
            {
                Settings.Default.RecentSegmentationPaths = new System.Collections.Specialized.StringCollection();
            }
            while ( Settings.Default.RecentSegmentationPaths.Contains( segmentationPath ) )
            {
                Settings.Default.RecentSegmentationPaths.Remove( segmentationPath );
            }
            Settings.Default.RecentSegmentationPaths.Add( segmentationPath );
            while ( Settings.Default.RecentSegmentationPaths.Count > NumRecentItems )
            {
                Settings.Default.RecentSegmentationPaths.RemoveAt( 0 );
            }
            Settings.Default.Save();
            ResetRecentSegmentations();
        }

        private void LoadDataset( String datasetPath )
        {
            if ( Engine.TileManager.ChangesMade )
            {
                var mbresult = MessageBox.Show( "Changes were made to this segmentation. Do you want to save the changes?", "Save Changes?", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning );
                switch ( mbresult )
                {
                    case MessageBoxResult.Yes:
                        Engine.TileManager.SaveSegmentation();
                        break;
                    case MessageBoxResult.No:
                        Engine.TileManager.DiscardChanges();
                        break;
                    default:
                        return;
                }
            }

            if ( datasetPath == null || datasetPath.Length == 0 )
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
                    datasetPath = folderBrowserDialog.SelectedPath;
                }
                else
                {
                    //
                    // Cancel button clicked or dialog closed
                    //
                    return;
                }
            }

            Settings.Default.LoadDatasetPath = datasetPath;

            try
            {
                //
                // Load the dataset and show (approximate) progress
                //

                TileManagerDataContext.Progress = 10;

                Engine.TileManager.LoadTiledDataset( datasetPath );

                TileManagerDataContext.Progress = 70;

                if ( Engine.TileManager.TiledDatasetLoaded )
                {
                    //
                    // Set the initial view
                    //
                    var viewportDataSpaceX = Engine.Viewers.Internal[ ViewerMode.TileManager2D ].D3D11RenderingPane.Viewport.Width / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileX;
                    var viewportDataSpaceY = Engine.Viewers.Internal[ ViewerMode.TileManager2D ].D3D11RenderingPane.Viewport.Height / Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileY;
                    var maxExtentDataSpaceX = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
                    var maxExtentDataSpaceY = Engine.TileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

                    var zoomLevel = Math.Min( viewportDataSpaceX / maxExtentDataSpaceX, viewportDataSpaceY / maxExtentDataSpaceY );

                    Engine.TileManager.TiledDatasetView.CenterDataSpace = new Vector3( maxExtentDataSpaceX / 2f, maxExtentDataSpaceY / 2f, 0f );
                    Engine.TileManager.TiledDatasetView.ExtentDataSpace = new Vector3( viewportDataSpaceX / zoomLevel, viewportDataSpaceY / zoomLevel, 0f );

                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;

                    Engine.TileManager.UpdateXYZ();

                    TileManagerDataContext.Progress = 90;

                    //
                    // Reset the segment info list
                    //
                    TileManagerDataContext.SortSegmentListBy( "Size", true );

                    //
                    // Maintain recent file list
                    //
                    RememberDatasetPath( datasetPath );

                }
                else
                {
                    Engine.CurrentToolMode = ToolMode.Null;
                    Engine.TileManager.UnloadTiledDataset();
                }

            }
            catch ( Exception e )
            {
                Engine.CurrentToolMode = ToolMode.Null;
                Engine.TileManager.UnloadTiledDataset();
                String errorMessage = "Error loading images from:\n" + datasetPath + "\n\n" + e.Message + "\n\nPlease check the path and try again.";
                MessageBox.Show( errorMessage, "Load Error", MessageBoxButton.OK, MessageBoxImage.Error );
                Console.WriteLine( errorMessage );
            }

            TileManagerDataContext.Progress = 100;

        }

        private void LoadSegmentation( String segmentationPath )
        {
            if ( Engine.TileManager.ChangesMade )
            {
                var mbresult = MessageBox.Show( "Changes were made to this segmentation. Do you want to save the changes?", "Save Changes?", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning );
                switch ( mbresult )
                {
                    case MessageBoxResult.Yes:
                        Engine.TileManager.SaveSegmentation();
                        break;
                    case MessageBoxResult.No:
                        Engine.TileManager.DiscardChanges();
                        break;
                    default:
                        return;
                }
            }

            if ( segmentationPath == null || segmentationPath.Length == 0 )
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
                    segmentationPath = folderBrowserDialog.SelectedPath;
                }
                else
                {
                    //
                    // Cancel button clicked or dialog closed
                    //
                    return;
                }
            }

            Settings.Default.LoadSegmentationPath = segmentationPath;

            try
            {
                //
                // Load the segmentation and show (approximate) progress
                //

                TileManagerDataContext.Progress = 10;

                Engine.TileManager.LoadSegmentation( segmentationPath );

                TileManagerDataContext.Progress = 70;

                if ( Engine.TileManager.SegmentationLoaded )
                {
                    //
                    // Set the initial view
                    //
                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;
                    Engine.TileManager.SegmentationVisibilityRatio = 0.5f;

                    //
                    // Load segment info list
                    //
                    TileManagerDataContext.SortSegmentListBy( "Size", true );

                    //
                    // Maintain recent file list
                    //
                    RememberSegmentationPath( segmentationPath );

                }
                else
                {
                    Engine.TileManager.UnloadSegmentation();
                    TileManagerDataContext.UpdateSegmentInfoList();
                }
            }
            catch ( Exception e )
            {
                Engine.TileManager.UnloadSegmentation();
                TileManagerDataContext.UpdateSegmentInfoList();
                String errorMessage = "Error loading segmentation from:\n" + segmentationPath + "\n\n" + e.Message + "\n\nPlease check the path and try again.";
                MessageBox.Show( errorMessage, "Load Error", MessageBoxButton.OK, MessageBoxImage.Error );
                Console.WriteLine( errorMessage );
            }

            TileManagerDataContext.Progress = 100;

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

    }
}