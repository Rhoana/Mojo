using System;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Forms;
using Mojo.Interop;
using SlimDX;
using Application = System.Windows.Application;
using MessageBox = System.Windows.MessageBox;
using Size = System.Windows.Size;

namespace Mojo.Wpf.ViewModel
{
    public class EngineDataContext : NotifyPropertyChanged, IDisposable
    {
        public Engine Engine { get; private set; }

        public TileManagerDataContext TileManagerDataContext { get; private set; }

        //
        // File menu commands
        //
        public RelayCommand LoadSourceImagesCommand { get; private set; }
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

        //
        // Segment list commands
        //
        public RelayCommand FirstSegmentPageCommand { get; private set; }
        public RelayCommand PreviousSegmentPageCommand { get; private set; }
        public RelayCommand NextSegmentPageCommand { get; private set; }
        public RelayCommand LastSegmentPageCommand { get; private set; }
        public RelayCommand LockSegmentLabelCommand { get; private set; }


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

        public double MainWindowWidth
        {
            get
            {
                return mMainWindowWidth;
            }
            set
            {
                mMainWindowWidth = value;
                SetMainWindowTitle();
            }
        }

        private string mMainWindowTitle;
        public string MainWindowTitle
        {
            get
            {
                return mMainWindowTitle;
            }
            set
            {
                mMainWindowTitle = value;
                OnPropertyChanged( "MainWindowTitle" );
            }
        }

        private string mMojoImgFile;
        private string mMojoSegFile;
        private double mMainWindowWidth;

        public EngineDataContext( Engine engine, TileManagerDataContext tileManagerDataContext )
        {
            Engine = engine;

            TileManagerDataContext = tileManagerDataContext;

            //
            // File menu commands
            //
            LoadSourceImagesCommand = new RelayCommand( param => LoadSourceImages() );
            LoadSegmentationCommand = new RelayCommand( param => LoadSegmentation(), param => Engine.TileManager.SourceImagesLoaded );
            SaveSegmentationCommand = new RelayCommand( param => SaveSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            SaveSegmentationAsCommand = new RelayCommand( param => SaveSegmentationAs(), param => Engine.TileManager.SegmentationLoaded );
            ExitCommand = new RelayCommand( param => Application.Current.MainWindow.Close() );

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
            NextImageCommand = new RelayCommand( param => Engine.NextImage(), param => Engine.TileManager.SourceImagesLoaded );
            PreviousImageCommand = new RelayCommand( param => Engine.PreviousImage(), param => Engine.TileManager.SourceImagesLoaded );
            ZoomInCommand = new RelayCommand( param => Engine.ZoomIn(), param => Engine.TileManager.SourceImagesLoaded );
            ZoomOutCommand = new RelayCommand( param => Engine.ZoomOut(), param => Engine.TileManager.SourceImagesLoaded );
            UpdateLocationFromTextCommand = new RelayCommand( param => Engine.UpdateLocationFromText(param), param => Engine.TileManager.SourceImagesLoaded );
            ToggleShowSegmentationCommand = new RelayCommand( param => Engine.TileManager.ToggleShowSegmentation(), param => Engine.TileManager.SegmentationLoaded );
            ToggleShowBoundaryLinesCommand = new RelayCommand( param => Engine.TileManager.ToggleShowBoundaryLines(), param => Engine.TileManager.SegmentationLoaded );
            ToggleJoinSplits3DCommand = new RelayCommand( param => Engine.TileManager.ToggleJoinSplits3D(), param => Engine.TileManager.SegmentationLoaded );
            IncreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.IncreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );
            DecreaseSegmentationVisibilityCommand = new RelayCommand( param => Engine.TileManager.DecreaseSegmentationVisibility(), param => Engine.TileManager.SegmentationLoaded );

            //
            // Segment list commands
            //
            FirstSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToFirstSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            PreviousSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToPreviousSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            NextSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToNextSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );
            LastSegmentPageCommand = new RelayCommand( param => TileManagerDataContext.MoveToLastSegmentInfoPage(), param => Engine.TileManager.SegmentationLoaded );

            TileManagerDataContext.StateChanged += StateChangedHandler;

            MergeModes = new List<MergeModeItem>
            {
                new MergeModeItem {MergeMode = MergeMode.GlobalReplace, DisplayName = "Global Replace"},
                new MergeModeItem {MergeMode = MergeMode.Fill2D, DisplayName = "2D Region Fill"},
                new MergeModeItem {MergeMode = MergeMode.Fill3D, DisplayName = "3D Region Fill (slow)"},
            };

            MergeControlModes = new List<MergeControlModeItem>
            {
                new MergeControlModeItem {MergeControlMode = MergeControlMode.Draw, DisplayName = "Draw"},
                new MergeControlModeItem {MergeControlMode = MergeControlMode.Click, DisplayName = "Click"},
            };

            SplitModes = new List<SplitModeItem>
            {
                new SplitModeItem {SplitMode = SplitMode.DrawSplit, DisplayName = "Draw Split Line"},
                new SplitModeItem {SplitMode = SplitMode.DrawRegions, DisplayName = "Draw Regions"},
                new SplitModeItem {SplitMode = SplitMode.JoinPoints, DisplayName = "Points"}
            };

            SetMainWindowTitle();

            OnPropertyChanged( "MergeModes" );
            OnPropertyChanged( "MergeControlModes" );
            OnPropertyChanged( "SplitModes" );

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

        private void LoadSourceImages()
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

            var initialPath = Settings.Default.LoadSourceImagesPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var openFileDialog = new OpenFileDialog
            {
                CheckFileExists = true,
                CheckPathExists = true,
                InitialDirectory = initialPath,
                Multiselect = false,
                Filter = "Mojo Image Dataset (*." + Constants.SOURCE_IMAGES_FILE_NAME_EXTENSION + ")|*." + Constants.SOURCE_IMAGES_FILE_NAME_EXTENSION,
                FilterIndex = 1,
                RestoreDirectory = true,
                Title = "Select Mojo Image Dataset (*." + Constants.SOURCE_IMAGES_FILE_NAME_EXTENSION + ")"
            };

            var result = openFileDialog.ShowDialog();

            if ( result == DialogResult.OK )
            {
                Settings.Default.LoadSourceImagesPath = Path.GetDirectoryName( openFileDialog.FileName );
                Settings.Default.Save();

                //
                // Load the dataset and show (approximate) progress
                //
                try
                {
                    TileManagerDataContext.Progress = 10;

                    var sourceImagesPath = Path.Combine( Path.GetDirectoryName( openFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( openFileDialog.FileName ) + Constants.SOURCE_IMAGES_ROOT_DIRECTORY_NAME_SUFFIX );
                    Engine.TileManager.LoadSourceImages( sourceImagesPath );

                    TileManagerDataContext.Progress = 70;

                    Release.Assert( Engine.TileManager.SourceImagesLoaded );

                    //
                    // Set the initial view
                    //
                    var viewportDataSpaceX = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Width / Engine.TileManager.SourceImagesTiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileX;
                    var viewportDataSpaceY = Engine.Viewers.Internal[ViewerMode.TileManager2D].D3D11RenderingPane.Viewport.Height / Engine.TileManager.SourceImagesTiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumVoxelsPerTileY;
                    var maxExtentDataSpaceX = Engine.TileManager.SourceImagesTiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
                    var maxExtentDataSpaceY = Engine.TileManager.SourceImagesTiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

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

                    TileManagerDataContext.Progress = 100;

                    mMojoImgFile = openFileDialog.FileName;
                    SetMainWindowTitle();
                }
                catch ( Exception e )
                {
                    var sourceImagesPath = Path.Combine( Path.GetDirectoryName( openFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( openFileDialog.FileName ) + Constants.SOURCE_IMAGES_ROOT_DIRECTORY_NAME_SUFFIX );
                    var errorMessage = "Error loading images from:\n" + sourceImagesPath + "\n\n" + e.Message + "\n\nPlease check the path and try again.";
                    MessageBox.Show( errorMessage, "Load Error", MessageBoxButton.OK, MessageBoxImage.Error );
                    Console.WriteLine( errorMessage );
                }
            }
        }

        private void LoadSegmentation()
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

            var initialPath = Settings.Default.LoadSegmentationPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var openFileDialog = new OpenFileDialog
            {
                CheckFileExists = true,
                CheckPathExists = true,
                InitialDirectory = initialPath,
                Multiselect = false,
                Filter = "Mojo Segmentation Dataset (*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION + ")|*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION,
                FilterIndex = 1,
                RestoreDirectory = true,
                Title = "Select Mojo Segmentation Dataset (*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION + ")"
            };

            var result = openFileDialog.ShowDialog();

            if ( result == DialogResult.OK )
            {
                Settings.Default.LoadSegmentationPath = Path.GetDirectoryName( openFileDialog.FileName );
                Settings.Default.Save();

                try
                {
                    //
                    // Load the segmentation and show (approximate) progress
                    //

                    TileManagerDataContext.Progress = 10;

                    var segmentationPath = Path.Combine( Path.GetDirectoryName( openFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( openFileDialog.FileName ) + Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX );
                    Engine.TileManager.LoadSegmentation( segmentationPath );

                    Release.Assert( Engine.TileManager.SegmentationLoaded );

                    TileManagerDataContext.Progress = 70;

                    //
                    // Set the initial view
                    //
                    Engine.CurrentToolMode = ToolMode.SplitSegmentation;
                    Engine.TileManager.SegmentationVisibilityRatio = 0.5f;

                    //
                    // Load segment info list
                    //
                    TileManagerDataContext.SortSegmentListBy( "Size", true );

                    mMojoSegFile = openFileDialog.FileName;
                    SetMainWindowTitle();

                    TileManagerDataContext.Progress = 100;
                }
                catch ( Exception e )
                {
                    var segmentationPath = Path.Combine( Path.GetDirectoryName( openFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( openFileDialog.FileName ) + Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX );
                    var errorMessage = "Error loading segmentation from:\n" + segmentationPath + "\n\n" + e.Message + "\n\nPlease check the path and try again.";
                    MessageBox.Show( errorMessage, "Load Error", MessageBoxButton.OK, MessageBoxImage.Error );
                    Console.WriteLine( errorMessage );
                }
            }
        }

        private void SaveSegmentation()
        {
            Engine.TileManager.SaveSegmentation();
        }

        private void SaveSegmentationAs()
        {
            var initialPath = Settings.Default.SaveSegmentationAsPath;
            if ( string.IsNullOrEmpty( initialPath ) || !Directory.Exists( initialPath ) )
            {
                initialPath = Environment.GetFolderPath( Environment.SpecialFolder.MyDocuments );
            }

            var saveFileDialog = new SaveFileDialog
            {
                DefaultExt = Constants.SEGMENTATION_FILE_NAME_EXTENSION,
                InitialDirectory = initialPath,
                Filter = "Mojo Segmentation Dataset (*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION + ")|*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION,
                FilterIndex = 1,
                RestoreDirectory = true,
                Title = "Save As Mojo Segmentation Dataset (*." + Constants.SEGMENTATION_FILE_NAME_EXTENSION + ")"
            };

            var result = saveFileDialog.ShowDialog();

            if ( result == DialogResult.OK )
            {
                Settings.Default.SaveSegmentationAsPath = Path.GetDirectoryName( saveFileDialog.FileName );
                Settings.Default.Save();

                try
                {
                    TileManagerDataContext.Progress = 10;

                    var segmentationPath = Path.Combine( Path.GetDirectoryName( saveFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( saveFileDialog.FileName ) + Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX );
                    Engine.TileManager.SaveSegmentationAs( segmentationPath );

                    TileManagerDataContext.Progress = 100;
                }
                catch ( Exception e )
                {
                    var segmentationPath = Path.Combine( Path.GetDirectoryName( saveFileDialog.FileName ),
                                                         Path.GetFileNameWithoutExtension( saveFileDialog.FileName ) + Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX );
                    var errorMessage = "Error saving segmentation to:\n" + segmentationPath + "\n\n" + e.Message + "\n\nPlease check the path and try again.";
                    MessageBox.Show( errorMessage, "Load Error", MessageBoxButton.OK, MessageBoxImage.Error );
                    Console.WriteLine( errorMessage );
                }
            }
        }

        private void SetMainWindowTitle()
        {
            if ( Engine.TileManager.SourceImagesLoaded && Engine.TileManager.SegmentationLoaded )
            {
                MainWindowTitle = Constants.MAIN_WINDOW_BASE_TITLE + " [Image Dataset: " + MainWindowTitlePathShortener( mMojoImgFile, 0.5, 400 ) + "] [Segmentation Dataset: " + MainWindowTitlePathShortener( mMojoSegFile, 0.5, 400 ) + "]";
            }
            else if ( Engine.TileManager.SourceImagesLoaded )
            {
                MainWindowTitle = Constants.MAIN_WINDOW_BASE_TITLE + " [Image Dataset: " + MainWindowTitlePathShortener( mMojoImgFile, 1.0, 200 ) + "]";                
            }
            else
            {
                MainWindowTitle = Constants.MAIN_WINDOW_BASE_TITLE;
            }
        }

        private string MainWindowTitlePathShortener( string path, double width, double pixelOffset )
        {
            string compactedString = string.Copy( path );
            var maxSize = new System.Drawing.Size( (int)( ( MainWindowWidth - pixelOffset ) * width ), 0 );
            TextRenderer.MeasureText( compactedString, new Font( "Microsoft Sans Serif", 10 ), maxSize, TextFormatFlags.PathEllipsis | TextFormatFlags.ModifyString );

            if ( compactedString.IndexOf( "\0", System.StringComparison.Ordinal ) != -1 )
            {
                return compactedString.Substring( 0, compactedString.IndexOf( "\0", System.StringComparison.Ordinal ) );
            }
            else
            {
                return compactedString;
            }
        }
    }
}