using System;
using System.ComponentModel;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using Mojo.Interop;
using Mojo.Wpf.ViewModel;

namespace Mojo.Wpf.View
{
    public partial class Application
    {
        private MainWindow mMainWindow;
        private Engine mEngine;
        private DispatcherTimer mUpdateTimer;
        private DispatcherTimer mAutoSaveSegmentationTimer;

        protected override void OnStartup( StartupEventArgs e )
        {
            base.OnStartup( e );

            mMainWindow = new MainWindow();

            var windowDescriptions = new ObservableDictionary<string, D3D11HwndDescription>
                                     {
                                         {
                                             "TileManager2D",
                                             new D3D11HwndDescription
                                             {
                                                 Handle = mMainWindow.TileManager2DViewerContext.D3D11RenderingPaneHost.Handle,
                                                 Width = mMainWindow.TileManager2DViewerContext.D3D11RenderingPaneHost.Width,
                                                 Height = mMainWindow.TileManager2DViewerContext.D3D11RenderingPaneHost.Height
                                             }
                                             },
                                     };

            mEngine = new Engine( windowDescriptions );

            mUpdateTimer = new DispatcherTimer( DispatcherPriority.Input ) { Interval = TimeSpan.FromMilliseconds( Settings.Default.TargetFrameTimeMilliseconds ) };
            mUpdateTimer.Tick += UpdateTimerTickHandler;
            mUpdateTimer.Start();

            mAutoSaveSegmentationTimer = new DispatcherTimer( DispatcherPriority.Input ) { Interval = TimeSpan.FromSeconds( Settings.Default.AutoSaveSegmentationFrequencySeconds ) };
            mAutoSaveSegmentationTimer.Tick += AutoSaveSegmentationTimerTickHandler;
            mAutoSaveSegmentationTimer.Start();

            if (Settings.Default.AutoSaveSegmentation)
            {
                Console.WriteLine(
                    "\nConfigured for to Autosave the segmentation every " + Settings.Default.AutoSaveSegmentationFrequencySeconds +
                    " seconds into the path " + Settings.Default.AutoSaveSegmentationPath + "...\n");
            }

            var engineDataContext = new EngineDataContext( mEngine, new TileManagerDataContext( mEngine.TileManager ) );

            mMainWindow.DataContext = engineDataContext;

            mMainWindow.Closing += OnMainWindowClosing;

            mMainWindow.Show();
        }

        protected override void OnExit( ExitEventArgs e )
        {

            ((EngineDataContext)mMainWindow.DataContext).Dispose();
            mMainWindow.DataContext = null;

            mAutoSaveSegmentationTimer.Stop();
            mAutoSaveSegmentationTimer.Tick -= AutoSaveSegmentationTimerTickHandler;

            mUpdateTimer.Stop();
            mUpdateTimer.Tick -= UpdateTimerTickHandler;

            Settings.Default.Save();

            if ( mEngine != null )
            {
                mEngine.Dispose();
                mEngine = null;
            }

            if ( mMainWindow != null )
            {
                mMainWindow.Dispose();
                mMainWindow = null;
            }

            base.OnExit( e );

        }

        private void UpdateTimerTickHandler(object sender, EventArgs e)
        {
            mEngine.Update();
        }

        public void AutoSaveSegmentationTimerTickHandler(object sender, EventArgs eventArgs)
        {
            if ( Settings.Default.AutoSaveSegmentation && mEngine.TileManager.SegmentationLoaded )
            {
                var dateTimeString = String.Format("{0:s}", DateTime.Now).Replace(':', '-');

                Console.WriteLine("Auto-saving segmentation: " + dateTimeString );

                mEngine.TileManager.SaveSegmentationAs( Path.Combine( Settings.Default.AutoSaveSegmentationPath, dateTimeString + Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX ) );
            }
        }

        public void OnMainWindowClosing( object sender, CancelEventArgs eventArgs )
        {
            if ( mEngine.TileManager.ChangesMade )
            {
                var result = MessageBox.Show( "Changes were made to this segmentation. Do you want to save the changes?", "Save Changes?", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning );
                switch ( result )
                {
                    case MessageBoxResult.Yes:
                        mEngine.TileManager.SaveSegmentation();
                        break;
                    case MessageBoxResult.No:
                        mEngine.TileManager.DiscardChanges();
                        break;
                    default:
                        eventArgs.Cancel = true;
                        break;
                }
            }
        }
    }
}
