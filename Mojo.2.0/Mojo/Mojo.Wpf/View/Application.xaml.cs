using System;
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

            if ( Settings.Default.AutoSaveSegmentation )
            {
                mEngine.Segmenter.EnableAutoSave( Settings.Default.AutoSaveSegmentationFrequencySeconds, Settings.Default.AutoSaveSegmentationPath );
            }

            mUpdateTimer = new DispatcherTimer( DispatcherPriority.Input ) { Interval = TimeSpan.FromMilliseconds( Settings.Default.TargetFrameTimeMilliseconds ) };
            mUpdateTimer.Tick += TickHandler;
            mUpdateTimer.Start();

            mMainWindow.DataContext = new EngineDataContext( mEngine, new TileManagerDataContext( mEngine.TileManager ), new SegmenterDataContext( mEngine.Segmenter ) );
            mMainWindow.Show();
        }

        protected override void OnExit( ExitEventArgs e )
        {
            mMainWindow.DataContext = null;

            mUpdateTimer.Stop();
            mUpdateTimer.Tick -= TickHandler;

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

        private void TickHandler( object sender, EventArgs e )
        {
            mEngine.Update();
        }
    }
}
