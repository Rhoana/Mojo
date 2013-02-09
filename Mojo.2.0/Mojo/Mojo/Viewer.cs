using System;
using System.Windows;
using System.Windows.Threading;

namespace Mojo
{
    public class Viewer : IDisposable
    {
        public Viewer()
        {
            mResizeTimer.Tick += ResizeRenderingPane;
        }

        public D3D11RenderingPane D3D11RenderingPane { get; set; }

        private IUserInputHandler mUserInputHandler;
        public IUserInputHandler UserInputHandler
        {
            get
            {
                return mUserInputHandler;
            }
            set
            {
                mUserInputHandler = value;

                if ( mUserInputHandler != null )
                {
                    mUserInputHandler.SetSize( (int)D3D11RenderingPane.Viewport.Width, (int)D3D11RenderingPane.Viewport.Height, (int)D3D11RenderingPane.Viewport.Width, (int)D3D11RenderingPane.Viewport.Height );                    
                }
            }
        }

        public void Dispose()
        {
            if ( D3D11RenderingPane != null )
            {
                D3D11RenderingPane.Dispose();
                D3D11RenderingPane = null;
            }
        }

        private readonly DispatcherTimer mResizeTimer = new DispatcherTimer();

        public void SetSize( Size oldSize, Size newSize )
        {
            if ( D3D11RenderingPane != null )
            {
                mResizeTimer.Stop();
                mResizeTimer.Interval = TimeSpan.FromSeconds( 0.01 );
                mResizeTimer.Tag = newSize;
                mResizeTimer.Start();
                //D3D11RenderingPane.SetSize( newSize );                
            }

            if ( UserInputHandler != null )
            {
                UserInputHandler.SetSize( (int)oldSize.Width, (int)oldSize.Height, (int)newSize.Width, (int)newSize.Height );                
            }
        }

        public void ResizeRenderingPane( object sender, EventArgs eventArgs )
        {
            mResizeTimer.Stop();
            D3D11RenderingPane.SetSize( (Size)mResizeTimer.Tag );
        }

    }
}
