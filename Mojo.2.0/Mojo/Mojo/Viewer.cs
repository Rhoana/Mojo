using System;
using System.Windows;

namespace Mojo
{
    public class Viewer : IDisposable
    {
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

        public void SetSize( Size oldSize, Size newSize )
        {
            if ( D3D11RenderingPane != null )
            {
                D3D11RenderingPane.SetSize( newSize );                
            }

            if ( UserInputHandler != null )
            {
                UserInputHandler.SetSize( (int)oldSize.Width, (int)oldSize.Height, (int)newSize.Width, (int)newSize.Height );                
            }
        }
    }
}
