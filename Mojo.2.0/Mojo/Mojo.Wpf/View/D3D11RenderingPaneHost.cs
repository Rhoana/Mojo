using System.Windows.Forms;

namespace Mojo.Wpf.View
{
    internal class D3D11RenderingPaneHost : UserControl
    {
        public ViewerContext ViewerContext { get; set; }

        protected override void OnPaintBackground( PaintEventArgs e )
        {
        }

        protected override void OnMouseDown( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseDown( e, Width, Height );                
            }
        }

        protected override void OnMouseUp( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseUp( e, Width, Height );
            }
        }

        protected override void OnMouseClick( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseClick( e, Width, Height );
            }
        }

        protected override void OnMouseMove( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseMove( e, Width, Height );
            }
        }

        protected override void OnMouseWheel( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseWheel( e, Width, Height );
            }
        }
    }
}
