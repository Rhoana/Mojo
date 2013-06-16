using System.Windows.Forms;

namespace Mojo.Wpf.View
{
    internal class D3D11RenderingPaneHost : UserControl
    {
        public D3D11RenderingPaneHost()
        {
            AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            AutoScaleDimensions = new System.Drawing.SizeF( 96F, 96F );
        }

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

        protected override void OnMouseDoubleClick( MouseEventArgs e )
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnMouseDoubleClick( e, Width, Height );
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

        protected void OnManipulationDelta( System.Windows.Input.ManipulationDeltaEventArgs e ) 
        {
            if ( ViewerContext != null )
            {
                ViewerContext.OnManipulationDelta( e, Width, Height );
            }
        }

        public System.Drawing.SizeF GetAutoScaleFactor()
        {
            return AutoScaleFactor;
        }
    }
}
