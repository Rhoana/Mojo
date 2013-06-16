using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace Mojo.Wpf.View
{
    /// <summary>
    ///   Interaction logic for ViewerContext.xaml
    /// </summary>
    public partial class ViewerContext
    {
        // Viewer Dependency Property
        public Viewer Viewer
        {
            get
            {
                return GetValue( ViewerProperty ) as Viewer;
            }
            set
            {
                SetValue( ViewerProperty, value );
            }
        }

        public static readonly DependencyProperty ViewerProperty = DependencyProperty.Register(
            "Viewer",
            typeof ( Viewer ),
            typeof ( ViewerContext ),
            new FrameworkPropertyMetadata() );

        public ViewerContext()
        {
            Loaded += LoadedHandler;
            SizeChanged += SizeChangedHandler;
            Unloaded += UnloadedHandler;

            InitializeComponent();
        }

        public void OnMouseUp( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseUp( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnMouseDown( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseDown( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnMouseClick( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseClick( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnMouseDoubleClick( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseDoubleClick( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnMouseMove( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseMove( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnMouseWheel( System.Windows.Forms.MouseEventArgs e, int width, int height )
        {
            if ( Viewer != null )
            {
                Viewer.UserInputHandler.OnMouseWheel( e, width, height );
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        public void OnManipulationStarting( object sender, System.Windows.Input.ManipulationStartingEventArgs e )
        {
            Console.WriteLine( "ViewerContext got manipulation started event." );
        }

        public void OnManipulationDelta( object sender, System.Windows.Input.ManipulationDeltaEventArgs e )
        {
            Console.WriteLine( "ViewerContext got manipulation delta event." );
        }

        public void OnManipulationInertiaStarting( object sender, System.Windows.Input.ManipulationInertiaStartingEventArgs e )
        {
            Console.WriteLine( "ViewerContext got manipulation inertia started event." );
        }

        protected override void OnKeyDown( KeyEventArgs e )
        {
            if (Viewer != null)
            {
                Viewer.UserInputHandler.OnKeyDown(e, D3D11RenderingPaneHost.Width, D3D11RenderingPaneHost.Height);
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        protected override void OnKeyUp(KeyEventArgs e)
        {
            if (Viewer != null)
            {
                Viewer.UserInputHandler.OnKeyUp(e, D3D11RenderingPaneHost.Width, D3D11RenderingPaneHost.Height);
                AquireKeyboardFocusAndLogicalFocus();
            }
        }

        private void LoadedHandler(object sender, RoutedEventArgs e)
        {
            D3D11RenderingPaneHost.ViewerContext = this;

            SetSize( new Size( ActualWidth, ActualHeight ), new Size( ActualWidth, ActualHeight ) );
            AquireKeyboardFocusAndLogicalFocus();
        }

        private void UnloadedHandler( object sender, RoutedEventArgs e )
        {
            D3D11RenderingPaneHost.ViewerContext = null;

            Loaded -= LoadedHandler;
            SizeChanged -= SizeChangedHandler;
            Unloaded -= UnloadedHandler;
        }

        private void SizeChangedHandler( object sender, SizeChangedEventArgs e )
        {
            SetSize( e.PreviousSize, e.NewSize );
            //AquireKeyboardFocusAndLogicalFocus();
        }

        private void AquireKeyboardFocusAndLogicalFocus()
        {
            WindowsFormsHost.TabInto( new TraversalRequest( FocusNavigationDirection.First ) );
            Keyboard.Focus( this );
        }

        private void SetSize( Size oldSize, Size newSize )
        {
            if ( WindowsFormsHost != null )
            {
                WindowsFormsHost.Width = newSize.Width;
                WindowsFormsHost.Height = newSize.Height;

                var source = PresentationSource.FromVisual( WindowsFormsHost );
                Matrix transformToDevice = source.CompositionTarget.TransformToDevice;

                var oldPixelSize = (Size)transformToDevice.Transform( (Vector)oldSize );
                var newPixelSize = (Size)transformToDevice.Transform( (Vector)newSize );

                if ( D3D11RenderingPaneHost != null )
                {
                    D3D11RenderingPaneHost.Width = (int)newPixelSize.Width;
                    D3D11RenderingPaneHost.Height = (int)newPixelSize.Height;
                }

                if ( Viewer != null )
                {
                    Viewer.SetSize( oldPixelSize, newPixelSize );
                }
            }
        }

    }
}
