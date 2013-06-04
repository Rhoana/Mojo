using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    class AdjustSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;
        private bool mCurrentlyMovingZ = false;

        public AdjustSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
        }

        public override void Select()
        {
            if ( mTileManager.SelectedSegmentId != 0 )
            {
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
            }
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
            }
        }

        public override void MoveZ()
        {
            if ( mTileManager.SelectedSegmentId != 0 && !mCurrentlyMovingZ )
            {
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = true;
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = true;
                    break;
                case System.Windows.Input.Key.Q:
                    mTileManager.ToggleShowBoundaryLines();
                    break;
                case System.Windows.Input.Key.A:
                    mTileManager.ToggleShowSegmentation();
                    break;
                case System.Windows.Input.Key.E:
                    mTileManager.IncreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.D:
                    mTileManager.DecreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.Z:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.UndoChange();
                        mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
                    }
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.RedoChange();
                        mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
                    }
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.SelectNewId();
                        mTileManager.Internal.PrepForAdjust( mTileManager.SelectedSegmentId, p );
                    }
                    break;
                case System.Windows.Input.Key.Tab:
                    mTileManager.CommmitAdjustChange();
                    keyEventArgs.Handled = true;
                    break;

                case System.Windows.Input.Key.Escape:
                    mTileManager.CancelAdjustChange();
                    mTileManager.SelectedSegmentId = 0;
                    break;

                case System.Windows.Input.Key.OemComma:
                case System.Windows.Input.Key.OemMinus:
                case System.Windows.Input.Key.Subtract:
                    mTileManager.DecreaseBrushSize();
                    break;
                case System.Windows.Input.Key.OemPeriod:
                case System.Windows.Input.Key.OemPlus:
                case System.Windows.Input.Key.Add:
                    mTileManager.IncreaseBrushSize();
                    break;

            }

            base.OnKeyDown( keyEventArgs, width, height );

        }

        public override void OnKeyUp(System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height)
        {
            base.OnKeyUp(keyEventArgs, width, height);
            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
            }
        }
        
        public override void OnMouseDown(System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height)
        {
            base.OnMouseDown( mouseEventArgs, width, height );
            if ( mTileManager.SegmentationLoaded && mTileManager.SelectedSegmentId != 0 )
            {
                //
                // Draw or erase here
                //

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                mCurrentlyDrawing = true;

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                    mEngine.QuickRender();
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                    mEngine.QuickRender();
                }
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseUp( mouseEventArgs, width, height );
            if ( mCurrentlyDrawing )
            {
                mCurrentlyDrawing = false;
            }
        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && mTileManager.SelectedSegmentId == 0 )
            {
                //Get the id of the segment being clicked

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float) mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float) mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                var clickedId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( clickedId > 0 && mouseEventArgs.Button != MouseButtons.Middle )
                {
                    //
                    // Select this segment
                    //
                    mTileManager.Internal.PrepForAdjust( clickedId, p );
                    mTileManager.SelectedSegmentId = clickedId;
                }
            }
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded )
            {
                mCurrentlyHandlingMouseOver = true;

                //
                // Mouseover - update display to highlight segment or area under mouse
                // Get the id of the segment under the mouse
                //

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                mTileManager.MouseOverSegmentId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( mTileManager.SelectedSegmentId != 0 )
                {
                    //
                    // Make a hover circle
                    //
                    mTileManager.MouseOverX = x;
                    mTileManager.MouseOverY = y;

                    if ( mCurrentlyDrawing )
                    {
                        if ( mouseEventArgs.Button == MouseButtons.Left )
                        {
                            mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                            mEngine.QuickRender();
                        }
                        else if ( mouseEventArgs.Button == MouseButtons.Right )
                        {
                            mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                            mEngine.QuickRender();
                        }
                    }
                }
                else
                {
                    mTileManager.MouseOverX = 0;
                    mTileManager.MouseOverY = 0;
                }

                mCurrentlyHandlingMouseOver = false;

            }
        }

    }
}
