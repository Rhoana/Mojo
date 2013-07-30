using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    class DrawMergeSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;
        private bool mCurrentlyMovingZ = false;

        public DrawMergeSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
        }

        public override void Select()
        {
            var p = new Vector2( 0.5f, 0.5f );
            mTileManager.PrepForDrawMerge( p );
            mTileManager.ResetDrawMergeState( p );
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForDrawMerge( p );
            }
        }

        public override void MoveZ()
        {
            if ( mTileManager.SelectedSegmentId != 0 && !mCurrentlyMovingZ )
            {
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForDrawMerge( p );
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {

            var p = new Vector2( 0.5f, 0.5f );

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
                        mTileManager.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.RedoChange();
                        mTileManager.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.SelectNewId();
                        mTileManager.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.Left:
                case System.Windows.Input.Key.Right:
                case System.Windows.Input.Key.Up:
                case System.Windows.Input.Key.Down:
                case System.Windows.Input.Key.X:
                case System.Windows.Input.Key.C:
                    mTileManager.PrepForDrawMerge( p );
                    break;

                case System.Windows.Input.Key.OemComma:
                case System.Windows.Input.Key.OemMinus:
                case System.Windows.Input.Key.Subtract:
                    mTileManager.DecreaseMergeBrushSize();
                    break;
                case System.Windows.Input.Key.OemPeriod:
                case System.Windows.Input.Key.OemPlus:
                case System.Windows.Input.Key.Add:
                    mTileManager.IncreaseMergeBrushSize();
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

        public override void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseDown( mouseEventArgs, width, height );

            if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                //
                // Draw here
                //

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                if ( mouseEventArgs.Button == MouseButtons.Left || mouseEventArgs.Button == MouseButtons.Right )
                {
                    mTileManager.PrepForDrawMerge( p );
                    mCurrentlyDrawing = true;
                    mTileManager.DrawRegionA( p, mTileManager.MergeBrushSize );
                    mEngine.QuickRender();
                }
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseUp( mouseEventArgs, width, height );

            var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

            if ( mouseEventArgs.Button == MouseButtons.Middle )
            {
                mTileManager.PrepForDrawMerge( p );
            }
            else if ( mCurrentlyDrawing && ( mouseEventArgs.Button == MouseButtons.Left || mouseEventArgs.Button == MouseButtons.Right ) )
            {
                mCurrentlyDrawing = false;

                switch ( mTileManager.CurrentMergeMode )
                {
                    case MergeMode.Fill2D:
                        mTileManager.CommitDrawMergeCurrentSlice();
                        break;
                    case MergeMode.Fill3D:
                        mTileManager.CommitDrawMergeCurrentConnectedComponent();
                        break;
                    default:
                        mTileManager.CommitDrawMerge();
                        break;
                }
            }
            mCurrentlyDrawing = false;
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.SourceImagesLoaded && mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                mCurrentlyHandlingMouseOver = true;

                //
                // Mouseover - update display to highlight segment or area under mouse
                // Get the id of the segment under the mouse
                //

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                mTileManager.MouseOver( p );

                if ( mCurrentlyDrawing )
                {
                    mTileManager.DrawRegionA( p, mTileManager.MergeBrushSize );
                    mEngine.QuickRender();
                }

                mCurrentlyHandlingMouseOver = false;

            }
        }

        public override void OnMouseWheel( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            //
            // Only scroll if the mouse within the extent
            //
            if ( mouseEventArgs.X < 0 || mouseEventArgs.X > width ||
                 mouseEventArgs.Y < 0 || mouseEventArgs.Y > height )
            {
                return;
            }

            base.OnMouseWheel( mouseEventArgs, width, height );

            var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );
            mTileManager.PrepForDrawMerge( p );
        }


    }
}
