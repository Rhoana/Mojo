using System;
using System.Linq;
using Mojo.Interop;
using SlimDX.DXGI;

namespace Mojo
{
    public enum ViewerMode
    {
        TileManager2D
    }

    public enum ToolMode
    {
        Null,
        AdjustSegmentation,
        MergeSegmentation,
        SplitSegmentation,
    }

    public enum MergeMode
    {
        Fill2D,
        Fill3D,
        GlobalReplace
    }

    public enum SplitMode
    {
        JoinPoints,
        DrawSplit,
        DrawRegions
    }

    public class Engine : NotifyPropertyChanged, IDisposable
    {
        private Factory mDxgiFactory;
        private SlimDX.Direct3D11.Device mD3D11Device;

        public TileManager TileManager { get; private set; }
        public Segmenter Segmenter { get; private set; }

        public ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, ITool>> Tools { get; private set; }
        public ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, IRenderingStrategy>> RenderingStrategies { get; private set; }
        public ObservableDictionary<ViewerMode, Viewer> Viewers { get; private set; }

        private ToolMode mCurrentToolMode;
        private bool mToolModeChanging = false;

        public ToolMode CurrentToolMode
        {
            get { return mCurrentToolMode; }
            set
            {

                /*
                 * TODO: remove mToolModeChanging (Currently required because a call loop can be created).
                 * On change from Split to Merge mode using keyboard shortcut this method is called twice and ends up back in Split mode.
                 */

                if ( value != mCurrentToolMode && !mToolModeChanging )
                {
                    mToolModeChanging = true;
                    mCurrentToolMode = value;

                    Tools.Internal.ToList().ForEach( viewerModeToolsMap => viewerModeToolsMap.Value.Internal[mCurrentToolMode].Select() );

                    Viewers.Internal.ToList()
                           .ForEach( viewer => viewer.Value.D3D11RenderingPane.RenderingStrategy = RenderingStrategies.Internal[viewer.Key].Internal[mCurrentToolMode] );
                    Viewers.Internal.ToList().ForEach( viewer => viewer.Value.UserInputHandler = Tools.Internal[viewer.Key].Internal[mCurrentToolMode] );

                    OnPropertyChanged( "CurrentToolMode" );
                    mToolModeChanging = false;
                }
            }
        }

        public Engine( ObservableDictionary<string, D3D11HwndDescription> d3d11HwndDescriptions )
        {
            Console.WriteLine( "\nMojo initializing...\n" );

            D3D11.Initialize( out mDxgiFactory, out mD3D11Device );
            Cuda.Initialize( mD3D11Device );
            Thrust.Initialize();

            TileManager = new TileManager( new Interop.TileManager( mD3D11Device, mD3D11Device.ImmediateContext, Constants.ConstParameters ) );
            Segmenter = new Segmenter( new Interop.Segmenter( mD3D11Device, mD3D11Device.ImmediateContext, Constants.ConstParameters ) );

            Tools = new ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, ITool>>
                    {
                        {
                            ViewerMode.TileManager2D,
                            new ObservableDictionary< ToolMode, ITool >
                            {
                                { ToolMode.Null, new NullTool() },
                                { ToolMode.AdjustSegmentation, new NullTool() },
                                { ToolMode.MergeSegmentation, new MergeSegmentationTool( TileManager, this ) },
                                { ToolMode.SplitSegmentation, new SimpleSegmenterTool( TileManager, this ) }
                            }
                            }
                    };

            RenderingStrategies = new ObservableDictionary<ViewerMode, ObservableDictionary<ToolMode, IRenderingStrategy>>
                                  {
                                      {
                                          ViewerMode.TileManager2D,
                                          new ObservableDictionary< ToolMode, IRenderingStrategy >
                                          {
                                              { ToolMode.Null, new NullRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext ) },
                                              { ToolMode.AdjustSegmentation, new NullRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext ) },
                                              { ToolMode.MergeSegmentation, new MergeSegmentationRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) },
                                              { ToolMode.SplitSegmentation, new SimpleSegmenterRenderingStrategy( mD3D11Device, mD3D11Device.ImmediateContext, TileManager ) }
                                          }
                                          }
                                  };

            Viewers = new ObservableDictionary<ViewerMode, Viewer>
                      {
                          {
                              ViewerMode.TileManager2D,
                              new Viewer
                              {
                                  D3D11RenderingPane = new D3D11RenderingPane( mDxgiFactory,
                                                                               mD3D11Device,
                                                                               mD3D11Device.ImmediateContext,
                                                                               d3d11HwndDescriptions.Get( "TileManager2D" ) )
                                                       {
                                                           RenderingStrategy = RenderingStrategies.Internal[ ViewerMode.TileManager2D ].Internal[ ToolMode.Null ]
                                                       },
                                  UserInputHandler = Tools.Internal[ ViewerMode.TileManager2D ].Internal[ ToolMode.Null ]
                              }
                              }
                      };

        }

        public void Dispose()
        {

            Viewers.Internal.Values.ToList().ForEach( viewer => viewer.Dispose() );
            Viewers.Internal.Clear();

            RenderingStrategies.Internal.Values.ToList().ForEach( renderingStrategies => renderingStrategies.Internal.Values.ToList().ForEach( renderingStrategy => renderingStrategy.Dispose() ) );
            RenderingStrategies.Internal.Clear();

            if ( Segmenter != null )
            {
                Segmenter.Dispose();
                Segmenter = null;
            }

            if ( TileManager != null )
            {
                TileManager.Dispose();
                TileManager = null;
            }

            Thrust.Terminate();
            Cuda.Terminate();
            D3D11.Terminate( ref mDxgiFactory, ref mD3D11Device );

            Console.WriteLine( "\nMojo terminating...\n" );
        }

        public void Update()
        {
            TileManager.Update();
            Segmenter.Update();

            Viewers.Internal.ToList().ForEach( viewer => viewer.Value.D3D11RenderingPane.Render() );
        }
    }
}
