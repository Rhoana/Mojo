using System.Diagnostics;
using SlimDX.Direct3D11;
//using TinyText;

namespace Mojo
{
    public class NullRenderingStrategy : IRenderingStrategy
    {
        private readonly Stopwatch mStopwatch = new Stopwatch();

        //private Context mTinyTextContext;

        private string FrameTimeString
        {
            get
            {
                return mStopwatch.ElapsedMilliseconds == 0 ? "< 1 ms" : mStopwatch.ElapsedMilliseconds + " ms";
            }
        }

        public NullRenderingStrategy( SlimDX.Direct3D11.Device device, DeviceContext deviceContext )
        {
            //bool result;
            //mTinyTextContext = new Context( device, deviceContext, Constants.MAX_NUM_TINY_TEXT_CHARACTERS, out result );
            //Release.Assert( result );

            mStopwatch.Start();
        }

        public void Dispose()
        {
            //if ( mTinyTextContext != null )
            //{
            //    mTinyTextContext.Dispose();
            //    mTinyTextContext = null;
            //}
        }

        public void Render( DeviceContext deviceContext, Viewport viewport, RenderTargetView renderTargetView, DepthStencilView depthStencilView )
        {
            deviceContext.ClearRenderTargetView( renderTargetView, Constants.CLEAR_COLOR );
            deviceContext.ClearDepthStencilView( depthStencilView, DepthStencilClearFlags.Depth | DepthStencilClearFlags.Stencil, 1.0f, 0x00 );

            //mTinyTextContext.Print( viewport, "Frame Time: " + FrameTimeString, 10, 10 );
            //mTinyTextContext.Render();

            mStopwatch.Reset();
            mStopwatch.Start();
        }
    }
}