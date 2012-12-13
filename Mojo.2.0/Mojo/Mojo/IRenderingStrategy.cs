using System;
using SlimDX.Direct3D11;

namespace Mojo
{
    public interface IRenderingStrategy : IDisposable
    {
        void Render( DeviceContext deviceContext, Viewport viewport, RenderTargetView renderTargetView, DepthStencilView depthStencilView );
    }
}
