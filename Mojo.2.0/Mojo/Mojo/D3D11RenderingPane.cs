using System;
using System.Windows;
using SlimDX;
using SlimDX.Direct3D11;
using SlimDX.DXGI;

namespace Mojo
{
    public class D3D11RenderingPane : IDisposable
    {
        private readonly Factory mDxgiFactory;
        private readonly SlimDX.Direct3D11.Device mD3D11Device;
        private readonly DeviceContext mD3D11DeviceContext;
        private SwapChain mSwapChain;
        private Texture2D mD3D11RenderTargetTexture2D;
        private Texture2D mD3D11DepthStencilTexture2D;
        private RenderTargetView mD3D11RenderTargetView;
        private DepthStencilView mD3D11DepthStencilView;

        public Viewport Viewport { get; private set; }
        public IRenderingStrategy RenderingStrategy { get; set; }

        public D3D11RenderingPane( Factory dxgiFactory, SlimDX.Direct3D11.Device d3D11Device, DeviceContext d3D11DeviceContext, D3D11HwndDescription d3D11HwndDescription )
        {
            mDxgiFactory = dxgiFactory;
            mD3D11Device = d3D11Device;
            mD3D11DeviceContext = d3D11DeviceContext;

            var swapChainDescription = new SwapChainDescription
                                       {
                                           BufferCount = 1,
                                           ModeDescription =
                                               new ModeDescription( d3D11HwndDescription.Width,
                                                                    d3D11HwndDescription.Height,
                                                                    new Rational( 60, 1 ),
                                                                    Format.R8G8B8A8_UNorm ),
                                           IsWindowed = true,
                                           OutputHandle = d3D11HwndDescription.Handle,
                                           SampleDescription = new SampleDescription( 1, 0 ),
                                           SwapEffect = SwapEffect.Discard,
                                           Usage = Usage.RenderTargetOutput
                                       };

            mSwapChain = new SwapChain( mDxgiFactory, mD3D11Device, swapChainDescription );
            mDxgiFactory.SetWindowAssociation( d3D11HwndDescription.Handle, WindowAssociationFlags.IgnoreAll );

            CreateD3D11Resources( d3D11HwndDescription.Width, d3D11HwndDescription.Height );
        }

        public void Dispose()
        {
            DestroyD3D11Resources();

            if ( mSwapChain != null )
            {
                mSwapChain.Dispose();
                mSwapChain = null;
            }

            if ( RenderingStrategy != null )
            {
                RenderingStrategy.Dispose();
                RenderingStrategy = null;
            }
        }

        public void Render()
        {
            mD3D11DeviceContext.OutputMerger.SetTargets( mD3D11DepthStencilView, mD3D11RenderTargetView );
            mD3D11DeviceContext.Rasterizer.SetViewports( Viewport );

            if ( RenderingStrategy != null )
            {
                RenderingStrategy.Render( mD3D11DeviceContext, Viewport, mD3D11RenderTargetView, mD3D11DepthStencilView );                
            }

            mSwapChain.Present( 0, PresentFlags.None );
        }

        public void SetSize( Size size )
        {
            DestroyD3D11Resources();

            mSwapChain.ResizeBuffers( 1,
                                        (int)size.Width,
                                        (int)size.Height,
                                        Format.R8G8B8A8_UNorm,
                                        SwapChainFlags.None );

            mSwapChain.ResizeTarget( new ModeDescription( (int)size.Width,
                                                            (int)size.Height,
                                                            new Rational( 60, 1 ),
                                                            Format.R8G8B8A8_UNorm ) );

            CreateD3D11Resources( (int)size.Width, (int)size.Height );

        }

        private void CreateD3D11Resources( int width, int height )
        {
            mD3D11RenderTargetTexture2D = SlimDX.Direct3D11.Resource.FromSwapChain<Texture2D>( mSwapChain, 0 );
            mD3D11RenderTargetView = new RenderTargetView( mD3D11Device, mD3D11RenderTargetTexture2D );

            var depthStencilTexture2DDescription = new Texture2DDescription
            {
                BindFlags = BindFlags.DepthStencil,
                Format = Format.D24_UNorm_S8_UInt,
                Width = width,
                Height = height,
                MipLevels = 1,
                SampleDescription = new SampleDescription( 1, 0 ),
                Usage = ResourceUsage.Default,
                OptionFlags = ResourceOptionFlags.Shared,
                CpuAccessFlags = CpuAccessFlags.None,
                ArraySize = 1
            };

            mD3D11DepthStencilTexture2D = new Texture2D( mD3D11Device, depthStencilTexture2DDescription );
            mD3D11DepthStencilView = new DepthStencilView( mD3D11Device, mD3D11DepthStencilTexture2D );

            Viewport = new Viewport( 0, 0, width, height, 0f, 1f );
        }

        private void DestroyD3D11Resources()
        {
            if ( mD3D11DepthStencilTexture2D != null )
            {
                mD3D11DepthStencilView.Dispose();
                mD3D11DepthStencilView = null;
            }

            if ( mD3D11DepthStencilTexture2D != null )
            {
                mD3D11DepthStencilTexture2D.Dispose();
                mD3D11DepthStencilTexture2D = null;
            }

            if ( mD3D11RenderTargetView != null )
            {
                mD3D11RenderTargetView.Dispose();
                mD3D11RenderTargetView = null;
            }

            if ( mD3D11RenderTargetTexture2D != null )
            {
                mD3D11RenderTargetTexture2D.Dispose();
                mD3D11RenderTargetTexture2D = null;
            }
        }
    }
}
