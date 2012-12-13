using System;
using SlimDX.Direct3D11;
using SlimDX.DXGI;

namespace Mojo
{
    public static class D3D11
    {
        public static void Initialize( out Factory dxgiFactory, out SlimDX.Direct3D11.Device d3d11Device )
        {
            dxgiFactory = new Factory();

            Adapter adapter = null;
            long videoMemory = 0;

            for ( var i = 0; i < dxgiFactory.GetAdapterCount(); i++ )
            {
                var tmpAdapter = dxgiFactory.GetAdapter( i );

                if ( tmpAdapter.Description.DedicatedVideoMemory > videoMemory )
                {
                    adapter = tmpAdapter;
                    videoMemory = tmpAdapter.Description.DedicatedVideoMemory;
                }
            }

            d3d11Device = null;

            try
            {
                d3d11Device = Constants.DEBUG_D3D11_DEVICE
                                  ? new SlimDX.Direct3D11.Device( adapter,
                                                                  DeviceCreationFlags.Debug,
                                                                  new[] { FeatureLevel.Level_11_0, FeatureLevel.Level_10_1, FeatureLevel.Level_10_0 } )
                                  : new SlimDX.Direct3D11.Device( adapter,
                                                                  DeviceCreationFlags.None,
                                                                  new[] { FeatureLevel.Level_11_0, FeatureLevel.Level_10_1, FeatureLevel.Level_10_0 } );                
            }
            catch ( Exception e )
            {
                Console.WriteLine( "\nError: Couldn't create Direct3D 11 device (exception: " + e.Source + ", " + e.Message + ").\n" );
            }
        }

        public static void Terminate( ref Factory dxgiFactory, ref SlimDX.Direct3D11.Device d3d11Device )
        {
            d3d11Device.Dispose();
            d3d11Device = null;

            dxgiFactory.Dispose();
            dxgiFactory = null;
        }
    }
}
