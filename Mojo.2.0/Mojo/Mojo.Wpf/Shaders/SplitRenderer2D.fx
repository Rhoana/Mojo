#define BLACK      float4( 0.0f, 0.0f, 0.0f, 1.0f )
#define WHITE      float4( 1.0f, 1.0f, 1.0f, 1.0f )
#define ERROR_PINK float4( 1.0f, 0.5f, 1.0f, 1.0f )

#define EQUALS( x, y, e ) ( abs( x - y ) < e )

#define SOURCE_TARGET 1
#define BORDER_TARGET 2
#define REGION_SPLIT 3
#define REGION_A 4
#define REGION_B 5
#define PATH_RESULT 6
#define MASK_VALUE 10

Texture3D<float4> gSourceTexture3D;
Texture3D<uint1>  gIdTexture3D;
Texture3D<uint1>  gOverlayTexture3D;
Buffer<float4>    gIdColorMapBuffer;
Buffer<uint1>     gLabelIdMapBuffer;
Buffer<float1>    gIdConfidenceMapBuffer;
float4x4          gTransform;
float             gSegmentationRatio;
bool              gBoundaryLinesVisible;
bool              gCrosshairVisible;
bool              gBrushVisible;
uint               gSelectedSegmentId;
uint               gMouseOverSegmentId;
float             gMouseOverX;
float             gMouseOverY;
float             gMouseHighlightSize;

sampler gSourceTextureSampler = 
sampler_state
{
    Filter = MIN_LINEAR_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;    
};

sampler gIdTextureSampler = 
sampler_state
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;    
};

struct VS_IN
{
    float4 position : POSITION;
    float4 texCoord : TEXCOORD0;
};

struct PS_IN
{
    float4 position : SV_POSITION;
    float4 texCoord : TEXCOORD0;
};

PS_IN VS( VS_IN input )
{
    PS_IN output = (PS_IN)0;
    
    output.position = mul( input.position, gTransform );
    output.texCoord = input.texCoord;

    return output;
}

float4 PS( PS_IN input ) : SV_Target
{
    float4 rawSourceColor = float4( gSourceTexture3D.Sample( gSourceTextureSampler, input.texCoord.xyz ).xyz, 1.0f );
    float4 sourceColor    = float4( rawSourceColor.xxx, 1.0f );

    int4   index3D        = int4( (int)( input.texCoord.x * 512 ), (int)( input.texCoord.y * 512 ), (int)( input.texCoord.z * 1 ), 0 );
	uint   previd          = gIdTexture3D.Load( index3D );
    uint   id              = gLabelIdMapBuffer.Load( previd );

	while ( id != previd )
	{
		previd = id;
		id = gLabelIdMapBuffer.Load( previd );
	}

	uint   overlay        = gOverlayTexture3D.Load( index3D );
    float4 idColor        = gIdColorMapBuffer.Load( id % gIdColorMapBuffer.Length );
    float1 idConfidence   = gIdConfidenceMapBuffer.Load( id );

	if ( idConfidence.x > 0.0f && index3D.x % 16 < 12 && index3D.y % 16 < 12 )
		idColor = float4( 0.0f, 0.0f, 0.3f, 0.0f );

	float4 returnColor    = ( ( 1.0f - gSegmentationRatio ) * sourceColor ) + ( gSegmentationRatio * idColor );

    float xDist           = ( (float) index3D.x ) - gMouseOverX * 512.0f + 0.5f;
	float yDist           = ( (float) index3D.y ) - gMouseOverY * 512.0f + 0.5f;
	float mouseDistance   = sqrt( xDist * xDist + yDist * yDist );

    bool border           = false;
    bool selectBorder     = false;

	if ( gBoundaryLinesVisible )
	{
		//Check for borders
		if ( index3D.x > 0 )
		{
			int4   index3DLeft      = int4( index3D.x - 1, index3D.y, index3D.z, 0 );
			uint   previdLeft       = gIdTexture3D.Load( index3DLeft );
			uint   idLeft           = gLabelIdMapBuffer.Load( previdLeft );
			while ( idLeft != previdLeft )
			{
				previdLeft = idLeft;
				idLeft = gLabelIdMapBuffer.Load( previdLeft );
			}
			if ( idLeft != id )
			{
				//idColor = float4( 0.0f, 0.0f, 0.0f, 0.0f );
				border = true;
				if ( id == gSelectedSegmentId || idLeft == gSelectedSegmentId )
				{
					selectBorder = true;
				}
			}
		}

		if ( index3D.y > 0 )
		{
			int4   index3DUp      = int4( index3D.x, index3D.y - 1, index3D.z, 0 );
			uint   previdUp       = gIdTexture3D.Load( index3DUp );
			uint   idUp           = gLabelIdMapBuffer.Load( previdUp );
			while ( idUp != previdUp )
			{
				previdUp = idUp;
				idUp = gLabelIdMapBuffer.Load( previdUp );
			}
			if ( idUp != id )
			{
				//idColor = float4( 0.0f, 0.0f, 0.0f, 0.0f );
				border = true;
				if ( id == gSelectedSegmentId || idUp == gSelectedSegmentId )
				{
					selectBorder = true;
				}
			}
		}

		if ( index3D.y > 0 && index3D.x > 0 )
		{
			int4   index3DLeftUp  = int4( index3D.x - 1, index3D.y - 1, index3D.z, 0 );
			uint   previdLeftUp       = gIdTexture3D.Load( index3DLeftUp );
			uint   idLeftUp           = gLabelIdMapBuffer.Load( previdLeftUp );
			while ( idLeftUp != previdLeftUp )
			{
				previdLeftUp = idLeftUp;
				idLeftUp = gLabelIdMapBuffer.Load( previdLeftUp );
			}
			if ( idLeftUp != id )
			{
				//idColor = float4( 0.0f, 0.0f, 0.0f, 0.0f );
				border = true;
				if ( id == gSelectedSegmentId || idLeftUp == gSelectedSegmentId )
				{
					selectBorder = true;
				}
			}
		}
	}

    if ( border && gSegmentationRatio > 0.0f )
    {
        if ( selectBorder && mouseDistance < gMouseHighlightSize && idConfidence.x == 0.0f )
		    returnColor = float4( 0.4f, 0.4f, 0.8f, 0.0f );
        else if ( selectBorder )
		    returnColor = float4( 1.0f, 1.0f, 1.0f, 0.0f );
        else
		    returnColor = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    }
	else if ( gBrushVisible && id != 0 && id == gSelectedSegmentId && ( mouseDistance < gMouseHighlightSize ) && idConfidence.x == 0.0f )
	{
		returnColor = sourceColor + float4( -0.2f, 0.2f, -0.2f, 0.0f);
	}
	else if ( gCrosshairVisible && id != 0 && id == gSelectedSegmentId && ( abs( xDist ) < 0.5f || abs( yDist ) < 0.5f ) && idConfidence.x == 0.0f )
	{
		returnColor = sourceColor + float4( -0.2f, -0.2f, 0.2f, 0.0f);
	}
	else if ( id != 0 && gSegmentationRatio > 0.0f )
	{
        if ( border )
        {
            returnColor = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        }
		else if ( id == gSelectedSegmentId && idConfidence.x == 0.0f )
		{
			returnColor = sourceColor * 1.2;
		}
		else if ( id == gSelectedSegmentId )
		{
			returnColor *= 1.4;
		}
		else if ( id == gMouseOverSegmentId )
		{
			returnColor *= 1.2;
		}
	}

	if ( gSegmentationRatio > 0.0f && idConfidence.x == 0.0f )
	{
		if ( overlay == BORDER_TARGET )
		{
			//Border target
			returnColor = float4( 0.2f, 0.2f, 0.4f, 0.0f);
		}
		else if ( overlay == SOURCE_TARGET )
		{
			//Source point
			returnColor = float4( 0.2f, 0.2f, 1.0f, 0.0f);
		}
		else if ( overlay == REGION_SPLIT && id == gSelectedSegmentId )
		{
			//Split bonus region ( drawn line )
		    returnColor = sourceColor + float4( -0.5f, 0.0f, -0.5f, 0.0f);
		}
		else if ( overlay == REGION_A && id == gSelectedSegmentId )
		{
			//Split bonus region ( drawn line )
		    returnColor = sourceColor + float4( 0.0f, -0.5f, -0.5f, 0.0f);
		}
		else if ( overlay == REGION_B && id == gSelectedSegmentId )
		{
			//Split bonus region ( drawn line )
		    returnColor = sourceColor + float4( -0.5f, -0.5f, 0.0f, 0.0f);
		}
		else if ( overlay == PATH_RESULT )
		{
			//Split line
			returnColor = float4( 0.2f, 1.0f, 0.2f, 0.0f);
		}
		else if ( overlay >= MASK_VALUE )
		{
            //Debug or hover line
			//returnColor = returnColor - float4( 0.1f, 0.1f, 0.1f, 0.0f) * (float) ( overlay - MASK_VALUE + 5 );
			if ( overlay % 3 == 0 )
				returnColor = float4( 0.1f, 0.0f, 0.0f, 0.0f) * (float) ( overlay - MASK_VALUE + 5 );
			if ( overlay % 3 == 1 )
				returnColor = float4( 0.0f, 0.1f, 0.0f, 0.0f) * (float) ( overlay - MASK_VALUE + 5 );
			if ( overlay % 3 == 2 )
				returnColor = float4( 0.0f, 0.0f, 0.1f, 0.0f) * (float) ( overlay - MASK_VALUE + 5 );
		}
	}

    return returnColor;
}

BlendState gBlendStateColorWriteMaskEnable
{
    RenderTargetWriteMask[ 0 ] = 15; // D3D11_COLOR_WRITE_ENABLE_ALL
};

RasterizerState gRasterizerState
{
    FillMode = Solid;
    CullMode = None;
};

DepthStencilState gDepthStencilState
{
    DepthEnable = true;
    DepthWriteMask = All;
    StencilEnable = false;
};

RasterizerState gRasterizerStateSolid
{
    FillMode = Solid;
    CullMode = None;
};

technique11 TileManager2D
{
    pass TileManager2D
    {
        SetBlendState( gBlendStateColorWriteMaskEnable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
        SetDepthStencilState( gDepthStencilState, 0x00000000 );
        SetRasterizerState( gRasterizerState );
        SetGeometryShader( 0 );
        SetVertexShader( CompileShader( vs_5_0, VS() ) );
        SetPixelShader( CompileShader( ps_5_0, PS() ) );
    }
}