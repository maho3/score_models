import torch
import torch.nn.functional as F
from score_models.layers import StyleGANConv, UpsampleLayer, DownsampleLayer, Combine, ResnetBlockBigGANpp
from score_models.layers import Conv3dSame
from score_models.layers.attention_block import SelfAttentionBlock, ScaledAttentionLayer
from score_models.definitions import default_init
from score_models.utils import get_activation
import numpy as np

def init_test_fn(shape, dtype=torch.float32, device="cpu"):
    return torch.ones(shape, dtype=dtype, device=device)

def test_attention():
    x = torch.randn([10, 4, 8, 8])
    print(x[0, 0, 0, 0], x[0, 0, 0, 1])
    att = SelfAttentionBlock(4)
    y = att(x)
    print(y[0, 0, 0, 0], y[0, 0, 0, 1])
    x = torch.randn([10, 4, 8, 8, 8])
    SelfAttentionBlock(4, dimensions=3)(x)
    x = torch.randn([10, 4, 8])
    SelfAttentionBlock(4, dimensions=1)(x)
    
    x = torch.randn(10, 5) * 100
    B, D = x.shape
    temb = torch.randn(B, D)
    context = torch.stack([x, temb], dim=1)
    print("context shape", context.shape)
    att = ScaledAttentionLayer(dimensions=5)
    out = att(x.view(B, 1, D), context)
    print("shape",out.shape)
    print("out", out)


def test_resnet_biggan():
    # out channels has to be at least 4
    act = get_activation("relu")
    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=None, up=False, down=False, fir=False, skip_rescale=True, dimensions=2) 
    x = torch.randn(1, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 8, 8]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 8, 8, 8]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=True, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 16, 16, 16]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 4, 4, 4]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=False, dimensions=1) 
    x = torch.randn(1, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 4]

def test_combine():
    x = torch.randn(1, 1, 8, 8)
    y = torch.randn(1, 1, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=2)
    out = layer(x, y)
    assert list(out.shape) == [1, 5, 8, 8]

    x = torch.randn(1, 1, 8, 8, 8)
    y = torch.randn(1, 1, 8, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=3)
    out = layer(x, y)
    assert list(out.shape) == [1, 5, 8, 8, 8]

    x = torch.randn(1, 1, 8, 8, 8)
    y = torch.randn(1, 4, 8, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="sum", dimensions=3)
    out = layer(x, y)
    assert list(out.shape) == [1, 4, 8, 8, 8]


def test_upsample_layer():
    x = torch.randn(1, 1, 8, 8)
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert list(out.shape) == [1, 3, 16, 16] 

    x = torch.randn(1, 1, 8)
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 3, 16] 

    x = torch.randn(1, 1, 8)
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 1, 16] 

    x = torch.randn(1, 1, 8, 8, 8)
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert list(out.shape) == [1, 1, 16, 16, 16] 
    

def test_downsample_layer():
    x = torch.randn(1, 1, 8, 8)
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert list(out.shape) == [1, 3, 4, 4] 

    x = torch.randn(1, 1, 8)
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 3, 4] 

    x = torch.randn(1, 1, 8)
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 1, 4] 

    x = torch.randn(1, 1, 8, 8, 8)
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert list(out.shape) == [1, 1, 4, 4, 4] 


def test_stylegan_conv_shape():
    x = torch.randn(1, 1, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8, 8]
   
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4, 4]
    
    x = torch.randn(1, 1, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4]

    x = torch.randn(1, 1, 8, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8, 8, 8]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16, 16, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4, 4, 4]


def test_stylegan_conv_resample_kernel():
    x = torch.ones(1, 1, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert np.all(out.detach().numpy()[..., 2:-2, 2:-2] == 9.)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert np.all(out.detach().numpy()[..., 1:-1, 1:-1] == 9.)

def test_transposed_conv():
    # Test that we can downsample and upsample odd numbered images with correct padding
    from score_models.layers import ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame
    from score_models.layers import Conv1dSame, Conv2dSame, Conv3dSame
    
    B = 10
    D = 15
    C = 16
    K = 3
    for dim in [1, 2, 3]:
        x = torch.randn(B, C, *[D]*dim)
        layer_ = [Conv1dSame, Conv2dSame, Conv3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        x_down= layer(x)
        print("Down", x_down.shape)
        assert x_down.shape == torch.Size([B, C, *[D//2]*dim])
        
        layer_ = [ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        y = layer(x_down)
        print("Up", x.shape)
        assert x.shape == torch.Size([B, C, *[D]*dim])


def test_conv3dsame_stride_circular_padding_matches_reference():
    layer = Conv3dSame(1, 1, kernel_size=3, stride=2, bias=False, padding_mode="circular")
    with torch.no_grad():
        layer.conv.weight.fill_(1.0)

    x = torch.arange(9 * 9 * 9, dtype=torch.float32).reshape(1, 1, 9, 9, 9)
    y = layer(x)

    output_size = x.shape[-1] // layer.stride
    pad = ((output_size - 1) * layer.stride + 1 + layer.dilation * (layer.kernel_size - 1) - x.shape[-1]) // 2
    padded = F.pad(x, (pad, pad + 1, pad, pad + 1, pad, pad + 1), mode="circular")
    y_ref = F.conv3d(padded, layer.conv.weight, bias=None, stride=2, padding=0)

    assert torch.allclose(y, y_ref)
        
