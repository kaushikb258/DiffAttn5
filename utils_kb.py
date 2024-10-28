import torch
import torch.nn.functional as F
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import pdb
import time
import kornia.morphology as km


def apply_attention_kb(batch_size, input_ndim, hidden_states, encoder_hidden_states, attn, attention_mask, blur_sigma, inf_blur, attn_guid_option, bilat=None):

        # hidden_states = torch.Size([2, 576, 1280]) 
        # encoder_hidden_states = torch.Size([2, 576, 1280])
        # batch_size = 2
        # input_ndim = 3 or 4

        device = hidden_states.device

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape 

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads        

        # inner_dim = 1280
        # head_dim = 64
        # attn.heads = 20        

        if attn_guid_option == 1:            
            sigma_ag = 0.2
            w_rnd = sigma_ag * torch.randn(hidden_states.size(-1), query.size(-1)).to(device).type(torch.float16)
            _, h2 = hidden_states.chunk(2)
            q2_ = h2 @ w_rnd   
            assert query.size(0) == 2
            query[1, :, :] += q2_.squeeze(0)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            

        # query = torch.Size([2, 20, 576, 64]) 
        # key = torch.Size([2, 20, 576, 64]) 
        # value = torch.Size([2, 20, 576, 64])

        hidden_states = scaled_dot_product_attention_kb2(query, key, value, attn_mask=attention_mask, 
                        blur_sigma=blur_sigma, inf_blur=inf_blur, dropout_p=0.0, is_causal=False, attn_guid_option=attn_guid_option, bilat=bilat)  

        # hidden_states = torch.Size([2, 20, 576, 64])
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        height = width = math.isqrt(query.shape[2])

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)    

        return hidden_states 

         


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention_kb2(query, key, value, attn_mask=None, blur_sigma=1, inf_blur=False, dropout_p=0.0,
        is_causal=False, attn_guid_option=0, bilat=None, scale=None, enable_gqa=False) -> torch.Tensor:
    
    assert is_causal == False
    assert dropout_p <= 1e-8
    assert enable_gqa == False
    assert scale == None
    assert attn_mask == None
     
    device = query.device    
         
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(device)      

    q1, q2 = query.chunk(2)
    k1, k2 = key.chunk(2)
    v1, v2 = value.chunk(2)
    qk1 = q1 @ k1.transpose(-2, -1)
    qk2 = q2 @ k2.transpose(-2, -1)
    
    if attn_guid_option == 0:
        if not inf_blur:
            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
            qk2 = gaussian_blur_2d(qk2, kernel_size, blur_sigma)
        else:
            qk2 = qk2.mean(dim=(-2, -1), keepdim=True)  
            qk2 = qk2.expand(qk2.size(0), qk2.size(1), query.size(-2), query.size(-2))
    elif attn_guid_option == 2:
        # qk dropout
        pmax = 0.21
        prob = torch.empty(qk2.size()).uniform_(0, pmax).to(device) 
        binary = torch.bernoulli(prob).type(torch.float16)
        qk2 = qk2 * binary
    
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    #if enable_gqa:
    #    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    #    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight1 = qk1 * scale_factor
    attn_weight2 = qk2 * scale_factor
    
    attn_weight1 += attn_bias
    attn_weight2 += attn_bias
    
    attn_weight1 = torch.softmax(attn_weight1, dim=-1)
    attn_weight2 = torch.softmax(attn_weight2, dim=-1)
    
    if attn_guid_option == 3:
        # apply bilateral filter on attn_weight2        
        assert attn_weight2.size(0) == 1
        _, n, h, w = attn_weight2.size()
        x = []
        for i in range(n):
            y = attn_weight2[0,i].unsqueeze(0).unsqueeze(0)
            #y = F.interpolate(y, size=(h//2, w//2), mode='bilinear')
            y = bilat(y)
            #y = y.unsqueeze(0)
            #y = F.interpolate(y, size=(h, w), mode='bilinear')
            x.append(y)
        attn_weight2 = torch.stack(x, dim=0).type(torch.float16).reshape((1, n, h, w))    
    elif attn_guid_option in [4, 5]:
        # 4: erosion
        # 5: dilation
        if attn_guid_option == 4:
            y = erosion(attn_weight2)
        elif attn_guid_option == 5:
            y = dilation(attn_weight2)            
        attn_weight2 = y.type(torch.float16) #.reshape((1, n, h, w))    
        
        
    attn_weight1 = torch.dropout(attn_weight1, dropout_p, train=True)
    attn_weight2 = torch.dropout(attn_weight2, dropout_p, train=True)
    
    h1 = attn_weight1 @ v1
    h2 = attn_weight2 @ v2
    
    # h1 = torch.Size([1, 20, 576, 64]) 
    # h2 = torch.Size([1, 20, 576, 64])
    output = torch.cat((h1, h2), dim=0)
    
    return output       


# KAUSHIK: THE FOLLOWING FUNCTION IS CALLED ONLY IF CFG IS USED
def scaled_dot_product_attention_kb(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    
    assert is_causal == False
    assert dropout_p <= 1e-8
    assert enable_gqa == False
    assert scale == None
    
    print("should not be in scaled_dot_product_attention_kb()")
    assert 1 == 2
    
    device = query.device     
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(device)      
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    output = attn_weight @ value
    return output
    
    
    
# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img    
    
    
   



def gkern2d(l=21, sig=3):
    """Returns a 2D Gaussian kernel array."""
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel


class Shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size ** 2)
        
        self.pad = self.kernel_size // 2
        #if self.kernel_size == 3:
        #    self.pad = 1
        #elif self.kernel_size == 5:
        #    self.pad = 2
        #elif self.kernel_size == 7:
        #    self.pad = 3
        
    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        # Alias for convenience
        cpg = self.channels_per_group
        cat_layers = []
        for i in range(self.in_planes):
            #Parse in row-major
            for y in range(0,self.kernel_size):
                y2 = y+h
                for x in range(0, self.kernel_size):
                    x2 = x+w
                    xx = x_pad[:,i:i+1,y:y2,x:x2]
                    cat_layers += [xx]
        return torch.cat(cat_layers, 1)


class BilateralFilter(nn.Module):
    r"""BilateralFilter computes:
        If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(self, channels=3, k=7, height=480, width=640, sigma_space=5, sigma_color=0.1):
        super(BilateralFilter, self).__init__()

        #space gaussian kernel
        #FIXME: do everything in torch
        self.g = Parameter(torch.Tensor(channels,k*k))
        self.gw = gkern2d(k,sigma_space)

        gw = np.tile(self.gw.reshape(channels,k*k,1,1),(1,1,height,width))
        self.g.data = torch.from_numpy(gw).float()
        #shift
        self.shift = Shift(channels,k)
        self.sigma_color = 2*sigma_color**2

    def forward(self, I):
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = (Is-Iex)**2 #here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)
        Dd = De * self.g.data
        W_denom = torch.sum(Dd,dim=1)
        If = torch.sum(Dd*Is,dim=1) / W_denom
        return If
    


def erosion(img):
    kernel = torch.ones(3, 3).cuda()
    out = km.erosion(img, kernel)                   
    return out

def dilation(img):
    kernel = torch.ones(7, 7).cuda() 
    out = km.dilation(img, kernel)  
    return out      

