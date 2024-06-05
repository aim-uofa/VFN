from functools import partial
from typing import Optional, Any, Callable, List, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from unicore.modules import LayerNorm
from unicore.utils import tensor_tree_map
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            # self._trunc_normal_init(1.0)
            self._glorot_uniform_init()
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

# def vec_relu(vec):
#     below_xy_mask = ~(vec[..., -1]>=0)
#     vec[below_xy_mask] = 0
#     return vec
class GeoVecMLPMultiLayers(nn.Module):
    def __init__(self, num_vec_in, n = 2, num_layers = 2):
        super(GeoVecMLPMultiLayers, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.num_layers = num_layers
        self.layers = nn.Sequential(*[GeoVecMLP(num_vec_in, n) for _ in range(num_layers)])

    def forward(self, vec):
        vec = self.layers(vec)
        return vec

class GeoVecMLPMultiLayers_gvp(nn.Module):
    def __init__(self, num_vec_in, n = 2, num_layers = 2):
        super(GeoVecMLPMultiLayers_gvp, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.num_layers = num_layers
        self.layers = nn.Sequential(*[GeoVecMLP_gvp(num_vec_in, n) for _ in range(num_layers)])

    def forward(self, vec):
        vec = self.layers(vec)
        return vec
    
class GeoVecMLPMultiLayers_relu(nn.Module):
    def __init__(self, num_vec_in, n = 2, num_layers = 2):
        super(GeoVecMLPMultiLayers_relu, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.num_layers = num_layers
        self.layers = nn.Sequential(*[GeoVecMLP_relu(num_vec_in, n) for _ in range(num_layers)])

    def forward(self, vec):
        vec = self.layers(vec)
        return vec
class GeoVecMLP(nn.Module):
    def __init__(self, num_vec_in, n = 2, dropout = 0.1):
        super(GeoVecMLP, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.Linear1 = GeoVecLinear(num_vec_in, num_vec_in*n, bias=True) #TODO try to remove bias
        self.Linear2 = GeoVecLinear(num_vec_in*n, num_vec_in, bias=True) #TODO try to remove bias
        self.Linear2.Linear._zero_init(False)

        self.dir_vec = nn.Parameter(torch.randn(num_vec_in*n,3))

        self.dropout = dropout
        self.drop_n = int(num_vec_in*dropout)
        self.scale_factor = 1.0 / (1 - self.drop_n/num_vec_in)

    def forward(self, vec):
        vec_res = vec
        vec = self.Linear1(vec)
        
        cos_sim = torch.einsum("bnc,nc->bn", vec, self.dir_vec) / (torch.norm(vec, dim=-1) * torch.norm(self.dir_vec, dim=-1)+1e-8)
        dir_factor = (cos_sim + 1)/2
        vec = vec*dir_factor.unsqueeze(-1)
        
        vec = self.Linear2(vec)
        if self.training:
            dropmask = torch.rand(vec.shape[:2], device=vec.device)<=self.dropout
            vec[dropmask] = 0
            vec = vec * self.scale_factor
        vec = vec + vec_res
        return vec
class GeoVecMLP_relu(nn.Module):
    def __init__(self, num_vec_in, n = 2, dropout = 0.1):
        super(GeoVecMLP_relu, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.Linear1 = GeoVecLinear(num_vec_in, num_vec_in*n, bias=True) #TODO try to remove bias
        self.Linear2 = GeoVecLinear(num_vec_in*n, num_vec_in, bias=True) #TODO try to remove bias
        self.Linear2.Linear._zero_init(False)
        self.relu = nn.ReLU()
        # self.dir_vec = nn.Parameter(torch.randn(num_vec_in*n,3))

        self.dropout = dropout
        self.drop_n = int(num_vec_in*dropout)
        self.scale_factor = 1.0 / (1 - self.drop_n/num_vec_in)

    def forward(self, vec):
        vec_res = vec
        vec = self.Linear1(vec)
        
        # cos_sim = torch.einsum("bnc,nc->bn", vec, self.dir_vec) / (torch.norm(vec, dim=-1) * torch.norm(self.dir_vec, dim=-1)+1e-8)
        # dir_factor = (cos_sim + 1)/2
        # vec = vec*dir_factor.unsqueeze(-1)
        vec = vec.flatten(start_dim=1)
        vec = self.relu(vec)
        vec = vec.reshape(vec.shape[0], vec.shape[1]//3, 3)
        vec = self.Linear2(vec)
        if self.training:
            dropmask = torch.rand(vec.shape[:2], device=vec.device)<=self.dropout
            vec[dropmask] = 0
            vec = vec * self.scale_factor
        vec = vec + vec_res
        return vec
class GeoVecMLP_gvp(nn.Module):
    def __init__(self, num_vec_in, n = 2, dropout = 0.1):
        super(GeoVecMLP_gvp, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.Linear1 = GeoVecLinear(num_vec_in, num_vec_in*n, bias=True) #TODO try to remove bias
        self.Linear2 = GeoVecLinear(num_vec_in*n, num_vec_in, bias=True) #TODO try to remove bias
        self.Linear2.Linear._zero_init(False)
        self.sigmoid = nn.Sigmoid()
        # self.dir_vec = nn.Parameter(torch.randn(num_vec_in*n,3))

        self.dropout = dropout
        self.drop_n = int(num_vec_in*dropout)
        self.scale_factor = 1.0 / (1 - self.drop_n/num_vec_in)

    def forward(self, vec):
        vec_res = vec
        vec = self.Linear1(vec)
        
        # cos_sim = torch.einsum("bnc,nc->bn", vec, self.dir_vec) / (torch.norm(vec, dim=-1) * torch.norm(self.dir_vec, dim=-1)+1e-8)
        # dir_factor = (cos_sim + 1)/2
        # vec = vec*dir_factor.unsqueeze(-1)
        # vec = vec.flatten(start_dim=1)
        vec = vec* self.sigmoid(_norm_no_nan(vec, axis=-1, keepdims=True))
        #vec = vec.reshape(vec.shape[0], vec.shape[1]//3, 3)
        vec = self.Linear2(vec)
        if self.training:
            dropmask = torch.rand(vec.shape[:2], device=vec.device)<=self.dropout
            vec[dropmask] = 0
            vec = vec * self.scale_factor
        vec = vec + vec_res
        return vec
class SimpleMLP(nn.Module): 
    def __init__(self, num_vec_in, n = 2, dropout = 0.1):
        super(SimpleMLP, self).__init__()
        # 与GeoVecMLP有相同输入输出，但使用普通的MLP
        self.num_vec_in = num_vec_in
        self.n = n
        self.Linear1 = Linear(num_vec_in*3, num_vec_in*3*n, bias=True) #TODO try to remove bias
        self.Linear2 = Linear(num_vec_in*3*n, num_vec_in*3, bias=True) #TODO try to remove bias
        self.Linear2._zero_init(False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, vec):
        vec_res = vec
        # vec n x 32 x 3
        vec = vec.reshape(vec.shape[0], -1)
        vec = self.Linear1(vec)
        vec = self.activation(vec)
        vec = self.Linear2(vec)
        vec = self.dropout(vec)
        # reshape
        vec = vec.reshape(vec.shape[0], -1, 3)
        vec = vec + vec_res
        return vec
class SimpleMLPMultiLayers(nn.Module):
    def __init__(self, num_vec_in, n = 2, num_layers = 2):
        super(SimpleMLPMultiLayers, self).__init__()
        self.num_vec_in = num_vec_in
        self.n = n
        self.num_layers = num_layers
        self.layers = nn.Sequential(*[SimpleMLP(num_vec_in, n) for _ in range(num_layers)])

    def forward(self, vec):
        vec = self.layers(vec)
        return vec



class VecLinear(nn.Module):
    def __init__(self, num_vec_in, num_vec_out, type):
        super(VecLinear, self).__init__()
        if type == "vanilla":
            self.Linear = VanVecLinear(num_vec_in, num_vec_out)
        elif type=="geometric":
            self.Linear = GeoVecLinear(num_vec_in, num_vec_out)
        else:
            raise ValueError("Invalid method.")

    def forward(self, vec):
        vec = self.Linear(vec)
        return vec
class FakeLinear(nn.Module):
    def __init__(self, num_vec_in, num_vec_out, type):
        super(FakeLinear, self).__init__()
        self.num_vec_in = num_vec_in
        self.num_vec_out = num_vec_out
        self.type = type
        self.linear = Linear(num_vec_in*3, num_vec_out*3)

    def forward(self, vec):
        vec = vec.reshape(vec.shape[0], -1)
        vec = self.linear(vec)
        vec = vec.reshape(vec.shape[0], self.num_vec_out, 3)

        return vec
        
class GeoVecLinear(nn.Module):
    def __init__(self, num_vec_in, num_vec_out, bias = False):
        super(GeoVecLinear, self).__init__()
        self.num_vec_in = num_vec_in
        self.num_vec_out = num_vec_out
        self.bias_enable = bias
        if num_vec_in==num_vec_out:
            self.Linear = Linear(num_vec_in, num_vec_in, bias=False, init="final")
        else:
            self.Linear = Linear(num_vec_in, num_vec_out, bias=False)
        if self.bias_enable:
            self.bias = nn.Parameter(torch.zeros(num_vec_out,3))

    def forward(self, vec):
        if self.num_vec_in==self.num_vec_out:
            vec_update = self.Linear(vec.transpose(-2, -1)).transpose(-2, -1)
            if self.bias_enable:
                vec_update = vec_update + self.bias[None]
            vec = residual(vec, vec_update, self.training)
            return vec
        else:
            vec = self.Linear(vec.transpose(-2, -1)).transpose(-2, -1)
            if self.bias_enable:
                vec = vec + self.bias[None]
            return vec
        
class VanVecLinear(nn.Module):
    def __init__(self, num_vec_in, num_vec_out):
        super(VanVecLinear, self).__init__()
        self.num_vec_in = num_vec_in
        self.num_vec_out = num_vec_out
        if num_vec_in==num_vec_out:
            self.Linear = Linear(num_vec_in*3, num_vec_in*3, bias=False, init="final")
        else:
            self.Linear = Linear(num_vec_in*3, num_vec_out*3, bias=False)
    def forward(self, vec):
        if self.num_vec_in==self.num_vec_out:
            vec_update = self.Linear(vec.view(-1, self.num_vec_in*3))
            vec_update = vec_update.view(-1, self.num_vec_out, 3)
            vec = residual(vec, vec_update, self.training)
            return vec
        else:
            vec = vec.view(-1, self.num_vec_in*3)
            vec = self.Linear(vec)
            vec = vec.view(-1, self.num_vec_out, 3)
            return vec
    

class Transition(nn.Module):
    def __init__(self, d_in, n):

        super(Transition, self).__init__()

        self.d_in = d_in
        self.n = n

        self.layer_norm = LayerNorm(self.d_in)
        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")

    def _transition(self, x):
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"x": x},
            chunk_size=chunk_size,
            num_batch_dims=len(x.shape[:-2]),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:

        if chunk_size is not None:
            x = self._chunk(x, chunk_size)
        else:
            x = self._transition(x=x)

        return x


class OuterProductMean(nn.Module):
    def __init__(self, d_msa, d_pair, d_hid, eps=1e-3):
        super(OuterProductMean, self).__init__()

        self.d_msa = d_msa
        self.d_pair = d_pair
        self.d_hid = d_hid
        self.eps = eps

        self.layer_norm = LayerNorm(d_msa)
        self.linear_1 = Linear(d_msa, d_hid)
        self.linear_2 = Linear(d_msa, d_hid)
        self.linear_out = Linear(d_hid**2, d_pair, init="relu")
        self.act = nn.GELU()
        self.linear_z = Linear(self.d_pair, self.d_pair, init="final")
        self.layer_norm_out = LayerNorm(self.d_pair)

    def _opm(self, a, b):
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        a = a.reshape((-1,) + a.shape[-3:])
        b = b.reshape((-1,) + b.shape[-3:])
        out = []
        # TODO: optimize this
        for a_prime, b_prime in zip(a, b):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                num_batch_dims=1,
            )
            out.append(outer)
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def apply_alphafold_original_mode(self):
        self.linear_z = None
        self.layer_norm_out = None

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:

        m = self.layer_norm(m)
        mask = mask.unsqueeze(-1)
        if self.layer_norm_out is not None:
            # for numerical stability
            mask = mask * (mask.size(-2) ** -0.5)
        a = self.linear_1(m)
        b = self.linear_2(m)
        if self.training:
            a = a * mask
            b = b * mask
        else:
            a *= mask
            b *= mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            z = self._chunk(a, b, chunk_size)
        else:
            z = self._opm(a, b)

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        z /= self.eps + norm
        if self.layer_norm_out is not None:
            z = self.act(z)
            z = self.layer_norm_out(z)
            z = self.linear_z(z)
        return z

pi = 3.14159
a = (2 * pi) ** 0.5

# @torch.jit.script
def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2))# / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, num_hidden, num_vec, K=4):
        super().__init__()
        # nn.init.uniform_(self.means.weight, 0, 3)
        # nn.init.uniform_(self.stds.weight, 0, 3)
        # nn.init.constant_(self.bias.weight, 0)
        # nn.init.constant_(self.mul.weight, 1)
        self.mean = Linear(num_hidden*2, num_vec*K, init="final")
        self.var = Linear(num_hidden*2, num_vec*K, init="final")
        self.anchor_means = nn.Parameter(torch.linspace(0, 20, K)[None].repeat(num_vec,1), requires_grad=False)
        self.anchor_var = nn.Parameter(torch.ones(num_vec, K) * (20 / (K/2)), requires_grad=False)
        

    def forward(self, x, h_src, h_dst):
        nodes = torch.cat([h_src, h_dst], dim=-1)
        mean = self.mean(nodes)
        var = self.var(nodes)

        mean = mean.view(mean.shape[0], -1, 4)
        var = var.view(mean.shape[0], -1, 4)

        mean = (mean + self.anchor_means[None]).abs()
        var = (var + self.anchor_var[None]).abs() + 0.5

        return gaussian(x[...,None], mean, var).flatten(1,2)

class GaussianNondeLayer(nn.Module):
    def __init__(self, num_vec, K=4):
        super().__init__()
        
        self.anchor_means = nn.Parameter(torch.linspace(0, 20, K)[None].repeat(num_vec,1), requires_grad=False)
        self.anchor_var = nn.Parameter(torch.ones(num_vec, K) * (20 / (K/2)), requires_grad=False)
        

    def forward(self, x):
        mean = (self.anchor_means[None]).abs()
        var = (self.anchor_var[None]).abs() + 0.5

        return gaussian(x[...,None], mean, var).flatten(1,2)

def residual(residual, x, training):
    if training:
        return x + residual
    else:
        residual += x
        return residual

@torch.jit.script
def vec_to_tensor(vec: torch.Tensor,) -> torch.Tensor:
    return vec.flatten(1)

@torch.jit.script
def tensor_to_vec(tensor: torch.Tensor,) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1, 3)

@torch.jit.script
def fused_bias_dropout_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    dropmask: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return (x + bias) * F.dropout(dropmask, p=prob, training=True) + residual


@torch.jit.script
def fused_bias_dropout_add_inference(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    residual += bias + x
    return residual


def bias_dropout_residual(module, residual, x, dropout_shared_dim, prob, training):
    bias = module.get_output_bias()
    if training:
        shape = list(x.shape)
        shape[dropout_shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return fused_bias_dropout_add(x, bias, residual, mask, prob)
    else:
        return fused_bias_dropout_add_inference(x, bias, residual)


@torch.jit.script
def fused_bias_gated_dropout_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    g_bias: torch.Tensor,
    residual: torch.Tensor,
    dropout_mask: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return (torch.sigmoid(g + g_bias) * (x + bias)) * F.dropout(
        dropout_mask,
        p=prob,
        training=True,
    ) + residual

def init_vec_with_true_atom(h_V_vector, V_xyz):
            #  h_V_vector: (batch_size, 32, 3)
            #  V_xyz: (batch_size, 3, 3)
            #  return: (batch_size, 32, 3)
            #  32 x 3 -> 4x8x3
            seq_len, vec_num  = h_V_vector.shape[0], h_V_vector.shape[1]
            h_V_vector = h_V_vector.view(-1, 4, vec_num//4, 3)
            # V_xyz: (batch_size, 3, 3) -> batch_size x 4 x 3, padding 0 , Calpha (0,0,0)
            V_xyz = torch.cat([torch.zeros(V_xyz.shape[0], 1, 3).to(V_xyz.device), V_xyz], dim=1)
            h_V_vector[:,:,0,:] = V_xyz
            # 4x8x3 -> 32x3
            h_V_vector = h_V_vector.view(seq_len, vec_num, 3)
            return h_V_vector

def tri_mul_residual(
    module,
    residual,
    outputs,
    dropout_shared_dim,
    prob,
    training,
    block_size,
):
    if training:
        x, g = outputs
        bias, g_bias = module.get_output_bias()
        shape = list(x.shape)
        shape[dropout_shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return fused_bias_gated_dropout_add(
            x,
            bias,
            g,
            g_bias,
            residual,
            mask,
            prob,
        )
    elif block_size is None:
        x, g = outputs
        bias, g_bias = module.get_output_bias()
        residual += (torch.sigmoid(g + g_bias) * (x + bias))
        return residual
    else:
        # gated is not used here
        residual += outputs
        return residual


class SimpleModuleList(nn.ModuleList):
    def __repr__(self):
        return str(len(self)) + " X ...\n" + self[0].__repr__()


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    num_batch_dims: int,
) -> Any:
    # TODO: support inplace add to output
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    def _dict_get_shapes(input):
        shapes = []
        if type(input) is torch.Tensor:
            shapes.append(input.shape)
        elif type(input) is dict:
            for v in input.values():
                shapes.extend(_dict_get_shapes(v))
        elif isinstance(input, Iterable):
            for v in input:
                shapes.extend(_dict_get_shapes(v))
        else:
            raise ValueError("Not supported")

        return shapes

    inputs = {k: v for k, v in inputs.items() if v is not None}
    initial_dims = [shape[:num_batch_dims] for shape in _dict_get_shapes(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d
    num_chunks = (flat_batch_dim + chunk_size - 1) // chunk_size

    def _flat_inputs(t):
        t = t.view(-1, *t.shape[num_batch_dims:])
        assert (
            t.shape[0] == flat_batch_dim or t.shape[0] == 1
        ), "batch dimension must be 1 or equal to the flat batch dimension"
        return t

    flat_inputs = tensor_tree_map(_flat_inputs, inputs)

    out = None
    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, flat_batch_dim)

        def select_chunk(t):
            if t.shape[0] == 1:
                return t[0:1]
            else:
                return t[chunk_start:chunk_end]

        chunkes = tensor_tree_map(select_chunk, flat_inputs)

        output_chunk = layer(**chunkes)

        if out is None:
            out = tensor_tree_map(
                lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output_chunk
            )

        out_type = type(output_chunk)
        if out_type is tuple:
            for x, y in zip(out, output_chunk):
                x[chunk_start:chunk_end] = y
        elif out_type is torch.Tensor:
            out[chunk_start:chunk_end] = output_chunk
        else:
            raise ValueError("Not supported")

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out
