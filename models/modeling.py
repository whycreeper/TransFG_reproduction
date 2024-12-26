# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import json

from os.path import join as pjoin

import jittor
from jittor import nn
from jittor import Module

import numpy as np

from jittor.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from jittor.modules.utils import _pair
from jittor.misc import _pair
from scipy import ndimage

import models.configs as configs

logger = logging.getLogger(__name__)

log_file = "/home/aiuser/TransFG/logs/jittor.jsonl"
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return jittor.array(weights)

def swish(x):
    return x * jittor.sigmoid(x)

ACT2FN = {"gelu": jittor.nn.gelu, "relu": jittor.nn.relu, "swish": swish}

import matplotlib.pyplot as plt
def selected_patches(x, part_inx, save_selected_path):  # [4,3,448,448,] / [4,12,]
    B, C, H, W = x.shape
    p_h, p_w = (16, 16)
    n_h, n_w = H // p_h, W // p_w

    for b in range(B):
        img = x[b].permute(1, 2, 0).detach().cpu().numpy()
        selected_indices = part_inx[b].detach().cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(img / img.max())
        plt.axis("off")

        for idx in selected_indices:
            idx = idx - 1
            row = idx // n_w
            col = idx % n_w
            
            y1, y2 = row * p_h, (row + 1) * p_h
            x1, x2 = col * p_w, (col + 1) * p_w
            plt.gca().add_patch(plt.Rectangle((x1, y1), p_w, p_h, edgecolor='red', facecolor='none', lw=2))

        plt.title(f"Image {b+1}: Selected Patches")
        plt.savefig(save_selected_path, bbox_inches='tight')
        plt.close()
    return

class LabelSmoothing(Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def execute(self, x, target):
        logprobs = jittor.nn.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def execute(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = jittor.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = jittor.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.gauss_(self.fc1.bias, std=1e-6)
        nn.init.gauss_(self.fc2.bias, std=1e-6)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(jittor.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(jittor.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def execute(self, x):   # [4,3,448,448,]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [4,1,768,]

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # [4,768,28,28,]
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = jittor.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)   # [4,785,768,] 785 = 28*28+1
        return embeddings

class Block(Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def execute(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with jittor.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight=jittor.copy(query_weight)
            self.attn.key.weight=jittor.copy(key_weight)
            self.attn.value.weight=jittor.copy(value_weight)
            self.attn.out.weight=jittor.copy(out_weight)
            self.attn.query.bias=jittor.copy(query_bias)
            self.attn.key.bias=jittor.copy(key_bias)
            self.attn.value.bias=jittor.copy(value_bias)
            self.attn.out.bias=jittor.copy(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")])
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")])

            self.ffn.fc1.weight=jittor.copy(mlp_weight_0)
            self.ffn.fc2.weight=jittor.copy(mlp_weight_1)
            self.ffn.fc1.bias=jittor.copy(mlp_bias_0)
            self.ffn.fc2.bias=jittor.copy(mlp_bias_1)

            self.attention_norm.weight=jittor.copy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias=jittor.copy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight=jittor.copy(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias=jittor.copy(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Part_Attention(Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def execute(self, x):# 11, 4, 12, 1370, 1370
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = jittor.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]# 4, 12, 1369
        
        _, max_inx = last_map.argmax(2)
        
        return max_inx, _ # 4, 12

class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = Block(config)
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def execute(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)            
        part_num, part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(hidden_states[i, part_inx[i,:]])
        parts = jittor.stack(parts)
        try:
            parts = parts.squeeze(1)
        except:
            None
        concat = jittor.cat((hidden_states[:,0].unsqueeze(1), parts), dim=1)# 1,13,768

        part_states, part_weights = self.part_layer(concat)# 1,13,768
        
        part_encoded = self.part_norm(part_states)   

        return part_encoded, part_inx
    

class EncoderVIT(Module):
    def __init__(self, config):
        super(EncoderVIT, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def execute(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def execute(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded, part_inx = self.encoder(embedding_output)
        return part_encoded, part_inx
    
class TransformerVIT(Module):
    def __init__(self, config, img_size):
        super(TransformerVIT, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = EncoderVIT(config)

    def execute(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, None

class VisionTransformer(Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False, vit=0):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size) if vit == 0 else TransformerVIT(config, img_size)
        self.part_head = Linear(config.hidden_size, num_classes)
        self.vit = vit

    def execute(self, x, labels=None, save_selected_path=None, args=None): # [4,3,448,448,]
        part_tokens, part_inx = self.transformer(x) # [4,785,768,] -> [4,13,768,]
        part_logits = self.part_head(part_tokens[:, 0])

        if save_selected_path is not None:
            selected_patches(x, part_inx, save_selected_path)

        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            contrast_loss = con_loss(part_tokens[:, 0], labels.view(-1), args.alpha)
            if args.contrastive_loss:
                loss = part_loss + contrast_loss
            else:
                loss = part_loss
            return loss, part_logits
        else:
            return part_logits

    def load_from(self, weights):
        with jittor.no_grad():
            self.transformer.embeddings.patch_embeddings.weight=jittor.copy(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias=jittor.copy(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token=jittor.copy(np2th(weights["cls"]))
            if self.vit == 0:
                self.transformer.encoder.part_norm.weight=jittor.copy(np2th(weights["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.part_norm.bias=jittor.copy(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings=jittor.copy(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings=jittor.copy(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight=jittor.copy(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight=jittor.copy(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias=jittor.copy(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 

def con_loss(features, labels, alpha):
    B, _ = features.shape
    features = jittor.normalize(features)
    cos_matrix = features.matmul(features.t())
    pos_label_matrix = jittor.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - alpha
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
