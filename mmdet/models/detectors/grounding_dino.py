# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from ..layers.transformer.SRBranch import SRBranch
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
import torch.nn.functional as F
# from llava_llama_descriptor import create_llama_image_descriptor
from einops import rearrange
def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


import traceback
from collections import defaultdict

hook_counts = defaultdict(int)

def make_hook(name):
    def hook(grad):
        hook_counts[name] += 1
        if hook_counts[name] > 1:
            print(f"[DEBUG] `{name}` hook called {hook_counts[name]} times. Stack:")
            traceback.print_stack()
        return grad
    return hook


from ..layers.transformer.grounding_dino_layers import Text_IF_Fusion
@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 fusion_module,
                 language_model,
                 use_sr_branch,
                 sr_branch_cfg,
                 *args,
                 prompt_template: Optional[str] = None,
                 use_autocast=False,
                 **kwargs) -> None:

        self.fusion_module_cfg = fusion_module
        self.language_model_cfg = language_model
        self.use_sr_branch = use_sr_branch
        self.sr_branch_cfg = sr_branch_cfg
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        # self.llm_cfg = llm_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        #fuser
        self.vision_fuser = None    
        if self.fusion_module_cfg['use_fusion']:
            if self.fusion_module_cfg['type'] == 'tif':
                self.vision_fuser = Text_IF_Fusion()
        # SR Branch
        if self.use_sr_branch:
            default_sr_cfg = dict(
                type='SRBranch',
                in_channels=512,
                out_channels=3,
                factor=2,
                num_layers=3,
                intermediate_channels=64
            )


            sr_cfg = {**default_sr_cfg, **self.sr_branch_cfg}
            self.sr_branch = MODELS.build(sr_cfg)
            sr_feat_channels = sr_cfg.get('intermediate_channels', 64)
            self.sr_feat_adapters = nn.ModuleList([
                nn.Conv2d(sr_feat_channels, self.embed_dims, kernel_size=1)
                for _ in range(self.num_feature_levels)
            ])
            self.sr_feat_scales = nn.Parameter(
                torch.full((self.num_feature_levels,), 0.1))

            self.sr_low_level_idx = self.sr_branch_cfg.pop('low_level_idx', 0)
            self.sr_high_level_idx = self.sr_branch_cfg.pop('high_level_idx', 2)
            self.sr_loss_weight = self.sr_branch_cfg.pop('loss_weight', 0.1)

        # self.llm = create_llama_image_descriptor()
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
        if self.use_sr_branch:
            for adapter in self.sr_feat_adapters:
                nn.init.xavier_uniform_(adapter.weight)
                if adapter.bias is not None:
                    nn.init.constant_(adapter.bias, 0)
        # if self.vision_fuser is not None:
        #     self.vision_fuser.init_weights()

    def _inject_sr_features(self, mlvl_feats: List[Tensor], sr_feat: Tensor) -> List[Tensor]:
        enhanced_feats = []
        num_levels = min(len(mlvl_feats), len(self.sr_feat_adapters))
        for lvl, feat in enumerate(mlvl_feats):
            if lvl < num_levels:
                sr_proj = self.sr_feat_adapters[lvl](sr_feat)
                if sr_proj.shape[2:] != feat.shape[2:]:
                    sr_proj = F.interpolate(
                        sr_proj,
                        size=feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                scale = self.sr_feat_scales[lvl].view(1, 1, 1, 1)
                enhanced_feats.append(feat + scale * sr_proj)
            else:
                enhanced_feats.append(feat)
        return enhanced_feats

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))

                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)
                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked
    
    def forward_transformer(
        self,
        img_feats: Union[Tensor, Tuple[Tensor]],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        if self.fusion_module_cfg['use_fusion']:
            vi_feats, ii_feats = img_feats
                
            if self.fusion_module_cfg['type']=='mae':
                
                vif_dict, _ = self.pre_transformer(vi_feats, batch_data_samples=None)
                iif_dict, _ = self.pre_transformer(ii_feats, batch_data_samples=None)
                src = []
                for v, i in zip(vi_feats, ii_feats):
                    feat_fuse = self.vision_fuser(v, i, vif_dict['feat_pos'], iif_dict['feat_pos'])
                    src.append(feat_fuse)
                encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(src, batch_data_samples)
                encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict)


            elif self.fusion_module_cfg['type']=='tif':
            
                src = []
                text_feat = text_dict['embedded'].mean(1)
                for v, i in zip(vi_feats, ii_feats):
                    feat_fuse = self.vision_fuser(v, i, text_feat)
                    src.append(feat_fuse)
                encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(src, batch_data_samples)
                encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict)

            elif self.fusion_module_cfg['type']=='add':
                feat_fuse = [(v+i)/2 for v, i in zip(vi_feats, ii_feats)]
                encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(feat_fuse, batch_data_samples)
                encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict)

        else:
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples=None)
            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict)
            

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        if batch_data_samples is not None:
            batch_input_shape = batch_data_samples[0].batch_input_shape
            input_img_h, input_img_w = batch_input_shape
            img_shape_list = [sample.img_shape for sample in batch_data_samples]
            same_shape_flag = all([
                s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
            ])
        else:
            same_shape_flag = True
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict,concated_feats:bool=False) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'],
            #concated feats
            concated_feats=concated_feats)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict


    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        #bs, _, 4
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,# bs, num_queries, 4
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta,
            det_features=memory,
            spatial_shapes=spatial_shapes,       # 新增：用于还原 4D
            level_start_index=level_start_index) if self.training else dict()  # 新增：用于还原 4Dif self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict


    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        if self.bbox_head.reg_branches[0][-1].out_features == 4:
            # GroundingDINO Head
            return super().forward_decoder(
                query=query,
                memory=memory,
                memory_mask=memory_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                dn_mask=dn_mask,
                **kwargs)
        else:
            # inter_states, references, pred_corners = self.decoder(
            inter_states, references = self.decoder(
                query=query,
                value=memory,
                key_padding_mask=memory_mask,
                self_attn_mask=dn_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reg_branches=self.bbox_head.reg_branches,
                integral=self.bbox_head.integral,
                **kwargs)

            if len(query) == self.num_queries:
                # NOTE: This is to make sure label_embeding can be involved to
                # produce loss even if there is no denoising query (no ground truth
                # target in this GPU), otherwise, this will raise runtime error in
                # distributed training.
                inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

            decoder_outputs_dict = dict(
                hidden_states=inter_states, references=list(references))
                # hidden_states=inter_states, references=list(references), pred_corners=pred_corners)
            return decoder_outputs_dict

    def loss_head_inputs_dict(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]


        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        # fusion_text_dict = self.language_model(fusion_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
            # fusion_text_dict['embedded'] = self.text_feat_map(fusion_text_dict['embedded'])
        
        template_text_dict = None
        
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        sr_feat = None
        if self.use_sr_branch and self.training and visual_features is not None:
            if isinstance(visual_features, tuple):
                features = visual_features[0]
            else:
                features = visual_features
            num_levels = len(features)
            low_idx = min(self.sr_low_level_idx, num_levels - 1)
            high_idx = min(self.sr_high_level_idx, num_levels - 1)
            sr_low_level_feat = features[low_idx]
            sr_high_level_feat = features[high_idx]

            _sr_output, sr_feat = self.sr_branch(sr_low_level_feat, sr_high_level_feat)
        else:
            _sr_output = None
            
        # # 如果是6通道输入
        # if batch_inputs.shape[1] == 6: 
        #     self._original_input = batch_inputs[:, :3, :, :]  #取前3通道(RGB)
        # else:
        #     self._original_input = batch_inputs
        _original_input = batch_inputs   

        if self.use_sr_branch and sr_feat is not None:
            if isinstance(visual_features, tuple):
                vi_feats, ii_feats = visual_features
                vi_feats = self._inject_sr_features(list(vi_feats), sr_feat)
                ii_feats = self._inject_sr_features(list(ii_feats), sr_feat)
                visual_features = (vi_feats, ii_feats)
            else:
                visual_features = self._inject_sr_features(list(visual_features), sr_feat)

        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                batch_data_samples)

        if _sr_output is not None:
            # head_inputs_dict['sr_output'] = _sr_output
            # head_inputs_dict['original_input'] = _original_input
            sr_output = _sr_output
            original_input = _original_input
        else:
            sr_output = None
            original_input = None
        return head_inputs_dict, sr_output, original_input
    


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        head_inputs_dict, _sr_output, _original_input = self.loss_head_inputs_dict(batch_inputs, batch_data_samples)

        enc_outputs_class = head_inputs_dict.get('enc_outputs_class', None)
        enc_outputs_coord = head_inputs_dict.get('enc_outputs_coord', None)
        
        det_features = head_inputs_dict.pop('det_features', None)
        spatial_shapes = head_inputs_dict.pop('spatial_shapes', None)
        level_start_index = head_inputs_dict.pop('level_start_index', None)
        
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # if hasattr(self.vision_fuser, 'get_logging_stats'):
        #     fusion_stats = self.vision_fuser.get_logging_stats()
        #     for stat_name, stat_value in fusion_stats.items():
        #         losses[stat_name] = stat_value

        # _original_input = head_inputs_dict['original_input']
        # _sr_output = head_inputs_dict['sr_output']
        if (self.use_sr_branch and self.training and 
                        hasattr(self, 'sr_branch') and 
                        hasattr(self.sr_branch, 'distillation_loss')):
            distill_losses = self.sr_branch.distillation_loss(
                det_features,
                spatial_shapes,
                level_start_index,
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord=enc_outputs_coord)
            for loss_name, loss_value in distill_losses.items():
                losses[loss_name] = loss_value
        if (self.use_sr_branch and self.training and _sr_output is not None and
                                                    _original_input is not None):
        
            # Prepare target: interpolate original input to SR scale
            target = F.interpolate(
                _original_input, 
                scale_factor=self.sr_branch.factor,
                mode='bilinear', 
                align_corners=False
            )
            # final_sr_output = _sr_output.repeat(target.shape[0], 1, 1, 1)

            sr_loss = self.sr_branch.compute_loss(_sr_output, target)
        
            losses['loss_sr'] = sr_loss * self.sr_loss_weight


        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        
        original_use_sr = self.use_sr_branch
        self.use_sr_branch = False  # Disable SR branch during inference
        
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        fusion_prompts = []

        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])
            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

        return batch_data_samples

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        visual_feats = None
    
        if self.fusion_module_cfg['use_fusion']:
            res = []
            for _batch_input in batch_inputs: # [bs, 6, h, w]
                c = _batch_input.shape[0] // 2
                res.append(_batch_input[:c, ...])
                res.append(_batch_input[c:, ...])
            # vi0,ii0, vi1, ii1, ..., viN, iiN
            batch_inputs = torch.stack(res, dim=0)

            # image feature extraction
            visual_feats = super().extract_feat(batch_inputs)
    
            _visual_feats = [[], []]
            for i in range(len(visual_feats)):# [2*bs,c0,h0,w0], ... ,[2*bs,c1,h1,w1]
                _visual_feats[0].append(visual_feats[i][0::2, ...])
                _visual_feats[1].append(visual_feats[i][1::2, ...])
        
            visual_feats_to_return = (_visual_feats[0], _visual_feats[1])
        else:
            # image feature extraction
            visual_feats = super().extract_feat(batch_inputs)
            visual_feats_to_return = visual_feats
    
        return visual_feats_to_return

