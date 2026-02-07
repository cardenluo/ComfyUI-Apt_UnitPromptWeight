import torch
import re
import torch.nn.functional as F
from nodes import CLIPTextEncode

class pre_Unit_PromptWeight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "main_prompt_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "process"

    def __init__(self):
        pass

    def get_pooled_from_cond(self, cond_feature):
        if cond_feature is None:
            return torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        device = cond_feature.device
        dtype = cond_feature.dtype
        embed_dim = int(cond_feature.shape[-1])
        pooled = torch.mean(cond_feature, dim=0)
        pooled = F.layer_norm(pooled, normalized_shape=[embed_dim])
        return pooled

    def clean_text(self, text):
        clean_chars = ',，.。 \t\n'
        cleaned = text.strip(clean_chars)
        return cleaned

    def parse_with_auto_pack(self, prompt):
        pattern = re.compile(r'\[([^\]]+)@([0-9.]+)\]')
        matches = list(pattern.finditer(prompt))
        segments = []
        weights = []
        all_text_parts = []
        last_end = 0
        for match in matches:
            start = match.start()
            end = match.end()
            prefix_text = prompt[last_end:start]
            cleaned_prefix = self.clean_text(prefix_text)
            if cleaned_prefix:
                segments.append(cleaned_prefix)
                weights.append(1.0)
                all_text_parts.append(cleaned_prefix)
            anchor_content = match.group(1).strip()
            anchor_weight = float(match.group(2)) if match.group(2).replace('.','').isdigit() else 1.0
            segments.append(anchor_content)
            weights.append(anchor_weight)
            all_text_parts.append(anchor_content)
            last_end = end
        suffix_text = prompt[last_end:]
        cleaned_suffix = self.clean_text(suffix_text)
        if cleaned_suffix:
            segments.append(cleaned_suffix)
            weights.append(1.0)
            all_text_parts.append(cleaned_suffix)
        whole_content = '，'.join(all_text_parts)
        return segments, weights, whole_content

    def fuse_conditions_equal(self, cond_features, cond_strengths):
        if len(cond_features) == 0 or len(cond_strengths) == 0:
            return None, None, None
        device = cond_features[0].device if cond_features[0] is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = cond_features[0].dtype if cond_features[0] is not None else torch.float16
        total_w = sum(cond_strengths)
        norm_w = [s / total_w if total_w > 1e-6 else 1.0/len(cond_strengths) for s in cond_strengths]
        max_len = max([cf.shape[0] for cf in cond_features if cf is not None], default=1)
        padded_feats = []
        for cf in cond_features:
            if cf is None:
                cf = torch.zeros((max_len, 768), dtype=dtype, device=device)
            pad_len = max_len - cf.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, cf.shape[-1]), device=device, dtype=dtype)
                padded_feat = torch.cat([cf, pad], dim=0)
            else:
                padded_feat = cf
            padded_feats.append(padded_feat)
        fused_feat = torch.zeros_like(padded_feats[0], device=device, dtype=dtype)
        for feat, w in zip(padded_feats, norm_w):
            fused_feat += feat * w
        pooled_list = [self.get_pooled_from_cond(cf) for cf in cond_features]
        fused_pooled = torch.zeros_like(pooled_list[0], device=device, dtype=dtype)
        for pl, w in zip(pooled_list, norm_w):
            fused_pooled += pl * w
        return fused_feat.unsqueeze(0), fused_pooled, norm_w

    def process(self, clip, prompt, main_prompt_ratio=0.5):
        prompt_info = ""
        segments, weights, whole_content = self.parse_with_auto_pack(prompt)
        
        has_custom_weight = any(w != 1.0 for w in weights)
        if not has_custom_weight and len(segments) > 0:
            main_prompt_ratio = 1.0
        
        if len(segments) == 0 or not whole_content:
            tok_empty = clip.tokenize("")
            c_empty = torch.zeros((1, 768), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            p_empty = torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            prompt_info = "Weight：main_prompt()0%|No valid individuals"
            return ([[c_empty, {"pooled_output": p_empty}]], prompt_info)
        
        try:
            tokens_whole = clip.tokenize(whole_content)
            c_whole, p_whole = clip.encode_from_tokens(tokens_whole, return_pooled=True)
            feat_whole = c_whole.squeeze(0) if c_whole is not None else torch.zeros((77, 768), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            pooled_whole = p_whole if p_whole is not None else self.get_pooled_from_cond(feat_whole)
        except:
            feat_whole = torch.zeros((77, 768), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            pooled_whole = torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        
        cond_features = []
        for seg in segments:
            try:
                tokens = clip.tokenize(seg)
                c_out, p_out = clip.encode_from_tokens(tokens, return_pooled=True)
                cond_feat = c_out.squeeze(0) if c_out is not None else None
            except:
                cond_feat = None
            cond_features.append(cond_feat)

        fused_individual, pooled_individual, norm_w = self.fuse_conditions_equal(cond_features, weights)
        if fused_individual is None:
            fused_individual = torch.zeros_like(feat_whole.unsqueeze(0))
            pooled_individual = torch.zeros_like(pooled_whole)
        
        w_overall = main_prompt_ratio
        w_individual_total = 1.0 - w_overall
        
        fused_feat = feat_whole.unsqueeze(0) * w_overall + fused_individual * w_individual_total
        fused_pooled = pooled_whole * w_overall + pooled_individual * w_individual_total

        final_parts = []
        main_pct = int(round(w_overall * 100))
        final_parts.append(f"main({whole_content}){main_pct}%")
        
        part_pcts = []
        for seg, nw in zip(segments, norm_w):
            pct = int(round(nw * w_individual_total * 100))
            part_pcts.append(pct)
            final_parts.append(f"{seg}{pct}%")
        
        total_pct = main_pct + sum(part_pcts)
        if total_pct != 100 and len(part_pcts) > 0:
            diff = 100 - total_pct
            part_pcts[0] += diff
            final_parts = [f"main_prompt({whole_content}){main_pct}%"]
            for i, seg in enumerate(segments):
                final_parts.append(f"{seg}{part_pcts[i]}%")
        
        prompt_info = "Weight：" + "|".join(final_parts)
        
        return ([[fused_feat, {"pooled_output": fused_pooled}]], prompt_info)


class pre_qwenimage_PromptWeight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "main_prompt_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "process"


    def __init__(self):
        pass

    def get_pooled_from_cond(self, cond_feature):
        if cond_feature is None:
            return torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        device = cond_feature.device
        dtype = cond_feature.dtype
        embed_dim = int(cond_feature.shape[-1])
        pooled = torch.mean(cond_feature, dim=0)
        pooled = F.layer_norm(pooled, normalized_shape=[embed_dim])
        return pooled

    def clean_text(self, text):
        clean_chars = ',，.。 \t\n'
        cleaned = text.strip(clean_chars)
        return cleaned

    def parse_with_auto_pack(self, prompt):
        pattern = re.compile(r'\[([^\]]+)@([0-9.]+)\]')
        matches = list(pattern.finditer(prompt))
        segments = []
        weights = []
        all_text_parts = []
        last_end = 0
        for match in matches:
            start = match.start()
            end = match.end()
            prefix_text = prompt[last_end:start]
            cleaned_prefix = self.clean_text(prefix_text)
            if cleaned_prefix:
                segments.append(cleaned_prefix)
                weights.append(1.0)
                all_text_parts.append(cleaned_prefix)
            anchor_content = match.group(1).strip()
            anchor_weight = float(match.group(2)) if match.group(2).replace('.','').isdigit() else 1.0
            segments.append(anchor_content)
            weights.append(anchor_weight)
            all_text_parts.append(anchor_content)
            last_end = end
        suffix_text = prompt[last_end:]
        cleaned_suffix = self.clean_text(suffix_text)
        if cleaned_suffix:
            segments.append(cleaned_suffix)
            weights.append(1.0)
            all_text_parts.append(cleaned_suffix)
        whole_content = '，'.join(all_text_parts)
        return segments, weights, whole_content

    def fuse_conditions_equal(self, cond_features, cond_strengths):
        if len(cond_features) == 0 or len(cond_strengths) == 0:
            return None, None, None
        device = cond_features[0].device if cond_features[0] is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = cond_features[0].dtype if cond_features[0] is not None else torch.float16
        total_w = sum(cond_strengths)
        norm_w = [s / total_w if total_w > 1e-6 else 1.0/len(cond_strengths) for s in cond_strengths]
        max_len = max([cf.shape[0] for cf in cond_features if cf is not None], default=1)
        padded_feats = []
        for cf in cond_features:
            if cf is None:
                cf = torch.zeros((max_len, 768), dtype=dtype, device=device)
            pad_len = max_len - cf.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, cf.shape[-1]), device=device, dtype=dtype)
                padded_feat = torch.cat([cf, pad], dim=0)
            else:
                padded_feat = cf
            padded_feats.append(padded_feat)
        fused_feat = torch.zeros_like(padded_feats[0], device=device, dtype=dtype)
        for feat, w in zip(padded_feats, norm_w):
            fused_feat += feat * w
        pooled_list = [self.get_pooled_from_cond(cf) for cf in cond_features]
        fused_pooled = torch.zeros_like(pooled_list[0], device=device, dtype=dtype)
        for pl, w in zip(pooled_list, norm_w):
            fused_pooled += pl * w
        return fused_feat.unsqueeze(0), fused_pooled, norm_w

    def align_features(self, feat1, feat2):
        len1 = feat1.shape[-2] if feat1.dim() >= 2 else 0
        len2 = feat2.shape[-2] if feat2.dim() >= 2 else 0
        
        if len1 == len2:
            return feat1, feat2
        
        target_len = max(len1, len2)
        device = feat1.device
        dtype = feat1.dtype
        
        if feat1.dim() >= 2:
            if len1 < target_len:
                pad_size = target_len - len1
                pad_shape = list(feat1.shape)
                pad_shape[-2] = pad_size
                pad = torch.zeros(pad_shape, device=device, dtype=dtype)
                feat1_aligned = torch.cat([feat1, pad], dim=-2)
            else:
                feat1_aligned = feat1[..., :target_len, :]
        else:
            feat1_aligned = feat1
        
        if feat2.dim() >= 2:
            if len2 < target_len:
                pad_size = target_len - len2
                pad_shape = list(feat2.shape)
                pad_shape[-2] = pad_size
                pad = torch.zeros(pad_shape, device=device, dtype=dtype)
                feat2_aligned = torch.cat([feat2, pad], dim=-2)
            else:
                feat2_aligned = feat2[..., :target_len, :]
        else:
            feat2_aligned = feat2
        
        return feat1_aligned, feat2_aligned

    def process(self, clip, prompt, main_prompt_ratio=0.5):
        prompt_info = ""
        segments, weights, whole_content = self.parse_with_auto_pack(prompt)
        
        has_custom_weight = any(w != 1.0 for w in weights)
        if not has_custom_weight and len(segments) > 0:
            main_prompt_ratio = 1.0
        
        if len(segments) == 0 or not whole_content:
            tok_empty = clip.tokenize("")
            c_empty = torch.zeros((1, 768), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            p_empty = torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            prompt_info = "Weight：main_prompt()0%|No valid individuals"
            return ([[c_empty, {"pooled_output": p_empty}]], prompt_info)
        
        cond_features = []
        for seg in segments:
            try:
                tokens = clip.tokenize(seg)
                c_out, p_out = clip.encode_from_tokens(tokens, return_pooled=True)
                cond_feat = c_out.squeeze(0) if c_out is not None else None
            except:
                cond_feat = None
            cond_features.append(cond_feat)

        fused_individual, pooled_individual, norm_w = self.fuse_conditions_equal(cond_features, weights)
        if fused_individual is None:
            c_empty = torch.zeros((1, 768), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            p_empty = torch.zeros((768,), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            prompt_info = "Weight：main_prompt()0%|No valid individuals"
            return ([[c_empty, {"pooled_output": p_empty}]], prompt_info)

        try:
            tokens_whole = clip.tokenize(whole_content)
            c_whole, p_whole = clip.encode_from_tokens(tokens_whole, return_pooled=True)
            feat_whole = c_whole.squeeze(0) if c_whole is not None else torch.zeros_like(fused_individual.squeeze(0))
        except:
            feat_whole = torch.zeros_like(fused_individual.squeeze(0))
            p_whole = torch.zeros_like(pooled_individual)

        if p_whole is None:
            p_whole = torch.zeros_like(pooled_individual)
        if pooled_individual is None:
            pooled_individual = torch.zeros_like(p_whole)
        
        w_overall = main_prompt_ratio
        w_individual_total = 1.0 - main_prompt_ratio
        
        feat_whole_aligned, fused_individual_aligned = self.align_features(
            feat_whole.unsqueeze(0), 
            fused_individual
        )
        
        fused_feat = feat_whole_aligned * w_overall + fused_individual_aligned * w_individual_total
        fused_pooled = p_whole * w_overall + pooled_individual * w_individual_total

        final_parts = []
        main_pct = int(round(w_overall * 100))
        final_parts.append(f"main_prompt({whole_content}){main_pct}%")
        
        part_pcts = []
        for seg, nw in zip(segments, norm_w):
            pct = int(round(nw * w_individual_total * 100))
            part_pcts.append(pct)
            final_parts.append(f"{seg}{pct}%")
        
        total_pct = main_pct + sum(part_pcts)
        if total_pct != 100 and len(part_pcts) > 0:
            diff = 100 - total_pct
            part_pcts[0] += diff
            final_parts = [f"main_prompt({whole_content}){main_pct}%"]
            for i, seg in enumerate(segments):
                final_parts.append(f"{seg}{part_pcts[i]}%")
        
        prompt_info = "Weight：" + "|".join(final_parts)
        
        return ([[fused_feat, {"pooled_output": fused_pooled}]], prompt_info)


class Unit_PromptWeight:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "mode": (["normal", "flux2.klein", "z-image", "qwen-image",], {"default": "normal"}),
                "main_prompt_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ( "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("positive", "negative", )
    CATEGORY = "Apt_UnitPromptWeight"
    FUNCTION = "sum_text_encode"
    OUTPUT_NODE = True
    DESCRIPTION = """
    特征权重追加：[语义@权重数值0~10] 
    例如：个性特征默认权重1.0 ，<1则减弱，>1则加强
    风格减弱： 3d风格，[女孩打伞，瀑布@0.2]
    风格增强：[3d风格@0.2]，女孩打伞，瀑布
    main_prompt_ratio：整体权重占比越高，语义越偏向整体
    """

    
    def sum_text_encode(self, clip, main_prompt_ratio=0.5, pos="", neg="bad", mode="normal"):
        prompt_info = "默认：无提示词处理"

        if neg.strip() == '':
            neg = "bad"
        negative = CLIPTextEncode().encode(clip, neg)[0]
       
        if pos.strip() == '': 
            pos = "a cat"

        if mode in ["z-image", "flux2.klein"]:                
            prompt_weight_processor = pre_Unit_PromptWeight()
            positive, prompt_info = prompt_weight_processor.process(clip, pos, main_prompt_ratio)
        elif mode == "qwen-image":
            prompt_weight_processor = pre_qwenimage_PromptWeight()
            positive, prompt_info = prompt_weight_processor.process(clip, pos, main_prompt_ratio)
        else:
            positive = CLIPTextEncode().encode(clip, pos)[0]
            prompt_info = f"原生编码：{pos[:20]}... | 模式：{mode}"

        return { "ui": {"text": (prompt_info,)}, "result": (positive, negative, ) }
    










