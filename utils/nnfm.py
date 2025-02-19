import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import cv2

def argmin_cos_distance_thre(a, b, thre=0.6, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    z_weights = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)
        # d_mat: this is cosine distance, 0 means the same direction, 1 means opposite direction
        
        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b) 
        
        z_best_min, z_best_batch = torch.min(d_mat, 2)
        z_best_batch[z_best_min > thre] = -1 # no good match
        z_best_min[z_best_min > thre] = 1 # no good match
        z_weight_batch = 1 - z_best_min
        z_best.append(z_best_batch)
        z_weights.append(z_weight_batch)
    z_best = torch.cat(z_best, dim=-1)
    z_weight = torch.cat(z_weights, dim=-1)
    #print(z_weight.shape, z_best.shape)

    return z_best, z_weight

def get_nn_feat_relation_thre(a,tmpl, mask=None):
    n, c, h, w = a.size() # cont

    a_flat = a.view(n, c, -1) 
    tmpl_flat = tmpl.view(n, c, -1) 
    mask = mask.repeat(1, c, 1) if mask is not None else None
    tmpl_flat = tmpl_flat[mask!=0] if mask is not None else tmpl_flat
    tmpl_flat = tmpl_flat.view(n, c, -1)
    assert n == 1
    for i in range(n):
        z_best, z_weight = argmin_cos_distance_thre(a_flat[i : i + 1], tmpl_flat[i : i + 1])
        z_best = z_best #.unsqueeze(1).repeat(1, c, 1)
    return z_best, z_weight

def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b, d, mask=None):
    # a: x
    # b: ref 
    # c: style
    
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()
    n3, c, h3, w3 = d.size()


    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1) # [n, c, hw]
    b_flat = b.view(n2, c, -1) # [n, c, h2w2]
    b_ref = b_flat.clone() # [n, c, h2w2]
    c_flat = d.view(n3, c, -1) # [n, c, h3w3]

    mask = mask.repeat(1, c, 1) if mask is not None else None
    b_flat = b_flat[mask!=0] if mask is not None else b_flat
    b_flat = b_flat.view(n2, c, -1)
    c_flat = c_flat[mask!=0] if mask is not None else c_flat
    c_flat = c_flat.view(n, c, -1)

    z_new = []
    for i in range(n): # 1
        
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        score = z_best
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(c_flat, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new, score.detach()


def nn_feat_replace_from_score(score, d, mask=None):
    n, c, h, w = d.size()
    c_flat = d.view(n, c, -1)
    mask = mask.repeat(1, c, 1) if mask is not None else None
    
    c_flat = c_flat[mask!=0] if mask is not None else c_flat
    c_flat = c_flat.view(n, c, -1)
    z_new = []
    for i in range(n):
        z_best = score
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(c_flat, 2, z_best)
        z_new.append(feat)
    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new #, score.detach()


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G
  
class CachedNNFMLoss(torch.nn.Module):
    def __init__(self, device, size=None, blocks=[2], mask=None):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.s_feats_list = {} # defined by per block
        self.size = (size, size) if size is not None else None
        self.ref_mask = mask[None,...] if mask is not None else None
        if mask is not None:
            self.ref_mask[self.ref_mask<0.95]=0
        self.block_mask = {}
        

    def compute_cache(self, styles, tpl, c_caches, loss_names=["nnfm_loss", "color_patch"], blocks=[2,3,4]):
        
        styles = styles[None,...]
        tpl = tpl[None,...]
        if self.size is not None:
            c_caches = [F.interpolate(x[None,...], size=self.size, mode='bilinear', align_corners=False) for x in c_caches]
            styles = F.interpolate(styles, size=self.size, mode='bilinear', align_corners=False)
            tpl = F.interpolate(tpl, size=self.size, mode='bilinear', align_corners=False)
        else:
            c_caches = [x[None,...]for x in c_caches]

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
        
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block] 
        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a 
        
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)
            tpl_feats_all = self.get_feats(tpl, all_layers)
                
        cached_matches = {}
        cached_matches_patch = []
        cached_matches_weight = []
        
        for ct in c_caches: 
            ct_feats_all = self.get_feats(ct, all_layers)
            
            for block in blocks:
                
                layers = block_indexes[block]
                ct_feats = torch.cat([ct_feats_all[ix_map[ix]] for ix in layers], 1)
                s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
                self.s_feats_list[block] = s_feats
                tpl_feats = torch.cat([tpl_feats_all[ix_map[ix]] for ix in layers], 1)
                if self.ref_mask is not None:
                    mask = torch.nn.functional.interpolate(self.ref_mask, size=ct_feats.shape[-2:], mode='nearest')
                    mask = mask.flatten(2)
                    mask[mask<0.95]=0
                else:
                    mask = None
                self.block_mask[block] = mask

                if "nnfm_loss" in loss_names and block != 4:
                    if cached_matches.get(block) is None:
                        cached_matches[block] = []
                    _, score = nn_feat_replace(ct_feats, tpl_feats, s_feats, mask=mask)
                    cached_matches[block].append(score)
                
                if "color_patch" in loss_names and block == 4:
                    score_thres, weights = get_nn_feat_relation_thre(ct_feats, tpl_feats, mask=None)
                    cached_matches_patch.append(score_thres)
                    cached_matches_weight.append(weights.detach())
                
        return cached_matches, cached_matches_patch, cached_matches_weight
    
    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs

    def forward(
        self,
        outputs,
        scores = {},
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss", "color_patch"],  
        tmpl_sty = None,
        tmpl_img = None,
        patch_weight = None,
        ref_mask = None,
        pop_id = None
    ):
        tmpl_sty = tmpl_sty[None,...]
        outputs = outputs[None,...]
        tmpl_img = tmpl_img[None,...] if tmpl_img is not None else None
        ref_mask = ref_mask[None,...] if ref_mask is not None else None
        if self.size is not None:
            tmpl_sty = F.interpolate(tmpl_sty, size=self.size, mode='bilinear', align_corners=False)
            outputs = F.interpolate(outputs, size=self.size, mode='bilinear', align_corners=False)
            if tmpl_img is not None:
                tmpl_img = F.interpolate(tmpl_img, size=self.size, mode='bilinear', align_corners=False)
    

        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss', "color_patch", "online_nnfm_loss"]
            if x == "color_patch":
                assert tmpl_sty is not None
            if x == "online_nnfm_loss":
                assert tmpl_img is not None
        

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
        blocks.sort()

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block] # 6 8

        x_feats_all = self.get_feats(outputs, all_layers) # stacked layers

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a 


        if 'color_patch' in loss_names:
            coarse_style_flat = []
            down_fact =16
            
            h_sty, w_sty  = tmpl_sty.shape[2:]
            small_tmpl_style = F.interpolate(tmpl_sty,
                                             (h_sty//down_fact, w_sty//down_fact),
                                              mode='bilinear',
                                              antialias=True, 
                                              align_corners=True)
            if ref_mask is not None:
                ref_mask = F.interpolate(ref_mask,
                                             (h_sty//down_fact, w_sty//down_fact),
                                              mode='nearest')
                #ic(ref_mask.shape, ref_mask.max(), ref_mask.min())
                ref_mask = ref_mask.flatten(2).repeat(1, 3, 1)
                ref_mask = torch.cat((ref_mask, torch.FloatTensor([[[0], [0], [0]]]).cuda()), dim=-1)
                ref_mask[ref_mask<0.9]=0
            
            
            coarse_style_flat = small_tmpl_style.view(1, 3, -1)
            coarse_style_flat = torch.cat((coarse_style_flat, torch.FloatTensor([[[0], [0], [0]]]).cuda()), dim=-1)
            coarse_out_flat = F.interpolate(outputs,
                                (h_sty//down_fact, w_sty//down_fact),
                                  mode='bilinear', 
                                  antialias=True, 
                                  align_corners=True
                                  ).view(1, 3, -1)

        if 'online_nnfm_loss' in loss_names:
            tmpl_feats_all = self.get_feats(tmpl_img, all_layers)

        loss_dict = dict([(x, 0.) for x in loss_names]) 
        for block in blocks: 
            layers = block_indexes[block] 
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1) 
            s_feats = self.s_feats_list[block]
            
            factor = 1
            if "nnfm_loss" in loss_names and block != 4:
                score = scores["nnfm_loss"][block]
                target_feats = nn_feat_replace_from_score(score, s_feats, mask=self.block_mask[block])
                
                loss_dict["nnfm_loss"] += factor*cos_loss(x_feats, target_feats)
            
            if "online_nnfm_loss" in loss_names and block != 4:
                target_feats, _ = nn_feat_replace(x_feats, s_feats, s_feats, mask= self.block_mask[block])
                loss_dict["online_nnfm_loss"] += factor*cos_loss(x_feats, target_feats)

            if "color_patch" in loss_names and block == 4:
                score = scores["color_patch"]
                relation_last = score.unsqueeze(1).repeat(1, 3, 1)
                relation_last[relation_last<0] = coarse_style_flat.shape[-1] - 1
                mask = torch.gather(ref_mask, 2, relation_last)                
                related_img = torch.gather(coarse_style_flat, 2, relation_last)
                loss_patch = (related_img - coarse_out_flat)**2
                loss_patch[relation_last == coarse_style_flat.shape[-1] - 1] = 0
                loss_patch = loss_patch.mean(dim=1)
                coarse_out_flat_mask = coarse_out_flat.mean(dim=1)
                loss_patch[coarse_out_flat_mask > 0.9] = 0
                loss_patch = loss_patch * mask
                if patch_weight is not None:
                    loss_patch = loss_patch * patch_weight
                loss_dict["color_patch"] = torch.mean(loss_patch)*15

        loss_dict["total"] = sum(loss_dict.values())
        return loss_dict

