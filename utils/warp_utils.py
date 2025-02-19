import torch
import numpy as np
from utils.camera_utils import fov2focal
import cv2

def to_image(im):
    im = torch.clamp(im, 0.0, 1.0)*255
    im = im.permute(1, 2, 0).cpu().numpy().astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def get_intrinsic_matrix(fovx, H, W, device):
    focal = fov2focal(fovx, W)
    cx = W / 2
    cy = H / 2
    K = torch.tensor([[focal, 0, cx], 
                      [0, focal, cy], 
                      [0, 0, 1]], 
                     dtype=torch.float32, 
                     device=device)
    return K

def depth_to_world(depth_map, K, T_wc):
    H, W = depth_map.shape 
    device = depth_map.device
    
    # Generate pixel coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float()+0.5
    y = y.float()+0.5

    # Flatten and convert to homogeneous coordinates
    z = depth_map.flatten()
    x = x.flatten()
    y = y.flatten()

    # Unproject to camera space (inverse intrinsics)
    K_inv = torch.linalg.inv(K)
    x_cam = (x - K[0, 2]) * z / K[0, 0]
    y_cam = (y - K[1, 2]) * z / K[1, 1]
    z_cam = z

    # Camera space points in homogeneous coordinates
    points_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=0)

    # Transform to world space (inverse of world-to-camera)
    T_cw = torch.linalg.inv(T_wc.transpose(0, 1))
    points_world = torch.matmul(T_cw[:3, :3], points_cam[:3, :]) + T_cw[:3, 3].unsqueeze(1)

    # Reshape to original image shape
    points= points_world.reshape(3, H, W).permute(1, 2, 0).view(-1,3)
    
    return points
def world_to_depth(points, T_wc):
    # to homogeneous coordinates
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1) # (N, 4)
    points = torch.matmul(points, T_wc)

    depth = points[:, 2]/points[:, 3]
    return depth

def world_to_image(points, K, T_wc):
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1) # (N, 4)
    points = torch.matmul(points, T_wc) # to camera space
    points = points[:, :3]/points[:, 3].unsqueeze(1) # to homogeneous coordinates
    points2d = torch.matmul(points, K.transpose(0, 1)) # to image space
    coords2d = points2d[:, :2]/points2d[:, 2].unsqueeze(1)
    
    return coords2d

def filter_points(points, H, W):
    mask = (points[:, 0] >= 0) & (points[:, 0] < W) & \
           (points[:, 1] >= 0) & (points[:, 1] < H) & \
           (points[:, 2] > 0)
    return mask

def mask_by_cos(ref_points3d, ref_cam, tgt_cam, cos_threshold=0.6):
    ref_dirs = ref_points3d - ref_cam.camera_center
    ref_dirs = ref_dirs / torch.linalg.norm(ref_dirs, dim=1).unsqueeze(1)
    tgt_dirs = ref_points3d - tgt_cam.camera_center
    tgt_dirs = tgt_dirs / torch.linalg.norm(tgt_dirs, dim=1).unsqueeze(1)
    cos = torch.sum(ref_dirs * tgt_dirs, dim=1)
    mask = cos > cos_threshold
    return mask

def colorize_image(points_2d, colors, H, W, depth_src, depth_tgt, mask, cos_mask):
    colors = colors.permute(1,2,0).flatten(0,1)
    mask = mask.flatten().bool()
    device = points_2d.device
    # Initialize the image tensor with a white background
    image_tensor = torch.ones((3, H, W), dtype=torch.float, device=device)
    valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) & mask & (cos_mask if cos_mask is not None else mask)
    
    valid_points = points_2d[valid_mask]-0.5
    valid_colors = colors[valid_mask]
    
    y_indices, x_indices = valid_points[:, 1].round().long(), valid_points[:, 0].round().long()
    valid_depth = depth_src.flatten()[valid_mask]
    curr_depth = depth_tgt[y_indices, x_indices].flatten()
    visible = (valid_depth<=(curr_depth+1e-3)) &  (valid_depth>0)
    valid_colors = valid_colors[visible]
    y_indices, x_indices = y_indices[visible], x_indices[visible]
    image_tensor[:, y_indices, x_indices] = valid_colors.permute(1, 0)
    return image_tensor, y_indices, x_indices
