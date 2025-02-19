#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
from gaussian_renderer import diffuse_render as content_render
from gaussian_renderer import render as render
from gaussian_renderer import depth_render
from utils.general_utils import PILtoTorch2
from PIL import Image
import torch.nn.functional as F

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.data_utils import getDataLoader, InfiniteSamplerWrapper
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from icecream import ic
import cv2
from utils.warp_utils import *
from utils.nnfm import CachedNNFMLoss
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def cv2img_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.
    img = torch.tensor(img, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    return img

def depth_to_image(depth, mask):
    depth_ref_masked = depth * mask.to(depth.device)
    depth_ref_masked = (depth_ref_masked/depth_ref_masked.max())*255.
    depth_ref_masked = depth_ref_masked.detach().cpu().numpy().astype("uint8")
    return depth_ref_masked



def training(dataset, 
             opt, 
             pipe, 
             ref_img,
             scene_id, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from,
             skip_style_loss,
             style_cfg=None, 
             loss_cfg=None
             ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # load a pre-trained scene. Read only     
    gaussians.training_setup(opt)
    if checkpoint and 'pth' not in checkpoint:
        model_params= torch.load(checkpoint)['state_dict'] 
        gaussians.restore_from_sugar(model_params, opt)
    else:
        (model_params, _) = torch.load(checkpoint) 
        gaussians.restore_for_sty(model_params, opt)
    

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()

    scene_id = scene_id

    # Load the reference image and view
    dir_name = os.path.dirname(ref_img)
    ref_img = cv2.imread(ref_img)
    ref_img = cv2img_to_tensor(ref_img)
    ref_cam = torch.load(os.path.join(dir_name, "ref_view.pth"))
    ref_view = Image.open(os.path.join(dir_name, "ref_view.png"))
    ref_view = PILtoTorch2(ref_view).cuda()
    ref_cam.mask = None
    
    # Resize the reference image to the same size as the reference view
    ref_img = F.interpolate(ref_img[None], (ref_view.shape[1], ref_view.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
    print("ref_img shape: ", ref_img.shape)
    print("ref_view shape: ", ref_view.shape)
    print("ref_cam shape: ", ref_cam.original_image.shape)


    ref_cam.original_image = ref_view.clamp(0.0, 1.0)
    viewpoint_stack.append(ref_cam)

    depth_ref = depth_render(ref_cam, gaussians, pipe, background)["render"][0]

    # compute intrinsic matrix
    K = get_intrinsic_matrix(ref_cam.FoVx, ref_cam.image_height, ref_cam.image_width, "cuda")
    # map depth to 3D points in world frame
    ref_points3d = depth_to_world(depth_ref, K, ref_cam.world_view_transform)
    
    if ref_cam.mask is None:
        ref_cam.mask = torch.ones((1, ref_cam.image_height, ref_cam.image_width), dtype=torch.float32, device="cpu")
    
    ref_mask = ref_cam.mask.cuda()
    ref_mask_th = (ref_cam.mask.permute(1,2,0).numpy()*255).astype('uint8')
    ref_mask_th = cv2img_to_tensor(ref_mask_th).cuda()[0:1]
    


    # pre-compute loss terms for each viewpoint:
    #   1. depth 
    #   2. nnfm matching score
    #   3. color patch matching score
    #   4. content image without stylization
    # for each viewpoint
    depth_stack = []
    cos_masks = []
    content_stack = []
    for i in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack[i]
        curr_depth = depth_render(viewpoint_cam, gaussians, pipe, background)["render"][0] # current view depth
        depth_stack.append(curr_depth.detach())
        # compute the cosine mask for visibility checking
        cos_mask = mask_by_cos(ref_points3d, ref_cam, viewpoint_cam, cos_threshold=0.01)
        cos_masks.append(cos_mask)
        # compute the content image without stylization
        content_stack.append(content_render(viewpoint_cam, gaussians, pipe, background)["render"].detach())

    # pre-compute the nnfm and color patch matching scores
    style_loss = CachedNNFMLoss("cuda", size=None, mask=ref_mask_th)
    if not skip_style_loss:
        nnfm_list, patch_list, patch_weight = style_loss.compute_cache(ref_img, content_stack[scene_id], content_stack, blocks=[2,3,4])

    # training loop setup
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # main training loop
    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        pop_id=randint(0, len(viewpoint_stack)-1)
        while pop_id == scene_id:
            pop_id = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack[pop_id]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") # if opt.random_background else background
        
        # Render to the reference view
        render_pkg = content_render(ref_cam, gaussians, pipe, bg)
        image, _, visibility_filter_tmpl, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss = 0
        # Compute the ref-view reconstruction loss
        Ll1 = l1_loss(image, ref_img)
        loss_rec = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, ref_img))
        
        # Construct stylized pseudo views
        # by warping the reference style view to the current view
        # this can be pre-computed and cached to improve efficiency
        ref_points_2d = world_to_image(ref_points3d, K, viewpoint_cam.world_view_transform)
        depth_ref_to_curr = world_to_depth(ref_points3d, viewpoint_cam.world_view_transform)
        warp_ref, ys, xs = colorize_image(ref_points_2d, 
                                              ref_img, 
                                              ref_cam.image_height, 
                                              ref_cam.image_width, 
                                              depth_ref_to_curr, 
                                              depth_stack[pop_id], 
                                              ref_mask,
                                              cos_masks[pop_id]
                                              )
        
        image_pkg= content_render(viewpoint_cam, gaussians, pipe, bg)
        curr_image, viewspace_point_tensor, visibility_filter, radii = image_pkg["render"], image_pkg["viewspace_points"], image_pkg["visibility_filter"], image_pkg["radii"]
        
        # Loss with stylized pesudo views
        if len(ys) == 0:
            loss_ref = 0
        else:
            loss_ref = l1_loss(warp_ref[:,ys,xs], curr_image[:,ys,xs])
        
        
        # Depth loss
        curr_depth = depth_render(viewpoint_cam, gaussians, pipe, bg)["render"][0]
        gt_depth = depth_stack[pop_id]
        loss_depth = l1_loss(curr_depth, gt_depth)        
        
        # Style loss with pre-computation
        if not skip_style_loss:
            if iteration<opt.iterations*0.7:
                curr_list = {}
                for block in nnfm_list.keys():
                    curr_list[block] = nnfm_list[block][pop_id]
                s_loss = style_loss(curr_image,
                                    scores ={"nnfm_loss": curr_list, "color_patch":patch_list[pop_id]}, 
                                    blocks=[2,3,4],
                                    tmpl_sty = ref_img,
                                    patch_weight = patch_weight[pop_id],
                                    ref_mask= ref_mask_th,
                                    pop_id = pop_id
                                    ) 
            
                
                s_loss = s_loss['total']
                
            else:
                loss_names = ["online_nnfm_loss"]
                s_loss = style_loss(curr_image, 
                                    scores ={"nnfm_loss":None}, 
                                    blocks=[2,3],
                                    loss_names = loss_names,
                                    tmpl_sty = ref_img,
                                    tmpl_img = content_stack[scene_id],
                                    ref_mask= ref_mask_th,
                                    pop_id = pop_id
                                    )
                s_loss = s_loss['online_nnfm_loss']*0.2
        else:
            s_loss = 0

        # optional tv loss
        w_variance = torch.mean(torch.pow(curr_image[:, :, :-1] - curr_image[ :, :, 1:], 2))
        h_variance = torch.mean(torch.pow(curr_image[ :, :-1, :] - curr_image[ :, 1:, :], 2))
        img_tv_loss = 1 * (h_variance + w_variance) / 2.0

        loss = 1*loss_rec + 2*loss_ref +10*loss_depth + 1*s_loss+ 0*img_tv_loss
        loss.backward()
        
        dc_tensor = gaussians._features_dc.grad.squeeze(1)
        # Update visibility filter for densification
        visibility_filter = torch.logical_or(visibility_filter, visibility_filter_tmpl)
        iter_end.record()
        
        # Define a linear schedule for densification threshold [optional]
        def grad_schedule(iteration, densify_grad_threshold):
            return densify_grad_threshold*max(0.2,((1-iteration/opt.iterations)**1.1))
        densify_grad_threshold = grad_schedule(iteration, opt.densify_grad_threshold)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss, loss, Ll1, iter_start.elapsed_time(iter_end), testing_iterations, scene, content_render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(dc_tensor, visibility_filter)

                if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0): #or (iteration>opt.iterations*0.7 and iteration%op.densification_interval==0 and iteration<opt.iterations*0.9):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.color_densify_by_split(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
            # this part need revisit
            #if iteration % opt.opacity_reset_interval == 0: #or (dataset.white_background and iteration == opt.densify_from_iter):
            #    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        import copy
        new_args = copy.deepcopy(args)
        new_args.sh_degree = 0
        cfg_log_f.write(str(Namespace(**vars(new_args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2999])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--ref_img", type=str, default = None)
    parser.add_argument("--scene_id", type=int, default = None)
    parser.add_argument("--skip_style_loss", action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(lp.extract(args), 
             op.extract(args), 
             pp.extract(args),
             args.ref_img,
             args.scene_id,
             args.test_iterations, 
             args.save_iterations, 
             args.checkpoint_iterations, 
             args.start_checkpoint, 
             args.debug_from,
             args.skip_style_loss)

    # All done
    print("\nTraining complete.")
