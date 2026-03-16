# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #
# import datetime
# import os
# import torch
# import numpy as np
# from utils.loss_utils import l1_loss
# from gaussian_renderer import render
# import sys
# from scene import Scene, GaussianModel, DeformModel
# from utils.general_utils import safe_state, get_expon_lr_func
# from utils.generate_camera import generate_new_cam

# import uuid
# from tqdm import tqdm
# from utils.image_utils import psnr
# from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams, OptimizationParams
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

# try:
#     from fused_ssim import fused_ssim
#     FUSED_SSIM_AVAILABLE = True
# except:
#     FUSED_SSIM_AVAILABLE = False

# try:
#     from diff_gaussian_rasterization import SparseGaussianAdam
#     SPARSE_ADAM_AVAILABLE = True
# except:
#     SPARSE_ADAM_AVAILABLE = False
# from utils.logger import logger_config 
# from scipy.spatial.transform import Rotation
# from utils.data_painter import paint_magnitude_compare 
# from skimage.metrics import structural_similarity as ssim





# def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

#     if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
#         sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

#     datadir = os.path.abspath("./dataset/asu_campus_3p5_4by64_log")
#     current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     tb_writer = prepare_output_and_logger(dataset,current_time)
#     logdir = os.path.join(dataset.model_path, "logs")
#     log_filename = "logger.log"
#     devices = torch.device('cuda')
#     log_savepath = os.path.join(logdir, log_filename)
#     os.makedirs(logdir,exist_ok=True)
#     logger = logger_config(log_savepath=log_savepath, logging_name='gsss')
#     logger.info("datadir:%s, logdir:%s", datadir, logdir)
    
#     first_iter = 0
#     gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    
#     scene = Scene(dataset, gaussians)
#     # gaussians.gaussian_init(vertices_path=os.path.join(scene.datadir, "vertices.mat"))
#     gaussians.gaussian_init()
    
#     scene.dataset_init()
#     opt.iterations = len(scene.train_set) * scene.num_epochs

#     if not testing_iterations:
#         testing_iterations = [opt.iterations]
#     if not saving_iterations:
#         saving_iterations = [opt.iterations]
#     if not checkpoint_iterations:
#         checkpoint_iterations = [opt.iterations]
#     for iter_list in (testing_iterations, saving_iterations, checkpoint_iterations):
#         if opt.iterations not in iter_list:
#             iter_list.append(opt.iterations)

#     deform = DeformModel()
#     deform.train_setting(opt)
    
#     gaussians.training_setup(opt)
#     if checkpoint:
#         (model_params, first_iter) = torch.load(checkpoint)
#         gaussians.restore(model_params, opt)

#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#     iter_start = torch.cuda.Event(enable_timing = True)
#     iter_end = torch.cuda.Event(enable_timing = True)
    
#     def compute_ssim_np(pred_np, gt_np):
#         # SSIM is undefined for very small spatial dims (e.g., 1x100).
#         # Skip SSIM computation for such cases.
#         min_dim = min(pred_np.shape[-2:])
#         if min_dim < 3:
#             return 0.0
#         data_range = float(max(pred_np.max() - pred_np.min(), gt_np.max() - gt_np.min(), 1e-6))
#         win = 3 if min_dim < 7 else 7  # handle 4x16 shapes
#         return ssim(pred_np, gt_np, data_range=data_range, win_size=win, channel_axis=None)
    
#     ema_loss_for_log = 0.0
#     progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
#     first_iter += 1
#     for iteration in range(first_iter, opt.iterations + 1):

#         iter_start.record()

#         gaussians.update_learning_rate(iteration)

#         # Every 1000 its we increase the levels of SH up to a maximum degree
#         if iteration % 1000 == 0:
#             gaussians.oneupSHdegree()
#         if iteration % 1000 == 0:
#             # print("nums of gaussians:", gaussians.get_xyz.shape[0])
#             print("nums of gaussians: {}, Avg Opacity: {:.4f}".format(gaussians.get_xyz.shape[0], gaussians.get_opacity.mean().item()))

#         # Pick a random Camera
#         try:
#             spectrum, tx_pos = next(scene.train_iter_dataset)

#         except:
#             scene.dataset_init()
#             spectrum, tx_pos = next(scene.train_iter_dataset)

#         r_o = scene.r_o
#         gateway_orientation = scene.gateway_orientation 
#         R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
#         tx_pos = tx_pos.cuda()
#         viewpoint_cam = generate_new_cam(R, r_o, image_height=scene.output_height, image_width=scene.output_width)
#         N = gaussians.get_xyz.shape[0]
#         time_input = tx_pos.expand(N, -1)
#         d_xyz, d_rotation, d_scaling, d_signal = deform.step(gaussians.get_xyz.detach(), time_input)


#         # Render
#         if (iteration - 1) == debug_from:
#             pipe.debug = True

#         bg = torch.rand((3), device="cuda") if opt.random_background else background

#         render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
#         image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
#         # if iteration%100==0:
#         #     print("radii.shape:", radii.shape)
#         #     print('radii:', radii)
        
#         pred_magnitude = image[0, :scene.output_height, :scene.output_width]
#         if tb_writer:
#             tb_writer.add_image('pred-magnitude', pred_magnitude.unsqueeze(0), iteration)
        
#         # Loss
#         gt_image = spectrum.cuda().squeeze(0)
#         # Ll1 = l1_loss(pred_magnitude, gt_image) # original loss function

#         # #########################################################
#         # # Linear mapping and weighted average loss function
#         # target = gt_image
#         # output = pred_magnitude

#         # # [설정] 전략 1: Linear Mapping
#         # alpha = 20.0  # target=1.0일 때 weight = 1 + 20 = 21
#         # weights = 1.0 + alpha * target

#         # # 제곱오차 후 가중 평균 (weights 합으로 나눔)
#         # squared_error = (output - target) ** 2
#         # Ll1 = (weights * squared_error).sum() / weights.sum().clamp(min=1e-8)

#         # #########################################################

        
#         # #########################################################
#         # # Threshold mapping and weighted average loss function
#         # target = gt_image
#         # output = pred_magnitude

#         # # Threshold mapping: target > threshold
#         # threshold = 0.05
#         # high_weight = 10.0
#         # weights = torch.ones_like(target, device=target.device)
#         # weights[target > threshold] = high_weight

#         # # squared error and weighted average (divide by weights sum)
#         # squared_error = (output - target) ** 2
#         # # absolute_error = (output - target).abs()
#         # Ll1 = (weights * squared_error).sum() / weights.sum().clamp(min=1e-8)

#         # #########################################################

#         #########################################################
#         # Hybrid: Threshold-activated Linear Weighting
#         target = gt_image
#         output = pred_magnitude

#         # [설정 1] 임계값 및 강도 설정
#         threshold = 0.05  # 이 값 이하는 그냥 노이즈로 보고 가중치 1만 줌
#         alpha = 20.0      # target=1.0일 때 최대 가중치 약 21배

#         # [설정 2] 가중치 맵 생성
#         # 1. 일단 모든 픽셀에 기본 가중치 1.0 부여
#         weights = torch.ones_like(target, device=target.device)

#         # 2. 임계값을 넘는 '진짜 신호'에만 Linear 가중치 적용
#         # (공식: Weight = 1 + alpha * target)
#         mask = target > threshold
#         weights[mask] = 1.0 + alpha * target[mask]

#         # [설정 3] Loss 계산
#         # 주의: 뭉개짐(Blurring)을 막으려면 제곱(**2) 대신 절댓값(.abs()) 사용을 강력 추천합니다.
#         # squared_error = (output - target) ** 2  # L2 (뭉개질 수 있음)
#         absolute_error = (output - target).abs()  # L1 (선명함 유지)

#         # 가중 평균 (Weighted Mean)
#         # Ll1 = (weights * squared_error).sum() / weights.sum().clamp(min=1e-8) # L2 사용 시
#         Ll1 = (weights * absolute_error).sum() / weights.sum().clamp(min=1e-8) # L1 사용 시


#         if FUSED_SSIM_AVAILABLE:
#             ssim_value = fused_ssim(pred_magnitude.unsqueeze(0).unsqueeze(0), gt_image.unsqueeze(0).unsqueeze(0))
#         else:
#             pred_np = pred_magnitude.detach().cpu().numpy()
#             gt_np = gt_image.detach().cpu().numpy()
#             ssim_value = compute_ssim_np(pred_np, gt_np)

#         loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

#         loss.backward()

#         iter_end.record()

#         with torch.no_grad():
#             # Progress bar
#             ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

#             if iteration % 10 == 0:
#                 progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
#                 progress_bar.update(10)
#             if iteration == opt.iterations:
#                 progress_bar.close()

#             # Log and save
            
#             tb_writer.add_scalar('train_loss', loss.item(), iteration)            
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

#             # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
#             if (iteration in saving_iterations):
#                 print("\n[ITER {}] Saving Gaussians".format(iteration))
#                 scene.save(iteration)

#             # Densification
#             if iteration < opt.densify_until_iter:
#                 # Keep track of max radii in image-space for pruning
#                 gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
#                 gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

#                 if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
#                     size_threshold = 3 if iteration > opt.opacity_reset_interval else None
#                     if scene.output_width == 16:
#                         size_threshold = None # 우선 잘 되는지 확인해보기 위해서 16일때도 임시로 None으로 설정
#                         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
#                     else:
#                         size_threshold = 10 # 맵 바꿔서 잘 될때 여기 3이었음
#                         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold, radii) # for 4 by 64, size_threshold is not used
                
#                 if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
#                     gaussians.reset_opacity()


#             #########################################################
#             # Final pruning pass after Gaussians are fixed
            
#             # if iteration % 1000 == 0:
#             #     print("nums of gaussians < 0.01: {}".format((gaussians.get_opacity < 0.01).sum()))

#             # if iteration == 20000:
#             #     gaussians.tmp_radii = radii
#             #     prune_mask = (gaussians.get_opacity < 0.01).squeeze()
#             #     gaussians.prune_points(prune_mask)
#             #     gaussians.tmp_radii = None

#             #########################################################

#             # Optimizer step
#             if iteration < opt.iterations:
#                 gaussians.exposure_optimizer.step()
#                 gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
#                 gaussians.optimizer.step()
#                 gaussians.update_learning_rate(iteration)
#                 deform.optimizer.step()
#                 gaussians.optimizer.zero_grad(set_to_none=True)
#                 deform.optimizer.zero_grad()
#                 deform.update_learning_rate(iteration)
            


#             if (iteration in checkpoint_iterations):
#                 print("\n[ITER {}] Saving Checkpoint".format(iteration))
#                 torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
           
#             if iteration in testing_iterations:
#                 torch.cuda.empty_cache()
                
#                 logger.info("Start evaluation")
#                 iteration_path = os.path.join(dataset.model_path, 'pred_magnitude', f'iter_{iteration}')
#                 os.makedirs(iteration_path, exist_ok=True) 
#                 metrics_path = os.path.join(dataset.model_path, f'iter_{iteration}')
#                 os.makedirs(metrics_path, exist_ok=True)
#                 save_img_idx = 0
#                 all_ssim = []
#                 total_eval = min(len(scene.test_set), 50)
#                 selected_indices = torch.randperm(len(scene.test_set))[:total_eval].tolist()
#                 for idx in selected_indices: 
                    
#                     test_input, test_label = scene.test_set[int(idx)]
                    
#                     r_o = scene.r_o
#                     gateway_orientation = scene.gateway_orientation 
#                     R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
#                     tx_pos = test_label.cuda()
#                     viewpoint_cam = generate_new_cam(R, r_o, image_height=scene.output_height, image_width=scene.output_width)
#                     N = gaussians.get_xyz.shape[0]
#                     time_input = tx_pos.expand(N, -1)
#                     d_xyz, d_rotation, d_scaling, d_signal = deform.step(gaussians.get_xyz.detach(), time_input)
    
#                     render_pkg = render(viewpoint_cam, gaussians, pipe, background,d_xyz, d_rotation, d_scaling, d_signal)

#                     pred_magnitude = render_pkg["render"][0, :scene.output_height, :scene.output_width]

#                     ## save predicted spectrum
#                     pred_spectrum = pred_magnitude.detach().cpu().numpy()
#                     gt_spectrum = test_input.detach().cpu().numpy()
 
                    
#                     pixel_error = np.mean(abs(pred_spectrum - gt_spectrum))
#                     ssim_i = compute_ssim_np(pred_spectrum, gt_spectrum)
#                     logger.info(
#                         "Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error,
#                                                                                         ssim_i))
#                     paint_magnitude_compare(pred_spectrum, gt_spectrum,
#                                         save_path=os.path.join(iteration_path,
#                                                                 f'{save_img_idx}_{idx}.png'))
#                     all_ssim.append(float(ssim_i))
#                     logger.info("Median SSIM is {:.6f}".format(np.median(all_ssim)))
#                     save_img_idx += 1
#                 np.savetxt(os.path.join(metrics_path, 'all_ssim.txt'), all_ssim, fmt='%.4f')

#                 torch.cuda.empty_cache() 



# def prepare_output_and_logger(args,time):    
#     if not args.model_path:
#         if os.getenv('OAR_JOB_ID'):
#             unique_str=os.getenv('OAR_JOB_ID')
#         else:
#             unique_str = str(uuid.uuid4())
#         args.model_path = os.path.join("./outputs/", time)
        
#     # Set up output folder
#     print("Output folder: {}".format(args.model_path))
#     os.makedirs(args.model_path, exist_ok = True)
#     with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
#         cfg_log_f.write(str(Namespace(**vars(args))))

#     # Create Tensorboard writer
#     tb_writer = None
#     if TENSORBOARD_FOUND:
#         tb_writer = SummaryWriter(args.model_path)
#     else:
#         print("Tensorboard not available: not logging progress")
#     return tb_writer

# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
#     if tb_writer:
#         tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar('iter_time', elapsed, iteration)

#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
#                 for idx, viewpoint in enumerate(config['cameras']):
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if train_test_exp:
#                         image = image[..., image.shape[-1] // 2:]
#                         gt_image = gt_image[..., gt_image.shape[-1] // 2:]
#                     if tb_writer and (idx < 5):
#                         tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])          
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
#                 if tb_writer:
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

#         if tb_writer:
#             tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     parser.add_argument('--debug_from', type=int, default=-1)
#     parser.add_argument('--detect_anomaly', action='store_true', default=False)
#     parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
#     parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
#     parser.add_argument("--start_checkpoint", type=str, default = None)
#     parser.add_argument('--gpu', type=int, default=0)
    
#     args = parser.parse_args(sys.argv[1:])
    
#     torch.cuda.set_device(args.gpu)
    
#     print("Optimizing " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     torch.autograd.set_detect_anomaly(args.detect_anomaly)
#     training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

#     # All done
#     print("\nTraining complete.")
