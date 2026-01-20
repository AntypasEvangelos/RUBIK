
import torch
import json
import cv2
import argparse
import numpy as np
import os
import os.path as osp
import time
import sys

# Import RUBIK evaluation utilities
# We assume this script is placed in RUBIK/ root, same as eval.py
from eval import (
    estimate_pose_essential, 
    estimate_pose_fundamental, 
    relative_pose_error, 
    backproject_to_3D, 
    scale_cost_function, 
    get_scale
)

from tqdm import tqdm
from scipy.optimize import least_squares

# Add canonicalizer to path
sys.path.append(osp.dirname(osp.abspath(__file__)))

from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer
from canonicalizer.wrappers.roma_wrapper import RoMaWrapper
from simple_roma import roma_model # We need to mock or import actual RoMa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="roma_canonicalized", help="Method to evaluate")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14", help="DINOv2 model for canonicalizer")
    parser.add_argument("--num_iterations", type=int, default=5, help="Canonicalizer iterations")
    parser.add_argument("--estimate_pose", type=str, default="essential", help="Method to estimate pose")
    parser.add_argument("--nuscenes_path", type=str, default="/vast/projects/kostas/geometric-learning/nuscenes/nuscenes-download")
    parser.add_argument("--unidepths_path", type=str, default="/vast/projects/kostas/geometric-learning/nuscenes/unidepths")
    parser.add_argument("--data_path", type=str, default="rubik.json")
    parser.add_argument("--output_path", type=str, default="./results")
    args = parser.parse_args()

    method = args.method
    print(f"Evaluating {method} with canonicalizer...")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1. Initialize Canonicalizer
    print("Initializing Canonicalizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canonicalizer = FrozenEnergyCanonicalizer(
        dino_model=args.dino_model,
        num_iterations=args.num_iterations
    ).to(device)
    
    # 2. Initialize Matcher (RoMa)
    print("Initializing RoMa...")
    # Import RoMa using the same logic as eval.py
    script_dir = osp.dirname(osp.abspath(__file__))
    sys.path.append(osp.join(script_dir, "detector_free/RoMa"))
    from romatch import roma_outdoor
    roma_model = roma_outdoor(device=device)
    
    # 3. Validation Loop (modified from eval.py)
    data = json.load(open(args.data_path))
    all_scenes = list(set([scene for box in data for scene in data[box]]))
    results = {}
    
    # Wrap RoMa
    # Our wrapper expects dictionary input, but RoMa in eval.py is called via match()
    # We will use the canonicalizer explicitly in the loop for clarity and to match eval.py structure
    
    for scene in tqdm(all_scenes):
        for box in tqdm(data, leave=False):
            if data[box].get(scene) is None: continue
            if results.get(box) is None: results[box] = {}

            pairs = [eval(el) for el in list(data[box][scene].keys())]
            paths = [[osp.join(args.nuscenes_path, "sweeps", el[0].split("__")[1].split("__")[0], el[0]),
                      osp.join(args.nuscenes_path, "sweeps", el[1].split("__")[1].split("__")[0], el[1])] for el in pairs]

            for i, pair in enumerate(paths):
                gt_pose = np.array(data[box][scene][str(pairs[i])]["rel_pose"])
                
                start_full = time.time()
                
                # A. Canonicalize
                # Load images as tensors for canonicalizer
                # RoMa uses PIL or path, DINO uses tensors.
                from PIL import Image
                import torchvision.transforms as T
                
                # Load and preprocess for DINO (canonicalizer)
                # DINOv2 expects multiple of 14 roughly, normalized
                img1_pil = Image.open(pair[0]).convert('RGB')
                img2_pil = Image.open(pair[1]).convert('RGB')
                
                # Simple resize to decent resolution for canonicalization
                # (You might want to tune this)
                W, H = img1_pil.size
                scale_factor = 1.0 # Keep original for now or resize to 518 (DINO default)
                
                transform = T.Compose([
                    T.Resize((H//14*14, W//14*14)), # Ensure divisible by 14
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                
                img1_tensor = transform(img1_pil).unsqueeze(0).to(device)
                img2_tensor = transform(img2_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # g_star maps Source -> Target
                    warped_source, g_star = canonicalizer(img1_tensor, img2_tensor)
                    
                # B. Match with RoMa
                # We need to pass the warped image to RoMa. 
                # RoMa accepts paths OR PIL images OR tensors (depending on implementation).
                # romatch.py's match method takes paths or PIL.
                
                # Convert warped source tensor back to PIL
                # Unnormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
                warped_unnorm = warped_source * std + mean
                warped_unnorm = torch.clamp(warped_unnorm, 0, 1)
                warped_pil = T.ToPILImage()(warped_unnorm.squeeze(0).cpu())
                
                # Match (Warped Source, Target)
                warp, certainty = roma_model.match(warped_pil, img2_pil, device=device)
                
                # Sample matches
                matches, certainty = roma_model.sample(warp, certainty)
                mkpts1_warped, mkpts2 = roma_model.to_pixel_coordinates(matches, H, W, H, W)
                
                # C. Transform points back to original source coordinates
                # COORDINATE SYSTEM EXPLANATION:
                #   - g_star operates in NORMALIZED coordinates [-1, 1]
                #   - RoMa outputs keypoints in PIXEL coordinates
                #   - We need to convert: pixel → normalized → transform → pixel
                #
                # TRANSFORMATION FLOW:
                #   mkpts1_warped (pixels) → normalize → mkpts1_warped_norm (normalized)
                #   mkpts1_warped_norm · g_star^{-1} → mkpts1_norm (normalized, original frame)
                #   mkpts1_norm → denormalize → mkpts1 (pixels, original frame)

                from canonicalizer.core.lie_group import Sim2
                from canonicalizer.core.warping import transform_points
                from canonicalizer.core.warping import normalized_to_pixel, pixel_to_normalized

                # Step 1: Convert RoMa keypoints from pixel to normalized coordinates
                mkpts1_warped_norm = pixel_to_normalized(mkpts1_warped, H, W)

                # Step 2: Apply inverse transformation (warped → original)
                g_inv = Sim2.inverse(g_star)
                mkpts1_norm = transform_points(mkpts1_warped_norm, g_inv)

                # Step 3: Convert back to pixel coordinates
                mkpts1 = normalized_to_pixel(mkpts1_norm, H, W)
                
                # Convert back to numpy
                mkpts1 = mkpts1.cpu().numpy()
                mkpts2 = mkpts2.cpu().numpy()
                
                elapsed = time.time() - start_full
                
                # D. Estimate Pose (Copied from eval.py)
                K1 = np.array(data[box][scene][str(pairs[i])]["K1"])
                K2 = np.array(data[box][scene][str(pairs[i])]["K2"])
                
                if args.estimate_pose == "essential":
                    ret = estimate_pose_essential(mkpts1, mkpts2, K1, K2, 0.5)
                elif args.estimate_pose == "fundamental":
                    ret = estimate_pose_fundamental(mkpts1, mkpts2, K1, K2, 0.5) # Typo fix in eval.py call
                    
                if ret is None:
                    results[box][str(pairs[i])] = {
                        "R_est": None, "t_est": None,
                        "t_err_angle": np.inf, "t_err_metric": np.inf, "R_err": np.inf,
                        "time": round(elapsed, 5)
                    }
                else:
                    R_est, t_est, _ = ret
                    t_est = t_est / np.linalg.norm(t_est)
                    
                    # Scale factor
                    depth1 = np.load(osp.join(args.unidepths_path, osp.basename(pair[0]).replace('.jpg', '.npy')))
                    depth2 = np.load(osp.join(args.unidepths_path, osp.basename(pair[1]).replace('.jpg', '.npy')))
                    
                    pts3D_1 = backproject_to_3D(mkpts1, depth1, K1)
                    pts3D_2 = backproject_to_3D(mkpts2, depth2, K2)
                    
                    scale = get_scale(scale_cost_function, 1, R_est, t_est, pts3D_1, pts3D_2)
                    t_est *= scale
                    
                    t_err_angle, t_err_metric, R_err = relative_pose_error(gt_pose, R_est, t_est)
                    
                    results[box][str(pairs[i])] = {
                        "R_est": R_est.tolist(),
                        "t_est": t_est.tolist(),
                        "t_err_angle": round(t_err_angle, 5),
                        "t_err_metric": round(t_err_metric, 5),
                        "R_err": round(R_err, 5),
                        "time": round(elapsed, 5)
                    }
    
    # Save
    json.dump(results, open(osp.join(args.output_path, f"results_{method}.json"), "w"), indent=2)
