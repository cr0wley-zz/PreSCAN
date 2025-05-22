
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import mean_squared_error
import rpcm

# Data should be in this format
def load_and_prepare_data(jax_path, oma_path):
    df_jax = pd.read_csv(jax_path)
    df_oma = pd.read_csv(oma_path)
    df_jax["Name"] = df_jax["Name"].apply(lambda x: f"JAX_{int(x):03d}")
    df_oma["Name"] = df_oma["Name"].apply(lambda x: f"OMA_{int(x):03d}")
    df = pd.concat([df_jax, df_oma], ignore_index=True)
    return df

def latlon_to_ecef_custom(lat, lon, alt):
    rad_lat = np.radians(lat)
    rad_lon = np.radians(lon)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 1 - (1 - f) ** 2
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat)**2)
    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def compute_psnr(img1, img2):
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_scene_features(scene_name, psnr_value=None, img_ext=".tif", resize_shape=(224, 224), max_images=26, angle_thresh=20, yaw_thresh=30):
    def compute_yaw_deg(x_axis_ground, lon, lat):
        rad_lat = np.radians(lat)
        rad_lon = np.radians(lon)
        east = np.array([-np.sin(rad_lon), np.cos(rad_lon), 0.0])
        east /= np.linalg.norm(east)
        yaw_rad = np.arccos(np.clip(np.dot(x_axis_ground, east), -1, 1))
        yaw_deg = np.degrees(yaw_rad)
        cross = np.cross(east, x_axis_ground)
        if cross[2] < 0:
            yaw_deg = 360 - yaw_deg
        return yaw_deg

    def get_view_vector_and_x_axis_and_yaw(json_path):
        with open(json_path, 'r') as f:
            d = json.load(f)

        rpc_model = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
        height, width = int(d["height"]), int(d["width"])
        min_alt = float(d["min_alt"])
        max_alt = float(d["max_alt"])
        alt = (min_alt + max_alt) / 2

        center_row, center_col = height // 2, width // 2
        offset_col = min(width - 1, center_col + 100)

        lon1, lat1 = rpc_model.localization([center_col], [center_row], [max_alt])
        x1, y1, z1 = latlon_to_ecef_custom(lat1, lon1, max_alt)
        lon2, lat2 = rpc_model.localization([center_col], [center_row], [min_alt])
        x2, y2, z2 = latlon_to_ecef_custom(lat2, lon2, min_alt)
        view_vec = np.array([x2[0] - x1[0], y2[0] - y1[0], z2[0] - z1[0]])
        view_vec /= np.linalg.norm(view_vec)

        lon_c, lat_c = rpc_model.localization([center_col], [center_row], [alt])
        lon_r, lat_r = rpc_model.localization([offset_col], [center_row], [alt])
        xc, yc, zc = latlon_to_ecef_custom(lat_c, lon_c, alt)
        xr, yr, zr = latlon_to_ecef_custom(lat_r, lon_r, alt)
        x_axis_ground = np.array([xr[0] - xc[0], yr[0] - yc[0], zr[0] - zc[0]])
        x_axis_ground /= np.linalg.norm(x_axis_ground)

        yaw = compute_yaw_deg(x_axis_ground, lon_c[0], lat_c[0])
        return view_vec, yaw

    # change according to your own paths
    if scene_name.startswith("JAX_"):
        root_path = "DFCNew/Track3-preprocess"
    elif scene_name.startswith("OMA_"):
        root_path = "DFCNew/Track3-preprocess-oma"
    else:
        raise ValueError(f"Unknown scene prefix: {scene_name}")

    scene_path = os.path.join(root_path, scene_name, "ba")
    crop_path = os.path.join(scene_path, "crops")
    if not os.path.isdir(crop_path):
        return None

    image_files = sorted(Path(crop_path).glob(f"*{img_ext}"))[:max_images]
    if len(image_files) < 2:
        return None

    images, vectors, yaws, variances = {}, {}, {}, []
    for img_path in image_files:
        stem = img_path.stem
        json_path = os.path.join(scene_path, f"{stem}.json")
        try:
            img = Image.open(img_path).convert("RGB").resize(resize_shape)
            img_np = np.array(img).astype(np.float32) / 255.0
            vec, yaw = get_view_vector_and_x_axis_and_yaw(json_path)
            images[stem] = img_np
            vectors[stem] = vec
            yaws[stem] = yaw
            variances.append(np.var(img_np))
        except:
            continue

    keys = list(images.keys())
    if len(keys) < 2:
        return None

    inv_psnrs = []
    inv_psnr_pairs = []
    cos_sims = []
    angles = []
    valid_pair_count = 0
    image_coverage = {k: 0 for k in keys}

    for i, ref in enumerate(keys):
        for j, cand in enumerate(keys):
            if i == j: continue
            v1, v2 = vectors[ref], vectors[cand]
            yaw1, yaw2 = yaws[ref], yaws[cand]
            yaw_diff = abs(yaw1 - yaw2)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
            angles.append(angle)
            if angle <= angle_thresh and yaw_diff <= yaw_thresh:
                try:
                    psnr = compute_psnr(images[ref], images[cand])
                    if np.isfinite(psnr) and psnr > 0:
                        inv_val = 1.0 / psnr
                        inv_psnrs.append(inv_val)
                        inv_psnr_pairs.append((ref, cand, inv_val))
                        cos_sims.append(np.dot(v1, v2))
                        image_coverage[ref] += 1
                        image_coverage[cand] += 1
                        valid_pair_count += 1
                except:
                    continue

    if not inv_psnrs or valid_pair_count == 0:
        return None

    avg_inv_psnr = np.mean(inv_psnrs)
    std_inv_psnr = np.std(inv_psnrs)
    pair_density = valid_pair_count / (len(keys) * (len(keys) - 1))
    coverage_score = sum(1 for v in image_coverage.values() if v > 0) / len(keys)
    mean_cos_sim = np.mean(cos_sims) if cos_sims else 0
    spread_angle = np.max(angles) if angles else 0
    variance_avg = np.mean(variances)

    epsilon = 1
    weighted_sum = 0
    weight_total = 0
    for ref, cand, inv_val in inv_psnr_pairs:
        w_ref = max(image_coverage[ref], epsilon)
        w_cand = max(image_coverage[cand], epsilon)
        weight = w_ref + w_cand
        weighted_sum += weight * inv_val
        weight_total += weight

    weighted_avg_inv_psnr = weighted_sum / weight_total if weight_total > 0 else np.nan

    return {
        "scene_name": scene_name,
        "avg_inv_psnr": avg_inv_psnr,
        "std_inv_psnr": std_inv_psnr,
        "weighted_avg_inv_psnr": weighted_avg_inv_psnr,
        "pair_density": pair_density,
        "coverage_score": coverage_score,
        "mean_cos_sim": mean_cos_sim,
        "view_spread_angle": spread_angle,
        "image_variance_avg": variance_avg,
        "true_psnr": psnr_value
    }
