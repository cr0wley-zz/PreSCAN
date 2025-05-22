import torch
import joblib
import numpy as np
import pandas as pd
import argparse

from data.features import compute_scene_features
from models.predictor import LearnableSimilarityModel

def predict_psnr_for_scene(scene_name, layers, feat, n_sample, std, model, scaler, device="cuda" if torch.cuda.is_available() else "cpu"):
    dummy_psnr = 20
    sim_feats = compute_scene_features(scene_name, psnr_value=dummy_psnr)
    if sim_feats is None:
        print(f"[!] Could not extract features for {scene_name}")
        return None

    feature_vector = [
        sim_feats['avg_inv_psnr'],
        sim_feats['std_inv_psnr'],
        sim_feats['pair_density'],
        sim_feats['coverage_score'],
        sim_feats['mean_cos_sim'],
        sim_feats['view_spread_angle'],
        sim_feats['image_variance_avg'],
        layers, feat, n_sample, std
    ]

    X_scaled = scaler.transform([feature_vector])
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).item()

    return pred

def load_model_and_scaler(model_path, scaler_path, input_dim=11):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableSimilarityModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

def main():
    parser = argparse.ArgumentParser(description="Predict PSNR for a given scene and architecture configuration.")
    parser.add_argument("--scene_name", required=True, help="Scene name (e.g., JAX_001)")
    parser.add_argument("--layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--feat", type=int, required=True, help="Number of features")
    parser.add_argument("--n_sample", type=int, required=True, help="Number of sample points")
    parser.add_argument("--std", type=float, required=True, help="STD value")
    parser.add_argument("--model_path", default="saved_models/psnr_model.pt", help="Path to saved model")
    parser.add_argument("--scaler_path", default="saved_models/psnr_scaler.pkl", help="Path to saved scaler")

    args = parser.parse_args()

    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)

    predicted_psnr = predict_psnr_for_scene(
        scene_name=args.scene_name,
        layers=args.layers,
        feat=args.feat,
        n_sample=args.n_sample,
        std=args.std,
        model=model,
        scaler=scaler
    )

    if predicted_psnr is not None:
        print(f"[✓] Predicted PSNR for {args.scene_name}: {predicted_psnr:.2f}")

if __name__ == "__main__":
    main()
