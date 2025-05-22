import pandas as pd
import torch
import joblib
import os
from tqdm import tqdm

from data.features import load_and_prepare_data, compute_scene_features
from train.trainer import train_model

def main():
    jax_path = "jaxoriginaldb.csv"
    oma_path = "omhoriginaldb.csv"
    df = load_and_prepare_data(jax_path, oma_path)

    # Save merged base CSV if needed
    df["scene_name"] = df["Name"]
    df.to_csv("merged_base.csv", index=False)

    # Generate scene features directly
    all_scene_features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = compute_scene_features(row["scene_name"], row["PSNR"])
        if result:
            all_scene_features.append(result)
    df_scene = pd.DataFrame(all_scene_features)

    # Merge scene features with architecture descriptors
    df_merged = df_scene.merge(
        df[["scene_name", "Layers", "Feat", "n_sample", "Lr ", "STD"]],
        on="scene_name", how="left"
    ).rename(columns={"Lr ": "Lr"})

    feature_cols = [
        'avg_inv_psnr', 'std_inv_psnr', 'pair_density', 'coverage_score',
        'mean_cos_sim', 'view_spread_angle', 'image_variance_avg',
        'Layers', 'Feat', 'n_sample', 'STD'
    ]
    target_col = 'true_psnr'

    model, scaler = train_model(df_merged, feature_cols, target_col)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/psnr_model.pt")
    joblib.dump(scaler, "saved_models/psnr_scaler.pkl")
    print("Model and scaler saved to saved_models/")

if __name__ == "__main__":
    main()
