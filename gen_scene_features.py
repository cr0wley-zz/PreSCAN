
import pandas as pd
from tqdm import tqdm
from data.features import compute_scene_features

def generate_scene_features(input_csv_path, output_csv_path):
    df_base = pd.read_csv(input_csv_path)
    all_scene_features = []

    for _, row in tqdm(df_base.iterrows(), total=len(df_base)):
        scene_name = row["Name"] if "Name" in row else row["scene_name"]
        psnr_val = row["PSNR"] if "PSNR" in row else row.get("true_psnr", None)

        result = compute_scene_features(scene_name, psnr_val)
        if result:
            all_scene_features.append(result)

    df_scene_features = pd.DataFrame(all_scene_features)
    df_scene_features.to_csv(output_csv_path, index=False)
    print(f"[✓] Scene features saved to {output_csv_path}")

if __name__ == "__main__":
    generate_scene_features("sample.csv", "scene_features.csv")