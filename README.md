# PreSCAN
PreSCAN: Scene Aware Satellite NeRF architecture Selector
#  Pipeline Overview

This project implements a lightweight framework to **predict NeRF reconstruction quality** (PSNR) for satellite scenes using a learned mapping between scene statistics and architecture configurations.

##  Key Steps Performed in `main.py`

##  Dataset Download

You must  download the required datasets from the following Google Drive link if you want to train your own model or predict on custom scenes:

🔗 [Download Dataset](https://drive.google.com/drive/folders/1Lv4VAdmYGPNjaxxi3zVOVsv98QebwGQJ?usp=sharing)


1. **Data Loading**  
   The pipeline loads two datasets (`jaxoriginaldb.csv` and `omhoriginaldb.csv`) containing per-scene metadata and model architecture descriptors. These are concatenated into a unified dataframe.

2. **Scene Feature Extraction**  
   For each scene, we compute geometric and photometric descriptors using the `compute_scene_features()` function. This includes:
   - Mean inverse PSNR
   - Cosine similarity between view directions
   - Yaw diversity
   - Image variance
   - Coverage density

3. **Data Merging**  
   Extracted scene descriptors are merged with architectural parameters (`Layers`, `Feat`, `n_sample`, etc.) to form a complete training feature matrix.

4. **Model Training**  
   A lightweight MLP model (`LearnableSimilarityModel`) is trained to regress the true PSNR of a scene using the combined feature representation.

5. **Saving Outputs**  
   The trained model and scaler are saved to disk under `saved_models/`.

6. **Example usage for Evaluation**
     python evaluator.py \
      --scene_name JAX_041 \
      --layers 10 \
      --feat 128 \
      --n_sample 64 \
      --std 0.01 \
      --model_path saved_models/weighted_psnr_model.pt \
      --scaler_path saved_models/weighted_psnr_scaler.pkl

##  Output Files

- `saved_models/psnr_model.pt`: Trained PyTorch model weights  
- `saved_models/psnr_scaler.pkl`: Normalization scaler used during training  
- `merged_base.csv`: Combined metadata from JAX and OMA scenes  
- `df_scene`: Extracted geometric/photometric features per scene
