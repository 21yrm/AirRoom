import os
from pathlib import Path
import warnings
import yaml
import shutil
import natsort
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from scipy.spatial import Delaunay
import numpy as np
from sklearn.cluster import KMeans
from models.build_model import build_resnet, build_anyloc
from third_party.SemanticSAM.semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator

# color
GREEN = '\033[92m'
RESET = '\033[0m'

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def build_model():
    root = Path(__file__).resolve().parent
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino, vlad = build_anyloc(root, 32, device)

    semantic_sam = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L',
                                                                      ckpt=root.parent/"models"/"swinl_only_sam_many2many.pth"),
                                                                      level=[2])
    resnet = build_resnet()
    resnet = resnet.to(device)
    return semantic_sam, resnet, clip, clip_processor, dino, vlad

def select_reference_image(rgb_path, ref_rgb_path, clip_processor, clip):
    image_files = [os.path.join(rgb_path, f) for f in os.listdir(rgb_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    features = []
    for image_file in image_files:
        image = Image.open(image_file)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip.get_image_features(**inputs)
        features.append(outputs.cpu().numpy())
    features = np.vstack(features)
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(features)

    cluster_indices = np.where(kmeans.labels_ == 0)[0]
    cluster_center = kmeans.cluster_centers_[0]
    closest_image_idx = cluster_indices[np.argmin(np.linalg.norm(features[cluster_indices] - cluster_center, axis=1))]
    selected_image = image_files[closest_image_idx]
    shutil.copy(selected_image, os.path.join(ref_rgb_path, "0.png"))

def prepare_image(image_pth):
    image = Image.open(image_pth).convert('RGB')
    image_np = np.asarray(image)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(device)

    return image_tensor

def calculate_center(bbox):
    X, Y, W, H = bbox
    cx = X + W / 2
    cy = Y + H / 2
    return [cx, cy]

def get_adjacent_matrix(centers):
    center_points = np.array(centers)
    tri = Delaunay(center_points)
    n = len(center_points)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                v0, v1 = simplex[i], simplex[j]
                adjacency_matrix[v0, v1] = 1
                adjacency_matrix[v1, v0] = 1
    
    return adjacency_matrix

def get_patches(bboxes, adjacency_matrix):
    bboxes = np.array(bboxes)
    expanded_bboxes = []

    for i in range(len(bboxes)):
        x_min, y_min = bboxes[i, 0], bboxes[i, 1]
        x_max, y_max = bboxes[i, 0] + bboxes[i, 2], bboxes[i, 1] + bboxes[i, 3]

        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        for neighbor in neighbors:
            nx_min = bboxes[neighbor, 0]
            ny_min = bboxes[neighbor, 1]
            nx_max = bboxes[neighbor, 0] + bboxes[neighbor, 2]
            ny_max = bboxes[neighbor, 1] + bboxes[neighbor, 3]

            x_min = min(x_min, nx_min)
            y_min = min(y_min, ny_min)
            x_max = max(x_max, nx_max)
            y_max = max(y_max, ny_max)

        expanded_w = x_max - x_min
        expanded_h = y_max - y_min
        expanded_bboxes.append([x_min, y_min, expanded_w, expanded_h])

    expanded_bboxes = np.array(expanded_bboxes, dtype=int)
    
    areas = expanded_bboxes[:, 2] * expanded_bboxes[:, 3]
    keep_indices = []
    indices = np.arange(len(expanded_bboxes))
    
    while len(indices) > 0:
        i = indices[0]
        keep_indices.append(i)
        rest_indices = indices[1:]
        
        xx1 = np.maximum(expanded_bboxes[i, 0], expanded_bboxes[rest_indices, 0])
        yy1 = np.maximum(expanded_bboxes[i, 1], expanded_bboxes[rest_indices, 1])
        xx2 = np.minimum(expanded_bboxes[i, 0] + expanded_bboxes[i, 2], expanded_bboxes[rest_indices, 0] + expanded_bboxes[rest_indices, 2])
        yy2 = np.minimum(expanded_bboxes[i, 1] + expanded_bboxes[i, 3], expanded_bboxes[rest_indices, 1] + expanded_bboxes[rest_indices, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[rest_indices] - intersection
        iou = intersection / union

        overlap_threshold = 0.7
        remaining_indices = rest_indices[iou <= overlap_threshold]
        indices = remaining_indices
    
    final_bboxes = expanded_bboxes[keep_indices]
    return final_bboxes

def segmentation(seg_model, room_path):
    image_files = natsort.natsorted(os.listdir(room_path))

    image_path = os.path.join(room_path, image_files[0])
    img_tensor = prepare_image(image_path)
    outputs = seg_model.generate(img_tensor)
    center_pts = [calculate_center(output['bbox']) for output in outputs]
    bboxes = [output['bbox'] for output in outputs]
    adjacent_matrix = get_adjacent_matrix(center_pts)
    patches = get_patches(bboxes, adjacent_matrix)
    masks = [output['segmentation'] for output in outputs]

    return masks, patches

def save_objectNpatch_embedding(image_path, embed_path, masks, patches, model):
    patches_feature = []
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = natsort.natsorted(os.listdir(image_path))
    rgb_img_path = os.path.join(image_path, image_files[0])
    image = Image.open(rgb_img_path).convert('RGB')
    image = np.asarray(image)

    # objects
    if len(masks) > 0:
        batch = []
        for mask in masks:
            image_np = np.copy(image)
            masked_image_np = image_np * mask[:, :, None]
            masked_image = Image.fromarray(masked_image_np.astype(np.uint8))
            image_tensor = preprocess(masked_image).unsqueeze(0)
            batch.append(image_tensor)
        batch = torch.cat(batch, dim=0).to(device)
        with torch.no_grad():
            image_encoding = model(batch)
            objects_feature = image_encoding.view(image_encoding.size(0), -1)
    else:
        objects_feature = None

    # patches
    if len(patches) > 0:
        batch = []
        for patch in patches:
            x, y, w, h = patch
            patch_image_np = image[y:y+h, x:x+w]
            masked_image = Image.fromarray(patch_image_np.astype(np.uint8))
            image_tensor = preprocess(masked_image).unsqueeze(0)
            batch.append(image_tensor)
        batch = torch.cat(batch, dim=0).to(device)
        with torch.no_grad():
            image_encoding = model(batch)
            patches_feature = image_encoding.view(image_encoding.size(0), -1)       
    else:
        patches_feature = None

    torch.save(objects_feature, os.path.join(embed_path, "objects.pt"))
    torch.save(patches_feature, os.path.join(embed_path, "patches.pt"))

def save_room_embedding(image_path, embed_path, dino, vlad):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feat = []
    for img in natsort.natsorted(os.listdir(image_path)):
        pth = os.path.join(image_path, img)
        # DINO features
        with torch.no_grad():
            img_pt = preprocess(Image.open(pth).convert('RGB')).to(device)
            # Make image patchable (14, 14 patches)
            c, h, w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = transforms.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            # Extract descriptor
            ret = dino(img_pt) # [1, num_patches, desc_dim]
        # VLAD global descriptor
        gd = vlad.generate(ret.cpu().squeeze()) # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        feat.append(gd_np[0])

    feat_tensor = torch.tensor(feat)
    averaged_feature = feat_tensor.mean(dim=0)

    torch.save(averaged_feature, os.path.join(embed_path, "room_feature.pt"))

def main(config):
    dataset_path = Path("datasets") / config["dataset_name"]
    seg_model, feat_model, clip, clip_processor, dino, vlad = build_model()

    for scene in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene)
        if not os.path.isdir(scene_path):
            continue
        for room in os.listdir(scene_path):
            room_path = os.path.join(scene_path, room)
            if not os.path.isdir(room_path):
                continue
            rgb_path = os.path.join(room_path, "rgb")
            ref_path = os.path.join(room_path, "ref")
            ref_rgb_path = os.path.join(ref_path, "rgb")
            embed_path = os.path.join(ref_path, "embed")
            if os.path.exists(ref_path):
                shutil.rmtree(ref_path)
            os.makedirs(ref_rgb_path, exist_ok=True)
            os.makedirs(embed_path, exist_ok=True)

            select_reference_image(rgb_path, ref_rgb_path, clip_processor, clip)
            
            with torch.no_grad():
                # instance segmentation
                masks, patches = segmentation(seg_model, ref_rgb_path)

                # extract feature embedding
                save_objectNpatch_embedding(ref_rgb_path, embed_path, masks, patches, feat_model)
                save_room_embedding(ref_rgb_path, embed_path, dino, vlad)

            print(f'{GREEN}Finish processing {scene}/{room}.{RESET}')


if __name__ == '__main__':
    config_path = Path("config") / "preprocess.yaml"
    config = load_config(config_path)
    warnings.filterwarnings("ignore")
    main(config)
