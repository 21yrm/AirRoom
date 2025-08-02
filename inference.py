import os
from pathlib import Path
import warnings
import yaml
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image as Image
from scipy.spatial import Delaunay
from third_party.SemanticSAM.semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
from data.query_dataset import Query
from data.reference_dataset import Reference
from models.build_model import build_lightglue, build_resnet, build_anyloc
from third_party.LightGlue.lightglue.utils import load_image, rbd

GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_room_dict(filepath):
    room_dict = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            room, room_label = line.strip().split()
            room_dict[int(room_label)] = room
    
    return room_dict

def prepare_image(image_pth, device):
    image = Image.open(image_pth).convert('RGB')
    image_np = np.asarray(image)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(device)

    return image_np, image_tensor

def calculate_center(bbox):
    X, Y, W, H = bbox
    cx = X + W / 2
    cy = Y + H / 2
    return [cx, cy]

def get_adjacent_matrix(centers):
    center_points = np.array(centers)
    n = len(center_points)
    if n == 1:
        return np.array([[0]], dtype=int)
    elif n == 2:
        adjacency_matrix = np.array([[0, 1],
                                     [1, 0]], dtype=int)
        return adjacency_matrix
    
    try:
        tri = Delaunay(center_points)
    except Exception as e:
        adjacency_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix
    
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                v0, v1 = simplex[i], simplex[j]
                adjacency_matrix[v0, v1] = 1
                adjacency_matrix[v1, v0] = 1
    
    return adjacency_matrix

def get_patches(bboxes, adjacency_matrix):
    bboxes = np.array(bboxes, dtype=int)
    if adjacency_matrix is None or len(adjacency_matrix) == 1:
        return bboxes
    
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

def extract_feature(image, masks, patches, model, device, resnet, vlad=None):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
            image_encoding = resnet(batch)
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
            image_encoding = resnet(batch)
            patches_feature = image_encoding.view(image_encoding.size(0), -1)
    else:
        patches_feature = None

    # room
    image_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_tensor = image_tensor[0]
        c, h, w = image_tensor.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        image_tensor = transforms.CenterCrop((h_new, w_new))(image_tensor)[None, ...]
        ret = model(image_tensor)
        gd = vlad.generate(ret.cpu().squeeze())
        gd_np = gd.numpy()[np.newaxis, ...]
        room_feature = torch.from_numpy(gd_np)
        room_feature = room_feature[0]

    return objects_feature, patches_feature, room_feature

def get_connection(query, reference, device):
    connection = torch.zeros(query.size(0), reference.size(0)).to(device)
    for i in range(query.size(0)):
        for j in range(reference.size(0)):
            if query[i] == reference[j]:
                connection[i, j] = 1

    return connection

def match_descriptors(x, y, mode, device):
    x_norm = F.normalize(x, p=2, dim=1).to(device)
    y_norm = F.normalize(y, p=2, dim=1).to(device)
    similarity = torch.mm(x_norm, y_norm.transpose(0, 1))
    max_similarities_qr, max_indices_qr = similarity.max(dim=1)
    similarity_tranposed = similarity.transpose(0, 1)
    max_similarities_rq, max_indices_rq = similarity_tranposed.max(dim=1)

    descriptor_score = 0.0
    if mode == 'max':
        for qry_idx, ref_idx in enumerate(max_indices_qr):
            ref_idx = ref_idx.item()
            if max_indices_rq[ref_idx].item() == qry_idx:
                sim = max_similarities_qr[qry_idx].item()
                if sim > descriptor_score:
                    descriptor_score = sim
    elif mode == 'mean':
        num = 0
        for qry_idx, ref_idx in enumerate(max_indices_qr):
            ref_idx = ref_idx.item()
            if max_indices_rq[ref_idx].item() == qry_idx:
                sim = max_similarities_qr[qry_idx].item()
                num += 1
                descriptor_score += sim
        if num > 0:
            descriptor_score /= num

    return descriptor_score

def get_topk_ref_indices_per_query(counts, topk_indices, k):
    topk_ref_indices = []
    for i, qry_counts in enumerate(counts):
        counts_tensor = torch.tensor(qry_counts)
        _, topk_counts_indices = torch.topk(counts_tensor, k=k, dim=0)
        topk_refs = [topk_indices[i][idx.item()] for idx in topk_counts_indices]
        topk_ref_indices.append(topk_refs)

    return topk_ref_indices

def classify(qry_room_feat, ref_room_feat, qry_objects_feat, ref_objects_feat, qry_patches_feat, ref_patches_feat, connection, dataset_dir, ref_labels, qry_img_paths, label_path, k, device, extractor, matcher):
    room_dict = get_room_dict(label_path)

    # Global Stage
    qry_room_feat = qry_room_feat[:, None, :]
    ref_room_feat = ref_room_feat[None, :, :]
    similarity = F.cosine_similarity(qry_room_feat, ref_room_feat, dim=-1)
    topk_values, topk_indices = torch.topk(similarity, k=k, dim=1)
    print(f'\n{BLUE}Global Stage completed.{RESET}')
    
    correct = 0
    total = qry_room_feat.size(0)
    y_true = []
    y_pred = []

    # Local Stage
    patch_scores_list = []
    for i in range(total):
        qry_patches = qry_patches_feat[i]
        qry_objects = qry_objects_feat[i]
        qry_scores = []

        for idx, ref_index in enumerate(topk_indices[i]):
            ref_patches = ref_patches_feat[ref_index]
            ref_objects = ref_objects_feat[ref_index]
            if qry_patches is None or ref_patches is None:
                patch_score = 0.0
            else:
                patch_score = match_descriptors(qry_patches, ref_patches, 'max', device)
            if qry_objects is None or ref_objects is None:
                object_score = 0.0
            else:
                object_score = match_descriptors(qry_objects, ref_objects, 'mean', device)
            qry_scores.append(topk_values[i][idx].item() + patch_score + object_score)
        patch_scores_list.append(qry_scores)

    top2_indices = get_topk_ref_indices_per_query(patch_scores_list, topk_indices, 2)
    print(f'{BLUE}Local Stage completed.{RESET}')

    # Fine-Grained Stage
    for i in tqdm(range(total), desc="LightGlue Matching Progress"):
        qry_img_path = qry_img_paths[i]
        qry_img = load_image(qry_img_path)
        qry_feats = extractor.extract(qry_img.to(device))

        max_match_pts = 0
        label = -1
        for j in range(2):
            ref_room_name = room_dict.get(ref_labels[top2_indices[i][j]].item())
            ref_img_dir = f"{dataset_dir}/{ref_room_name}/ref/rgb"
            for ref_img_name in os.listdir(ref_img_dir):
                ref_img_path = os.path.join(ref_img_dir, ref_img_name)

                ref_img = load_image(ref_img_path)
                ref_feats = extractor.extract(ref_img.to(device))
                matches01 = matcher({"image0": qry_feats, "image1": ref_feats})
                feats0, feats1, matches01 = [rbd(x) for x in [qry_feats, ref_feats, matches01]]
                kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                if len(m_kpts0) > max_match_pts:
                    max_match_pts = len(m_kpts0)
                    label = top2_indices[i][j]

        true_label = connection[i].argmax().item()
        y_true.append(true_label)
        if label == -1:
            label = torch.tensor(0)
        y_pred.append(label.item())

        if connection[i, label] == 1:
            correct += 1

    print(f'\n{BLUE}Fine-Grained Stage completed.{RESET}')

    return y_true, y_pred, correct

def inference(config):
    root = Path(__file__).resolve().parent

    # config
    inference_cfg = config['inference']
    dataset_cfg = config['dataset']

    # device
    gpu_id = inference_cfg['gpu_id']
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
    else:
        device = "cpu"

    # Candidate List Size (Global Stage)
    k = inference_cfg['top_k']

    # dataset
    dataset_path = root / "datasets" / dataset_cfg['name']
    room_label_path = dataset_path / "room_label.txt"
    query_dataset = Query(
        root_dir=dataset_path,
        room_label_path=room_label_path,
    )
    ref_dataset = Reference(
        root_dir=dataset_path,
        room_label_path=room_label_path,
    )
    query_loader = DataLoader(query_dataset, batch_size=inference_cfg['batch_size'], shuffle=False, num_workers=dataset_cfg['num_workers'])

    # model
    extractor, matcher = build_lightglue()
    extractor = extractor.eval().to(device)
    matcher = matcher.eval().to(device)
    ssam = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L',
                                                                  ckpt=root.parent/"models"/"swinl_only_sam_many2many.pth",),
                                                                  level=[2]) 
    model, vlad = build_anyloc(root, 32, device)
    resnet = build_resnet()
    resnet = resnet.to(device)

    # reference data preprocess
    num_room = len(ref_dataset)
    ref_objects_feature = []
    ref_patches_feature = []
    ref_room_feature = []
    ref_label = []
    ref_data = [ref_dataset[i] for i in range(num_room)]

    with torch.no_grad():
        for i in range(num_room):
            objects = ref_data[i][0].to(device)
            patches = ref_data[i][1].to(device)
            room = ref_data[i][2].to(device)
            label = ref_data[i][3]

            ref_objects_feature.append(objects)
            ref_patches_feature.append(patches)
            ref_room_feature.append(room)
            ref_label.append(label)

        ref_room_feature = torch.stack(ref_room_feature).to(device)
        ref_label = torch.tensor(ref_label).to(device)

    # inference
    true = []
    pred = []
    total_correct = 0
    sample_runtimes = []
    with torch.no_grad():
        for qry_image_path, qry_label in tqdm(query_loader, desc="Inference Progress"):
            start_time = time.time()
            qry_objects_feature = []
            qry_patches_feature = []
            qry_room_feature = []

            for img_path in tqdm(qry_image_path, desc="Image Feature Extraction Progress"):
                # instance segmentation
                img_np, img_tensor = prepare_image(img_path, device)
                masks = ssam.generate(img_tensor)
                center_pts = [calculate_center(mask['bbox']) for mask in masks]
                bboxes = [mask['bbox'] for mask in masks]
                adjacent_matrix = get_adjacent_matrix(center_pts)
                patches = get_patches(bboxes, adjacent_matrix)
                masks = [mask['segmentation'] for mask in masks]

                # extract feature
                objects_feature, patches_feature, room_feature = extract_feature(img_np, masks, patches, model, device, resnet, vlad)
                qry_objects_feature.append(objects_feature)
                qry_patches_feature.append(patches_feature)
                qry_room_feature.append(room_feature)
            
            qry_room_feature = torch.stack(qry_room_feature).to(device)
            qry_label = qry_label.to(device)
            
            # classify
            connection = get_connection(qry_label, ref_label, device)
            y_true, y_pred, correct = classify(qry_room_feature, ref_room_feature, qry_objects_feature, ref_objects_feature,
                                               qry_patches_feature, ref_patches_feature, connection, 
                                               dataset_path, ref_label, qry_image_path, 
                                               room_label_path, k, device, 
                                               extractor, matcher)           
            true.extend(y_true)
            pred.extend(y_pred)
            total_correct += correct

            end_time = time.time()
            runtime = end_time - start_time
            sample_runtimes.append(runtime)
        
        # Get unique labels
        labels = list(set(true + pred))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        num_classes = len(labels)

        # Initialize confusion matrix
        confusion_matrix = torch.zeros(num_classes, num_classes).to(device)
        for t, p in zip(true, pred):
            i = label_to_index[t]
            j = label_to_index[p]
            confusion_matrix[i, j] += 1

        # Compute per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for idx in range(num_classes):
            TP = confusion_matrix[idx, idx]
            FP = confusion_matrix[:, idx].sum() - TP
            FN = confusion_matrix[idx, :].sum() - TP

            if TP + FP > 0:
                precision = TP / (TP + FP)
            else:
                precision = 0.0

            if TP + FN > 0:
                recall = TP / (TP + FN)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        # Compute macro-averaged metrics
        macro_precision = sum(precision_per_class) / num_classes
        macro_recall = sum(recall_per_class) / num_classes
        macro_f1 = sum(f1_per_class) / num_classes

        # Compute accuracy
        accuracy = total_correct / len(query_dataset)

        # Compute runtime
        average_runtime = sum(sample_runtimes) / len(query_dataset)

        # Output metrics
        print(f'{GREEN}Total: {len(query_dataset)}')
        print(f'Correct: {total_correct}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {macro_precision.item():.4f}')
        print(f'Recall: {macro_recall.item():.4f}')
        print(f'F1: {macro_f1:.4f}')
        print(f"Average processing time per sample: {average_runtime:.4f} seconds{RESET}")

        output_path = root / "results" / dataset_cfg['name'] / "metrics.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f'Total: {len(query_dataset)}\n')
            f.write(f'Correct: {total_correct}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {macro_precision.item():.4f}\n')
            f.write(f'Recall: {macro_recall.item():.4f}\n')
            f.write(f'F1: {macro_f1:.4f}\n')
            f.write(f"Average processing time per sample: {average_runtime:.4f} seconds\n")
        print(f"Evaluation metrics successfully saved at: {output_path}")


if __name__ == "__main__":
    config_path = Path("config") / "inference.yaml"
    config = load_config(config_path)
    warnings.filterwarnings("ignore")
    inference(config)
