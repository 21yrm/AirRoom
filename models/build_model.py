import os
import sys
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from typing import Literal
from transformers import Dinov2Model
from timm import create_model
from third_party.LightGlue.lightglue import LightGlue, SuperPoint
from third_party.pytorch_NetVlad import netvlad
from third_party.PatchNetVLAD.patchnetvlad.models.models_generic import get_backend, get_model
from third_party.CVNet.model.CVNet_Rerank_model import CVNet_Rerank
import third_party.CVNet.core.checkpoint as checkpoint
from third_party.AnyLoc.demo.utilities import DinoV2ExtractFeatures, VLAD


def build_lightglue():
    extractor = SuperPoint(max_num_keypoints=2048)
    matcher = LightGlue(features="superpoint")
    return extractor, matcher

def build_dino():
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino.eval()
    return dino

def build_dinov2():
    dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
    return dinov2

def build_vit():
    vit = create_model('vit_base_patch16_224', pretrained=True)
    vit.reset_classifier(0)
    return vit

def build_resnet():
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet

def build_cvnet(root):
    cvnet = CVNet_Rerank(50, 2048)
    ckpt_path = f"{root}/third_party/CVNet/CVPR2022_CVNet_R50.pyth"
    if not os.path.exists(ckpt_path):
        print("\033[31mPlease download the CVNet checkpoint file (CVPR2022_CVNet_R50.pyth) from the following link:\033[0m")
        print("\033[31mhttps://drive.google.com/drive/folders/1gMbZ8JlTyPlH2-6HW5NHRe58LmKfJybx\033[0m")
        print("\033[31mand place it in the 'third_party/CVNet' directory.\033[0m")
        sys.exit("Program terminated due to missing checkpoint file.")
    else:
        print("CVNet checkpoint file already exists.")
    checkpoint.load_checkpoint(ckpt_path, cvnet)
    return cvnet

def build_netvlad(root):
    encoder_dim = 512
    encoder = models.vgg16(pretrained=True)
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)
    model.add_module('pool', net_vlad)
    # Please download checkpoint from "https://drive.google.com/open?id=17luTjZFCX639guSVy00OUtzfTQo4AMF2"
    resume_ckpt = f"{root}/third_party/pytorch_NetVlad/checkpoint.pth.tar"
    checkpoint = torch.load(resume_ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def build_patchnetvlad(root, dim):
    encoder_dim, encoder = get_backend()

    ckpt = f"{root}/third_party/PatchNetVLAD/patchnetvlad/pretrained_models/pittsburgh_WPCA{dim}.pth.tar"
    # Download the pretrained model
    if not os.path.exists(ckpt):
        url = f"https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/pitts_WPCA{dim}.pth.tar?download=true"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(ckpt, "wb") as file, tqdm(
                desc="Downloading Patch-NetVLAD",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    bar.update(len(chunk))
            print("Download Patch-NetVLAD completed successfully.")
        else:
            print("Failed to download Patch-NetVLAD weights. Status code:", response.status_code)
    
    checkpoint = torch.load(ckpt, map_location='cpu')
    config = {
        "pooling": "patchnetvlad",
        "num_pcs": dim,
        "patch_sizes": "2,5,8",
        "strides": "1,1,1",
        "num_clusters": str(checkpoint['state_dict']['pool.centroids'].shape[0]),
        "vladv2": False,
    }
    patchnetvlad = get_model(encoder, encoder_dim, config, append_pca_layer=True)
    patchnetvlad.load_state_dict(checkpoint['state_dict'])
    patchnetvlad.eval()
    return patchnetvlad

def build_anyloc(root, num_c, device):
    cache_dir = f"{root}/third_party/AnyLoc/cache"

    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer, desc_facet, device=device)

    domain = "indoor"
    ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
    c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier, domain, "c_centers.pt")
    assert os.path.isfile(c_centers_file), "Vocabulary not cached!"
    c_centers = torch.load(c_centers_file)
    assert c_centers.shape[0] == num_c, "Wrong number of clusters!"
    vlad = VLAD(num_c, desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
    vlad.fit(None)

    return extractor, vlad

def build_maskrcnn():
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    return model
