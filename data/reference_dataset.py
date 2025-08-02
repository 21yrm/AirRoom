import os
import torch
from torch.utils.data import Dataset


class Reference(Dataset):
    def __init__(self, root_dir, room_label_path):
        self.root_dir = root_dir
        self.group_dirs = self._collect_group_dirs()
        self.labels = self._load_label_mapping(room_label_path)

    def _collect_group_dirs(self):
        group_dirs = []
        for scene in os.listdir(self.root_dir):
            scene_dir = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_dir):
                for room in os.listdir(scene_dir):
                    room_dir = os.path.join(scene_dir, room)
                    ref_dir = os.path.join(room_dir, "ref")
                    if os.path.isdir(ref_dir):
                        group_dirs.append(ref_dir)

        return group_dirs
    
    def _load_label_mapping(self, path):
        labels = {}
        with open(path, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                labels[name] = int(label)

        return labels

    def _get_label(self, name):
        return self.labels[name]
    
    def _extract_scene_room(self, group_dir):
        parts = group_dir.split(os.sep)
        scene = parts[-3]
        room = parts[-2]

        return f"{scene}/{room}"

    def __len__(self):
        return len(self.group_dirs)

    def __getitem__(self, idx):
        group_dir = self.group_dirs[idx]
        room_name = self._extract_scene_room(group_dir)
        room_label = self._get_label(room_name)
        
        room_embedding_path = os.path.join(group_dir, "embed/room_feature.pt")
        room_embedding = torch.load(room_embedding_path, map_location='cpu')

        object_embedding_path = os.path.join(group_dir, "embed/objects.pt")
        objects_embedding = torch.load(object_embedding_path, map_location='cpu')

        patch_embedding_path = os.path.join(group_dir, "embed/patches.pt")
        patches_embedding = torch.load(patch_embedding_path, map_location='cpu')
        
        return objects_embedding, patches_embedding, room_embedding, room_label
