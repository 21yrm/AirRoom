import os
from torch.utils.data import Dataset

class Query(Dataset):
    def __init__(self, root_dir, room_label_path):
        self.root_dir = root_dir
        self.image_paths = self._collect_image_path()
        self.labels = self._load_label_mapping(room_label_path)

    def _collect_image_path(self):
        images = []
        for scene in os.listdir(self.root_dir):
            scene_dir = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_dir):
                for room in os.listdir(scene_dir):
                    room_dir = os.path.join(scene_dir, room)
                    query_dir = os.path.join(room_dir, "rgb")
                    if os.path.isdir(query_dir):
                        for image in os.listdir(query_dir):
                            image_path = os.path.join(query_dir, image)
                            if os.path.isfile(image_path):
                                images.append(image_path)

        return images
    
    def _load_label_mapping(self, path):
        labels = {}
        with open(path, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                labels[name] = int(label)

        return labels

    def _get_label(self, name):
        return self.labels[name]
    
    def _extract_scene_room(self, path):
        parts = path.split(os.sep)
        scene = parts[-4]
        room = parts[-3]

        return f"{scene}/{room}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        room_name = self._extract_scene_room(image_path)
        label = self._get_label(room_name)

        return image_path, label
