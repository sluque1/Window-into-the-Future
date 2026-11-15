import os
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm


# Real EgoVLPv2 feature extractor.
#
# EGOVLPV2_REPO must point at a clone of https://github.com/facebookresearch/EgoVLPv2
# -- specifically at the directory CONTAINING the inner `EgoVLPv2` package, so that
# `from EgoVLPv2.model.model import FrozenInTime` resolves. See the README beside
# this file.
import sys
EGOVLPV2_REPO = os.environ.get("EGOVLPV2_REPO", "")
if EGOVLPV2_REPO:
    sys.path.insert(0, EGOVLPV2_REPO)
try:
    from EgoVLPv2.model.model import FrozenInTime
except ImportError as exc:                                      # pragma: no cover
    raise SystemExit(
        "Could not import EgoVLPv2. Clone https://github.com/facebookresearch/EgoVLPv2 "
        "and point EGOVLPV2_REPO at the directory containing the inner EgoVLPv2 "
        f"package (currently EGOVLPV2_REPO={EGOVLPV2_REPO!r}).\n  {exc}")
import torchvision.transforms as T

class EgoVLPv2FeatureExtractor:
    def __init__(self, weight_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Model parameters (adjust num_frames as needed for your weights)
        video_params = {
            'model': 'SpaceTimeTransformer',
            'num_frames': 16,  # Use 16 if your weights are trained with 16
            'pretrained': False,
            'drop_path_rate': 0.1
        }
        text_params = {
            'model': 'roberta-base',
            'pretrained': True
        }
        self.model = FrozenInTime(
            video_params=video_params,
            text_params=text_params,
            projection_dim=4096,
            load_checkpoint=weight_path,
            projection='minimal'
        )
        self.model.eval()
        self.model.to(self.device)
        self.num_frames = video_params['num_frames']
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frames):
        # Preprocess frames
        processed = [self.transform(f) for f in frames]
        # Shape: [num_frames, 3, 224, 224]
        video_tensor = torch.stack(processed)  # [T, C, H, W]
        # Add batch dimension and reorder to [B, F, C, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, T, C, H, W]
        # Pad or trim to num_frames
        current_frames = video_tensor.shape[1]
        if current_frames < self.num_frames:
            pad = torch.zeros((1, self.num_frames - current_frames, 3, 224, 224), device=video_tensor.device)
            video_tensor = torch.cat([video_tensor, pad], dim=1)
        elif current_frames > self.num_frames:
            video_tensor = video_tensor[:, :self.num_frames]
        video_tensor = video_tensor.to(self.device)
        with torch.no_grad():
            features = self.model.compute_video(video_tensor)
        return features.squeeze(0).cpu()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, nargs='+', default=None,
                        help='Process only these participants (e.g. --participant P01 P02). Omit to process all.')
    parser.add_argument('--gpu_batch_size', type=int, default=16,
                        help='Number of seconds to process per GPU forward pass.')
    # Paths: CLI flag wins, then the environment, then the layout this was
    # developed against. HDEPIC_ROOT moves video_root and output_root together
    # and is the same variable loader.py reads, so training finds what this writes.
    _root = os.environ.get('HDEPIC_ROOT', '/data/video_datasets/HD-EPIC')
    parser.add_argument('--weights', type=str,
                        default=os.environ.get('EGOVLPV2_WEIGHTS', ''),
                        help='Path to EgoVLPv2.pth. Required (or set EGOVLPV2_WEIGHTS).')
    parser.add_argument('--video_root', type=str,
                        default=os.environ.get('VIDEOS_BASE', f'{_root}/Videos'),
                        help='Directory holding <pid>/*.mp4.')
    parser.add_argument('--output_root', type=str,
                        default=os.environ.get(
                            'FEATURES_BASE',
                            f'{_root}/fiction_hd_epic/intermediate/'
                            'extracted_rgb_features_EgoVLPv2'),
                        help='Where to write <split>/<pid>/<video>_features.pt.')
    args = parser.parse_args()

    if not args.weights:
        raise SystemExit(
            "No EgoVLPv2 checkpoint given. Pass --weights /path/to/EgoVLPv2.pth "
            "or set EGOVLPV2_WEIGHTS. The checkpoint is not redistributed here; "
            "see preprocessing/README.md for where to get it.")
    if not os.path.exists(args.weights):
        raise SystemExit(f"EgoVLPv2 checkpoint not found: {args.weights}")

    extractor = EgoVLPv2FeatureExtractor(args.weights)
    # Participant -> split. Must match loader.py's PARTICIPANT_TO_SPLIT, since
    # loader.py looks features up by the split directory this writes them into.
    splits = {
        'train': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06'],
        'val':   ['P07'],
        'test':  ['P08', 'P09'],
    }
    video_root = args.video_root
    output_root = args.output_root
    print(f"[extract] videos  <- {video_root}")
    print(f"[extract] features -> {output_root}")
    os.makedirs(output_root, exist_ok=True)
    for split, folders in splits.items():
        for folder in folders:
            if args.participant and folder not in args.participant:
                continue
            folder_path = os.path.join(video_root, folder)
            video_paths = sorted(glob.glob(os.path.join(folder_path, '*.mp4')))
            for video_path in tqdm(video_paths, desc=f'{split}/{folder}', unit='video'):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                out_dir = os.path.join(output_root, split, folder)
                feature_path = os.path.join(out_dir, f'{video_name}_features.pt')
                if os.path.exists(feature_path):
                    print(f"SKIP: {video_name} already exists", flush=True)
                    continue
                print(f"START: Processing {video_name}", flush=True)
                cap = cv2.VideoCapture(video_path)
                video_fps = int(cap.get(cv2.CAP_PROP_FPS))
                vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                num_seconds = vlen // video_fps
                frame_indices = list(range(0, num_seconds * video_fps, video_fps))
                print(f"  {num_seconds} seconds at {video_fps} FPS — reading sequentially...", flush=True)

                # Pre-compute which frame indices are needed for each second
                all_sample_indices = []
                for t in range(num_seconds):
                    s = np.linspace(t * video_fps, (t + 1) * video_fps - 1, 16, dtype=int)
                    all_sample_indices.append(s)

                # Process in chunks to bound RAM usage (~2-3 GB per chunk)
                chunk_size = 300  # seconds per chunk
                features_list = []
                current_frame_idx = 0
                num_chunks = (num_seconds + chunk_size - 1) // chunk_size
                for chunk_start in tqdm(range(0, num_seconds, chunk_size),
                                        total=num_chunks, desc='  chunks', unit='chunk', leave=False):
                    chunk_end = min(chunk_start + chunk_size, num_seconds)

                    # Frames needed for this chunk
                    chunk_needed = set()
                    for t in range(chunk_start, chunk_end):
                        chunk_needed.update(all_sample_indices[t].tolist())
                    last_chunk_frame = max(chunk_needed)

                    # Read frames sequentially (continuing from where we left off)
                    frame_store = {}
                    with tqdm(total=last_chunk_frame - current_frame_idx + 1,
                              desc='  read', unit='frame', leave=False) as pbar:
                        while current_frame_idx <= last_chunk_frame:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if current_frame_idx in chunk_needed:
                                frame_store[current_frame_idx] = extractor.transform(frame)
                            current_frame_idx += 1
                            pbar.update(1)

                    # GPU inference for this chunk
                    num_batches = (chunk_end - chunk_start + args.gpu_batch_size - 1) // args.gpu_batch_size
                    for batch_start_t in tqdm(range(chunk_start, chunk_end, args.gpu_batch_size),
                                              total=num_batches, desc='  infer', unit='batch', leave=False):
                        batch_end_t = min(batch_start_t + args.gpu_batch_size, chunk_end)
                        clips = []
                        for t in range(batch_start_t, batch_end_t):
                            tensors = [frame_store.get(int(fn), torch.zeros(3, 224, 224))
                                       for fn in all_sample_indices[t]]
                            clips.append(torch.stack(tensors))  # [16, 3, 224, 224]
                        video_batch = torch.stack(clips).to(extractor.device)  # [B, 16, 3, 224, 224]
                        with torch.no_grad():
                            feats = extractor.model.compute_video(video_batch)  # [B, 4096]
                        features_list.append(feats.cpu())

                    del frame_store  # free RAM before next chunk

                cap.release()

                features = torch.cat(features_list, dim=0)  # [num_seconds, 4096]
                os.makedirs(out_dir, exist_ok=True)
                torch.save({'features': features, 'frame_indices': frame_indices}, feature_path)
                print(f"DONE: {video_name} — {num_seconds} features saved to {feature_path}", flush=True)
