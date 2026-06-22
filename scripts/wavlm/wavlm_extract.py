import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


@torch.no_grad()
def extract_features(model, loader, device, desc="extracting"):
    model.eval()
    all_feats = []
    all_labels = []

    for x, mask, y in tqdm(loader, desc=desc, leave=False):
        x, mask = x.to(device), mask.to(device)
        out = model.wavlm_model(
            input_values=x,
            attention_mask=mask,
            output_hidden_states=True,
        )
        frame_mask = model.wavlm_model._get_feature_vector_attention_mask(
            out.last_hidden_state.shape[1],
            attention_mask=mask,
        )
        # drop embedding/CNN hidden state (hidden_states[0]); keep the 12 transformer layers
        pooled = [model.mean_pooling(h, frame_mask) for h in out.hidden_states[1:]]  # 12 x (B, 768)
        feats = torch.stack(pooled, dim=1)                                           # (B, 12, 768)
        all_feats.append(feats.cpu())
        all_labels.append(y)

    return torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def extract_xvectors(model, loader, device, desc="extracting"):
    # x-vector probe: one 512-d embedding per clip from the frozen WavLMForXVector (model.embed
    # already L2-normalizes). Cacheable like the layer-sum feats -> same {features, labels} format.
    model.eval()
    all_feats = []
    all_labels = []
    n_degenerate = 0   # ultra-short clips zeroed by model.embed (NaN std pooling) -> zero-norm row

    for x, mask, y in tqdm(loader, desc=desc, leave=False):
        x, mask = x.to(device), mask.to(device)
        emb = model.embed(x, mask)            # (B, 512)
        n_degenerate += int((emb.norm(dim=1) == 0).sum())
        all_feats.append(emb.cpu())
        all_labels.append(y)

    if n_degenerate:
        print(f"  WARNING: {n_degenerate} ultra-short clip(s) produced a degenerate (zeroed) "
              f"x-vector and won't contribute signal.")
    return torch.cat(all_feats), torch.cat(all_labels)


def load_or_extract(model, loader, cache_path, device, extract_fn=extract_features):
    # extract_fn lets callers swap the layer-sum extractor (default) for extract_xvectors,
    # reusing the same cache load/save logic.
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  loading cache  {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["features"], data["labels"]

    print(f"  extracting  ->  {cache_path}")
    feats, labels = extract_fn(model, loader, device, desc=f"extract {cache_path.stem}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "labels": labels}, cache_path)
    return feats, labels


def cached_loader(feats, labels, batch_size, shuffle):
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
