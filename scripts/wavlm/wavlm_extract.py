import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_feats = []
    all_labels = []

    for x, mask, y in loader:
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
        pooled = [model.mean_pooling(h, frame_mask) for h in out.hidden_states]  # 13 x (B, 768)
        feats = torch.stack(pooled, dim=1)                                       # (B, 13, 768)
        all_feats.append(feats.cpu())
        all_labels.append(y)

    return torch.cat(all_feats), torch.cat(all_labels)


def load_or_extract(model, loader, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  loading cache  {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["features"], data["labels"]

    print(f"  extracting  ->  {cache_path}")
    feats, labels = extract_features(model, loader, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "labels": labels}, cache_path)
    return feats, labels


def cached_loader(feats, labels, batch_size, shuffle):
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
