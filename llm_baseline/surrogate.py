"""Load the frozen surrogate MLP (checkpoints/surrogate_mlp.pth) as a grader:
39 design params -> (Model Mass [kg], Sim 1 Dropout Y Disp. [m]).
"""
import joblib
import numpy as np
import torch
import torch.nn as nn

from common import CHECKPOINT_DIR, INPUT_COLS, TARGET_COLS


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim=39, hidden=(128, 128, 64), out_dim=2, p_drop=0.1):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p_drop)]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Surrogate:
    def __init__(self):
        ckpt = torch.load(CHECKPOINT_DIR / "surrogate_mlp.pth", map_location="cpu", weights_only=False)
        assert ckpt["input_cols"] == INPUT_COLS
        assert ckpt["target_cols"] == TARGET_COLS
        self.model = SurrogateMLP(**ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.input_scaler = joblib.load(CHECKPOINT_DIR / "input_scaler.joblib")
        self.target_scalers = joblib.load(CHECKPOINT_DIR / "target_scalers.joblib")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 39) raw design params, in INPUT_COLS order. Returns (N, 2) [mass, disp]."""
        Xs = self.input_scaler.transform(X.astype(np.float32))
        with torch.no_grad():
            pred_s = self.model(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
        mass = self.target_scalers["Model Mass"].inverse_transform(pred_s[:, [0]])
        disp = self.target_scalers["Sim 1 Dropout Y Disp."].inverse_transform(pred_s[:, [1]])
        return np.hstack([mass, disp])

    def predict_one(self, vec: dict) -> tuple[float, float]:
        x = np.array([[vec[c] for c in INPUT_COLS]], dtype=np.float32)
        pred = self.predict(x)[0]
        return float(pred[0]), float(pred[1])
