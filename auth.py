import numpy as np
from utils import cosine_sim, augment_cov_vec
from config import DEVICE, LATENT_DIM, WARMUP_EPOCHS, EPOCHS, PATIENCE
from hybrid import hybrid_build_realistic_laplacian, compute_spatial_laplacian
import torch
import torch.nn as nn

class RiemannAuthenticator:
    """
    Nearest-centroid classifier in Riemannian covariance space.

    Threshold tau is estimated as the lower-tail percentile of genuine cosine similarity scores computed on the enrolment 
    set itself.
    """

    def __init__(self, threshold_percentile: float = 10.0):
        self.threshold_percentile = threshold_percentile
        self.templates: dict[int, np.ndarray] = {}
        self.tau: float = 0.0

    def fit(self, X_cov: np.ndarray, y: np.ndarray) -> None:
        """
        X_cov : (N, D) normalised covariance vectors
        y     : (N,)  integer class labels
        """
        for cls in np.unique(y):
            tmpl = X_cov[y == cls].mean(axis=0)
            self.templates[int(cls)] = tmpl / (np.linalg.norm(tmpl) + 1e-9)

        genuine_scores = []
        for cls in np.unique(y):
            tmpl = self.templates[int(cls)]
            for v in X_cov[y == cls]:
                genuine_scores.append(cosine_sim(v, tmpl))

        self.tau = float(np.percentile(genuine_scores, self.threshold_percentile))

    def verify(self, cov_vec: np.ndarray,
               claimed_id: int) -> tuple[bool, float]:
        score = cosine_sim(cov_vec, self.templates[claimed_id])
        return score >= self.tau, score
    
class HybridClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits   = self.classifier(features)
        return logits, features
    
class HybridAuthenticator:
    """
    exact to RiemannAuthenticator except the methods:

    fit()    : trains HybridClassifier on enrolment covariance vectors
                 - 3× augmentation per enrolment trial
                 - warmup for WARMUP_EPOCHS epochs (CE loss only)
                 - at epoch WARMUP_EPOCHS: build Laplacian from latent features
                 - post-warmup: CE + 0.01 * Laplacian loss
                 - early stopping with PATIENCE on CE loss
    verify() : accepts/rejects a claimed identity using the softmax probability of the claimed class as the similarity score. Threshold tau is estimated from 
               the distribution of genuine softmax scores on the enrolment set, mirroring the percentile-based tau used in RiemannAuthenticator.
    """

    def __init__(self, threshold_percentile: float = 10.0):
        self.threshold_percentile = threshold_percentile
        self.model:     HybridClassifier | None = None
        self.n_classes: int   = 0
        self.tau:       float = 0.0
        self.input_dim: int   = 0

    def fit(self, X_cov: np.ndarray,
            y:     np.ndarray,
            rng:   np.random.Generator) -> None:
        """
        X_cov : (N, D) normalised covariance vectors  — enrolment set only
        y     : (N,)   integer class labels
        rng   : shared RNG (for augmentation reproducibility)
        """
        self.n_classes = len(np.unique(y))
        self.input_dim = X_cov.shape[1]

        X_aug, y_aug = [], []
        for vec, lbl in zip(X_cov, y):
            X_aug.append(vec);  y_aug.append(lbl)
            for _ in range(3):
                X_aug.append(augment_cov_vec(vec, rng))
                y_aug.append(lbl)

        X_aug_t = torch.tensor(np.stack(X_aug), dtype=torch.float32).to(DEVICE)
        y_aug_t = torch.tensor(np.array(y_aug), dtype=torch.long).to(DEVICE)

        model     = HybridClassifier(self.input_dim, self.n_classes, LATENT_DIM).to(DEVICE)
        opt       = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        L                = None
        best_loss        = float("inf")
        patience_counter = 0
        best_state       = None

        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            logits, features = model(X_aug_t)
            ce_loss          = criterion(logits, y_aug_t)

            # at warmup boundary: extract latent features, build Laplacian
            if epoch == WARMUP_EPOCHS:
                model.eval()
                with torch.no_grad():
                    _, fp = model(X_aug_t)
                    fp    = fp.cpu().numpy()
                model.train()
                Lc = hybrid_build_realistic_laplacian(fp, k=4).to(DEVICE)
                if not torch.isnan(Lc).any():
                    L = Lc

            # laplacian regularisation (post-warmup)
            if L is not None:
                u   = features.unsqueeze(-1)
                lap = compute_spatial_laplacian(u, L)
                ll  = torch.mean(lap ** 2)
                if torch.isnan(ll):
                    ll = torch.tensor(0.0, device=DEVICE)
                loss = ce_loss + 0.01 * ll
            else:
                loss = ce_loss

            loss.backward()
            opt.step()

            # early stopping, only active after Laplacian is built
            if L is not None:
                if ce_loss.item() < best_loss - 1e-4:
                    best_loss        = ce_loss.item()
                    best_state       = {k: v.clone()
                                        for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model

        # eEstimate tau from genuine softmax scores on enrolment set
        self.model.eval()
        X_enrol_t = torch.tensor(X_cov, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits_e, _ = self.model(X_enrol_t)
            probs_e     = torch.softmax(logits_e, dim=1).cpu().numpy()

        genuine_scores = [float(probs_e[i, y[i]]) for i in range(len(y))]
        self.tau = float(np.percentile(genuine_scores, self.threshold_percentile))

    def verify(self, cov_vec: np.ndarray,
               claimed_id:   int) -> tuple[bool, float]:
        """
        Score = softmax probability assigned to claimed_id.
        Accepted if score >= tau.
        """
        assert self.model is not None, "Call fit() before verify()"
        self.model.eval()
        x_t = torch.tensor(cov_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = self.model(x_t)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # guard: claimed_id may be outside the range seen at training (edge case when a subject has no enrolment trials on this day)
        if claimed_id >= len(probs):
            return False, 0.0

        score = float(probs[claimed_id])
        return score >= self.tau, score

    # lets the attack loop check for known subjects
    @property
    def templates(self) -> dict[int, bool]:
        return {i: True for i in range(self.n_classes)}