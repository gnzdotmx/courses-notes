import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

data_home = Path(__file__).resolve().parent / "data" / "sklearn_cache"
data_home.mkdir(parents=True, exist_ok=True)

lfw = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4,
    data_home=str(data_home),
    download_if_missing=True,
    n_retries=3,
    delay=1.0,
)
print(f"Loaded {lfw.data.shape[0]} faces, {lfw.data.shape[1]} pixels, {len(lfw.target_names)} people")

X = lfw.data
y = lfw.target
images = lfw.images
h, w = images.shape[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_components = min(150, X_train.shape[0] - 1, X_train.shape[1])
pipe = Pipeline([
    ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)),
])

pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"Accuracy: {acc:.3f}")

pca = pipe.named_steps["pca"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(images[0], cmap="gray")
axes[0].set_title("Sample face")
axes[0].axis("off")
for i in range(3):
    axes[i + 1].imshow(pca.components_[i].reshape(h, w), cmap="gray")
    axes[i + 1].set_title(f"Eigenface {i}")
    axes[i + 1].axis("off")
fig.suptitle(f"LFW eigenfaces + MLP  accuracy={acc:.3f}")
out = Path(__file__).resolve().parent / "FacialRecognitionEigenfaces.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
