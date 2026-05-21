import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import uuid
from chromadb import HttpClient
import subprocess
import atexit
import time
import requests


class VisualRAP:
    def __init__(self, storage_path="visual_memory",
                 model_name="openai/clip-vit-base-patch32",
                 device=None,
                 chroma_host="localhost",
                 chroma_port=8001,
                 auto_start_server=True):
        """Initialize Visual RAG with CLIP and ChromaDB.

        Args:
            storage_path: Path to persistent ChromaDB storage.
            model_name: CLIP model name.
            device: Torch device override.
            chroma_host: ChromaDB host.
            chroma_port: ChromaDB port.
            auto_start_server: Whether to auto-start ChromaDB.

        Returns:
            None
        """
        
        self.storage_path = os.path.abspath(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.server_process = None
        self._owns_server = False

        # Start ChromaDB server if needed
        if auto_start_server and not self._is_server_running():
            self._start_chroma_server()
            atexit.register(self.close)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.embeddings_dim = self.model.visual_projection.out_features

        self.client = HttpClient(host=chroma_host, port=chroma_port)

        self.collection = self.client.get_or_create_collection(name="visual_rag")
    
    def _is_server_running(self):
        """Check if ChromaDB server is already running.

        Args:
            None

        Returns:
            True if server responds to heartbeat, else False.
        """
        base_url = f"http://{self.chroma_host}:{self.chroma_port}"
        heartbeat_paths = ("/api/v1/heartbeat", "/api/v2/heartbeat")
        for path in heartbeat_paths:
            try:
                response = requests.get(f"{base_url}{path}", timeout=1)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                continue
            except Exception:
                continue
        return False

    def _start_chroma_server(self):
        """Start ChromaDB server as a background process.

        Args:
            None

        Returns:
            None
        """
        print(f"Starting ChromaDB server on port {self.chroma_port}...")
        self.server_process = subprocess.Popen(
            ["chroma", "run", "--port", str(self.chroma_port), "--path", self.storage_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        self._owns_server = True
        
        # Wait for server to be ready
        max_wait = 10
        for _ in range(max_wait):
            if self._is_server_running():
                print(f"ChromaDB server started successfully on port {self.chroma_port}")
                return
            time.sleep(1)
        
        raise RuntimeError(f"Failed to start ChromaDB server on port {self.chroma_port}")
    
    def close(self):
        """Cleanup: stop the server if we started it.

        Args:
            None

        Returns:
            None
        """
        if self._owns_server and self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                try:
                    self.server_process.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self.server_process = None
                self._owns_server = False

    def __del__(self):
        """Destructor: best-effort cleanup."""
        try:
            self.close()
        except Exception:
            pass

    def embed_image(self, image: Image.Image):
        """Compute a CLIP embedding for an image.

        Args:
            image: PIL image.

        Returns:
            1D float32 embedding vector.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.vision_model(**inputs)
            img_emb = self.model.visual_projection(image_features.pooler_output)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().numpy().astype("float32")[0]

    def add_image(self, image, label):
        """Add an image and label to the ChromaDB collection.

        Args:
            image: PIL image.
            label: Label string.

        Returns:
            None
        """
        emb = self.embed_image(image)
        self.collection.add(
            embeddings=[emb.tolist()],
            documents=[str(label).strip()],
            ids=[str(uuid.uuid4())]
        )
        
    def query(self, image, top_k=3, threshold=0.3):
        """Query for the closest label to an image embedding.

        Args:
            image: PIL image.
            top_k: Number of nearest neighbors.
            threshold: Distance threshold for unknown classification.

        Returns:
            Tuple of (label, distance).
        """
        emb = self.embed_image(image)
        results = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=top_k
        )

        if not results["documents"][0]:
            return "unknown", 1.0

        best_distance = float(results["distances"][0][0])
        best_label = str(results["documents"][0][0]).strip()

        if best_distance > threshold:
            return "unknown", best_distance
        return best_label, best_distance
