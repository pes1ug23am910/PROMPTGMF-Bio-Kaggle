# Deployment Guide

This guide covers various deployment options for PromptGFM-Bio in production or research environments.

---

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [API Deployment](#api-deployment)
5. [Batch Inference](#batch-inference)

---

## Local Deployment

### Development Environment

For local development and experimentation:

```bash
# 1. Install in development mode
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (API keys, paths, etc.)

# 3. Download data
python scripts/download_data.py --all

# 4. Preprocess
python scripts/preprocess_all.py

# 5. Train
python scripts/train.py --config configs/base_config.yaml
```

### Production Environment

For production deployment on a local server:

```bash
# 1. Use production configuration
pip install -r requirements.txt --no-dev

# 2. Set production environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# 3. Run with optimized settings
python scripts/train.py \
    --config configs/base_config.yaml \
    --fp16 \
    --gradient_checkpointing
```

---

## Docker Deployment

### Building Docker Image

Create a `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric
RUN pip install torch-geometric==2.4.0 && \
    pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for API (if using)
EXPOSE 8000

CMD ["python", "scripts/train.py", "--config", "configs/base_config.yaml"]
```

### Build and Run

```bash
# Build image
docker build -t promptgfm-bio:latest .

# Run training
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    promptgfm-bio:latest

# Run inference
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    promptgfm-bio:latest \
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Docker Compose

For multi-container setup (training + monitoring):

```yaml
# docker-compose.yml
version: '3.8'

services:
  training:
    build: .
    image: promptgfm-bio:latest
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python scripts/train.py --config configs/base_config.yaml

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
    command: tensorboard --logdir=/logs --host=0.0.0.0
```

Usage:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS (Amazon Web Services)

#### EC2 Instance

1. **Launch Deep Learning AMI**
   ```bash
   aws ec2 run-instances \
       --image-id ami-0c55b159cbfafe1f0 \
       --instance-type p3.2xlarge \
       --key-name your-key-pair \
       --security-group-ids sg-xxxxxxxx
   ```

2. **Setup instance**
   ```bash
   ssh -i your-key.pem ec2-user@instance-ip
   
   git clone https://github.com/yourusername/PromptGFM-Bio.git
   cd PromptGFM-Bio
   
   conda activate pytorch
   pip install -r requirements.txt
   ```

3. **Transfer data** (use S3):
   ```bash
   aws s3 sync s3://your-bucket/data/ ./data/
   ```

#### SageMaker

Create a training job:

```python
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='scripts',
    role=sagemaker_role,
    framework_version='2.1.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    hyperparameters={
        'config': 'configs/base_config.yaml',
        'epochs': 100
    }
)

estimator.fit({'training': 's3://your-bucket/data/'})
```

### Google Cloud Platform (GCP)

#### Vertex AI

```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=promptgfm-training \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_V100,accelerator-count=1,replica-count=1,container-image-uri=gcr.io/your-project/promptgfm-bio:latest
```

### Azure

#### Azure Machine Learning

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget

ws = Workspace.from_config()
compute_target = ComputeTarget(workspace=ws, name='gpu-cluster')

env = Environment.from_conda_specification(
    name='promptgfm-env',
    file_path='environment.yml'
)

config = ScriptRunConfig(
    source_directory='.',
    script='scripts/train.py',
    arguments=['--config', 'configs/base_config.yaml'],
    compute_target=compute_target,
    environment=env
)

experiment = Experiment(ws, 'promptgfm-training')
run = experiment.submit(config)
```

---

## API Deployment

### FastAPI Inference Server

Create `api/server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from typing import List

from src.models.promptgfm import PromptGFM
from src.data.preprocess import load_graph

app = FastAPI(title="PromptGFM-Bio API")

# Load model and graph on startup
@app.on_event("startup")
async def load_model():
    global model, graph
    model = PromptGFM.load_from_checkpoint('checkpoints/best_model.pt')
    model.eval()
    graph = torch.load('data/processed/biomedical_graph.pt')

class PredictionRequest(BaseModel):
    disease_description: str
    candidate_genes: List[str]

class PredictionResponse(BaseModel):
    genes: List[str]
    scores: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        with torch.no_grad():
            scores = model.predict(
                disease_description=request.disease_description,
                candidate_genes=request.candidate_genes,
                graph=graph
            )
        
        # Sort by score
        ranked = sorted(
            zip(request.candidate_genes, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        
        genes, scores = zip(*ranked)
        return PredictionResponse(genes=list(genes), scores=list(scores))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Run API Server

```bash
# Development
uvicorn api.server:app --reload --port 8000

# Production
gunicorn api.server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Deploy with Docker

```dockerfile
FROM promptgfm-bio:latest

COPY api/ ./api/

RUN pip install fastapi uvicorn gunicorn

EXPOSE 8000

CMD ["gunicorn", "api.server:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promptgfm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: promptgfm-api
  template:
    metadata:
      labels:
        app: promptgfm-api
    spec:
      containers:
      - name: api
        image: promptgfm-bio:api
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: promptgfm-service
spec:
  type: LoadBalancer
  selector:
    app: promptgfm-api
  ports:
  - port: 80
    targetPort: 8000
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## Batch Inference

### Large-Scale Prediction

For processing many diseases at once:

```python
# scripts/batch_inference.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.promptgfm import PromptGFM
from src.data.dataset import InferenceDataset

def batch_inference(
    model_checkpoint: str,
    input_file: str,
    output_file: str,
    batch_size: int = 32
):
    # Load model
    model = PromptGFM.load_from_checkpoint(model_checkpoint)
    model.eval()
    model = model.cuda()
    
    # Load dataset
    dataset = InferenceDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # Run inference
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.cuda()
            predictions = model(batch)
            all_predictions.append(predictions.cpu())
    
    # Save results
    torch.save(torch.cat(all_predictions), output_file)

if __name__ == "__main__":
    batch_inference(
        model_checkpoint='checkpoints/best_model.pt',
        input_file='data/inference_queries.csv',
        output_file='results/predictions.pt'
    )
```

### Distributed Inference

For very large datasets, use multiple GPUs:

```python
# Use DataParallel or DistributedDataParallel
model = torch.nn.DataParallel(model)

# Or with torchrun
# torchrun --nproc_per_node=4 scripts/batch_inference.py
```

---

## Performance Optimization

### Inference Optimization

1. **TorchScript Compilation**
   ```python
   model.eval()
   scripted_model = torch.jit.script(model)
   scripted_model.save("model_scripted.pt")
   ```

2. **ONNX Export**
   ```python
   torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       export_params=True,
       opset_version=14
   )
   ```

3. **TensorRT (NVIDIA)**
   ```python
   import torch_tensorrt
   
   compiled_model = torch_tensorrt.compile(
       model,
       inputs=[torch_tensorrt.Input(shape=[1, 512])],
       enabled_precisions={torch.float, torch.half}
   )
   ```

### Caching Strategies

```python
# Cache prompt embeddings for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_prompt_embedding(disease_description: str):
    return model.prompt_encoder(disease_description)
```

---

## Monitoring

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inference.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
request_count = Counter('predictions_total', 'Total predictions')
request_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@request_duration.time()
def predict(disease_description, genes):
    request_count.inc()
    return model.predict(disease_description, genes)

# Start metrics server
start_http_server(9090)
```

---

## Security Considerations

1. **Authentication**: Implement API key authentication
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Sanitize disease descriptions
4. **HTTPS**: Use SSL/TLS for API endpoints
5. **Model Versioning**: Track which model version is deployed

---

## Backup and Recovery

```bash
# Backup checkpoints to S3
aws s3 sync checkpoints/ s3://backup-bucket/checkpoints/

# Backup data
aws s3 sync data/processed/ s3://backup-bucket/data/

# Restore
aws s3 sync s3://backup-bucket/checkpoints/ checkpoints/
```

---

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision (FP16)

2. **API Timeout**
   - Increase worker count
   - Optimize model (TorchScript, quantization)
   - Cache embeddings

3. **Cold Start Latency**
   - Keep model loaded in memory
   - Use serverless with provisioned concurrency
   - Implement warm-up endpoint

---

## Cost Optimization

### Cloud Cost Reduction
- Use spot/preemptible instances
- Scale down during off-hours
- Optimize instance types
- Use reserved instances for production

### Storage Optimization
- Compress checkpoints
- Delete old training runs
- Use lifecycle policies for S3

---

For additional deployment questions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
