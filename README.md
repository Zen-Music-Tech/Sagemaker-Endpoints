# SageMaker Endpoints

For deploying and managing serverless or instance-based AWS SageMaker endpoints for text and audio embedding models inference.

**Note: Aws Only allows upto 6 gb for sagemaker serverless inference**

## Overview

This project provides a toolkit(still updating some features) for:
- Deploying SageMaker endpoints for text and audio embedding models
- Running inference on deployed endpoints
- Monitoring endpoint status and performance
- Managing endpoint lifecycle

## Project Structure

```
SageMaker-Endpoints/
├── deploy/
│   ├── deploy_model.py          # Script to deploy and manage endpoints
├── inference/
│   ├── run_inference.py         # Script for inference requests
├── monitoring/
│   ├── monitor_endpoints.py     # Endpoint status monitoring utility
├── model/                       # Model files and entry points
│   ├── embed_audio.py           # Audio embedding entry point
│   ├── embed_text.py            # Text embedding entry point
│   ├── model.tar.gz             # Compressed model artifact
├── .env                         # Configuration environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Prerequisites

### AWS Requirements
- AWS Keys
- IAM Role with permissions:
  - `sagemaker:CreateModel`
  - `sagemaker:CreateEndpoint`
  - `s3:GetObject`
  - `s3:ListBucket`

### Development Environment
- Python 3.10+
- Required packages listed in `requirements.txt`

## Setup Guide

1. **Clone the Repository**
   ```bash
   git clone repo_url
   cd sagemaker-endpoints
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   
   Create a `.env` file in the project root:
   ```plaintext
   AWS_ROLE=your_arn_number
   TEXT_MODEL_S3_PATH=s3://your-bucket-name/model.tar.gz
   AUDIO_MODEL_S3_PATH=s3://your-bucket-name/model.tar.gz
   TEXT_ENDPOINT_NAME=embedd-text
   AUDIO_ENDPOINT_NAME=embedd-audio
   FRAMEWORK_VERSION=2.2
   PY_VERSION=py310
   SERVERLESS_MEMORY=16384  # Memory in MB (16 GB)
   SERVERLESS_CONCURRENCY=10  # Maximum concurrency
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
   ```

5. **Prepare Model Artifacts**
   ```bash
   tar -czvf model.tar.gz model_contents               # or put contents in a director and use : tar -czvf model.tar.gz -C directory_name .
   aws s3 cp model.tar.gz s3://your-bucket-name/
   ```

## Usage

### Endpoint Management

**Deploy Endpoints**
```bash
python deploy/deploy_model.py deploy
```

**Check Status**
```bash
python deploy/deploy_model.py status
```

**Delete Endpoints**
```bash
python deploy/deploy_model.py delete
```

### Running Inference

Execute the inference script:
```bash
python inference/run_inference.py
```

#### Example Inputs

Text Inference:
```json
{
  "texts": ["hello world", "SageMaker testing"]
}
```

Audio Inference:
```json
{
  "fileUrls": ["s3://your-bucket-name/audio1.mp3", "s3://your-bucket-name/audio2.mp3"]
}
```



## Monitoring

Monitor endpoint status:
```bash
python monitoring/monitor_endpoints.py
```

Logs are available in:
- AWS CloudWatch
- Local logs via Python's logging module

## Troubleshooting

### Common Issues

1. **ValidationException: Endpoint Configuration Exists**
   ```bash
   python deploy/deploy_model.py delete
   ```

2. **Memory Errors**
   - Increase `SERVERLESS_MEMORY` in `.env`
   - Or use larger instance type
  
     
3. **Credential Issues**
   ```bash
   aws sts get-caller-identity
   ```
