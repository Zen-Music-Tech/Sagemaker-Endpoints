from dotenv import load_dotenv
import boto3
import os

# Load .env file
load_dotenv()

# Get AWS credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

# Initialize S3 and SageMaker clients
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region,
)

sagemaker = boto3.client(
    "sagemaker",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region,
)

# # Test S3 upload
# s3.upload_file("model.tar.gz", "tracks-dev", "model.tar.gz")

# Test SageMaker endpoint listing
response = sagemaker.list_endpoints()
print("SageMaker Endpoints:", response)
