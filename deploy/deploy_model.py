from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker import Session
import os
from dotenv import load_dotenv
import boto3
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from .env
ROLE = os.getenv("AWS_ROLE")
TEXT_MODEL_S3_PATH = os.getenv("TEXT_MODEL_S3_PATH")
AUDIO_MODEL_S3_PATH = os.getenv("AUDIO_MODEL_S3_PATH")
TEXT_ENDPOINT_NAME = os.getenv("TEXT_ENDPOINT_NAME", "embedd-text")
AUDIO_ENDPOINT_NAME = os.getenv("AUDIO_ENDPOINT_NAME", "embedd-audio")
FRAMEWORK_VERSION = os.getenv("FRAMEWORK_VERSION", "2.2")
PY_VERSION = os.getenv("PY_VERSION", "py310")
SERVERLESS_MEMORY = int(os.getenv("SERVERLESS_MEMORY", 4096))


print("memore", SERVERLESS_MEMORY)
SERVERLESS_CONCURRENCY = int(os.getenv("SERVERLESS_CONCURRENCY", 5))

# Create ServerlessInferenceConfig object
SERVERLESS_CONFIG = ServerlessInferenceConfig(
    memory_size_in_mb=SERVERLESS_MEMORY,
    max_concurrency=SERVERLESS_CONCURRENCY,
)

def deploy_model(model_data, entry_point, endpoint_name):
    """
    Deploy a model to SageMaker with serverless inference.
    """
    logger.info(f"Deploying model {entry_point} to endpoint {endpoint_name}...")
    model = PyTorchModel(
        model_data=model_data,
        role=ROLE,
        entry_point=entry_point,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
    )
    predictor = model.deploy(
        serverless_inference_config=SERVERLESS_CONFIG,
        endpoint_name=endpoint_name,
    )
    logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
    return predictor


def delete_endpoint(endpoint_name):
    """
    Delete a SageMaker endpoint to stop incurring costs.
    """
    logger.info(f"Deleting endpoint: {endpoint_name}...")
    session = Session()
    try:
        session.delete_endpoint(endpoint_name)
        logger.info(f"Endpoint {endpoint_name} deleted.")
    except Exception as e:
        logger.error(f"Error deleting endpoint {endpoint_name}: {e}")


def check_endpoint_status(endpoint_name):
    """
    Check the status of a SageMaker endpoint.
    """
    logger.info(f"Checking status of endpoint: {endpoint_name}...")
    client = boto3.client("sagemaker")
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        logger.info(f"Endpoint {endpoint_name} is in status: {status}")
    except client.exceptions.ClientError as e:
        logger.error(f"Error checking status for endpoint {endpoint_name}: {e}")


def main(action):
    """
    Main function to handle deploy, delete, or status actions.
    """
    if action == "deploy":
        # Deploy text and audio models
        deploy_model(TEXT_MODEL_S3_PATH, "embed_text.py", TEXT_ENDPOINT_NAME)
        deploy_model(AUDIO_MODEL_S3_PATH, "embed_audio.py", AUDIO_ENDPOINT_NAME)

    elif action == "delete":
        # Delete endpoints
        delete_endpoint(TEXT_ENDPOINT_NAME)
        delete_endpoint(AUDIO_ENDPOINT_NAME)

    elif action == "status":
        # Check endpoint statuses
        check_endpoint_status(TEXT_ENDPOINT_NAME)
        check_endpoint_status(AUDIO_ENDPOINT_NAME)

    else:
        logger.error("Invalid action. Use 'deploy', 'delete', or 'status'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Usage: python deploy_model.py <action>")
    else:
        main(sys.argv[1])
