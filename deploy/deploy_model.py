from sagemaker.pytorch import PyTorchModel
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
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")  # Default instance type

def delete_endpoint_config(endpoint_config_name):
    """
    Delete an existing SageMaker endpoint configuration.
    """
    logger.info(f"Deleting endpoint configuration: {endpoint_config_name}...")
    client = boto3.client("sagemaker")
    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        logger.info(f"Deleted endpoint configuration: {endpoint_config_name}")
    except client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" in str(e):
            logger.warning(f"Endpoint configuration {endpoint_config_name} does not exist.")
        else:
            logger.error(f"Error deleting endpoint configuration {endpoint_config_name}: {e}")


def deploy_model(model_data, entry_point, endpoint_name, instance_type):
    """
    Deploy a model to SageMaker with instance-based inference.
    """
    # Delete existing endpoint configuration
    delete_endpoint_config(endpoint_name)

    logger.info(f"Deploying model {entry_point} to endpoint {endpoint_name} with instance type {instance_type}...")
    model = PyTorchModel(
        model_data=model_data,
        role=ROLE,
        entry_point=entry_point,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
    )
    predictor = model.deploy(
        initial_instance_count=1,  # Number of instances
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
    return predictor


def delete_endpoint(endpoint_name):
    """
    Delete a SageMaker endpoint and its configuration to stop incurring costs.
    """
    logger.info(f"Deleting endpoint: {endpoint_name}...")
    session = Session()
    client = boto3.client("sagemaker")

    # Delete the endpoint
    try:
        session.delete_endpoint(endpoint_name)
        logger.info(f"Endpoint {endpoint_name} deleted.")
    except client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            logger.warning(f"Endpoint {endpoint_name} does not exist.")
        else:
            logger.error(f"Error deleting endpoint {endpoint_name}: {e}")

    # Delete the endpoint configuration
    delete_endpoint_config(endpoint_name)


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
        if "Could not find endpoint" in str(e):
            logger.warning(f"Endpoint {endpoint_name} does not exist.")
        else:
            logger.error(f"Error checking status for endpoint {endpoint_name}: {e}")


def main(action):
    """
    Main function to handle deploy, delete, or status actions.
    """
    instance_type = INSTANCE_TYPE  # Load instance type from .env

    if action == "deploy":
        # Deploy text and audio models
        deploy_model(TEXT_MODEL_S3_PATH, "embed_text.py", TEXT_ENDPOINT_NAME, instance_type)
        deploy_model(AUDIO_MODEL_S3_PATH, "embed_audio.py", AUDIO_ENDPOINT_NAME, instance_type)

    elif action == "delete":
        # Delete endpoints and configurations
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
