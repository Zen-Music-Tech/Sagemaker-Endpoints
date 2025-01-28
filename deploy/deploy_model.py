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
MODEL_S3_PATH = os.getenv("MODEL_S3_PATH")  # Combined model path in S3
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "multi-model-endpoint")  # Name of the MME
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


def deploy_multi_model_endpoint(model_data, entry_point, endpoint_name, instance_type):
    """
    Deploy a multi-model endpoint to SageMaker.
    """
    # Delete existing endpoint configuration
    delete_endpoint_config(endpoint_name)

    logger.info(f"Deploying multi-model endpoint {endpoint_name} with instance type {instance_type}...")

    # Define the PyTorch model for MME
    model = PyTorchModel(
        model_data=model_data,
        role=ROLE,
        entry_point=entry_point,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
    )

    # Deploy the multi-model endpoint
    predictor = model.deploy(
        initial_instance_count=1,  # Number of instances
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    logger.info(f"Multi-model endpoint deployed successfully: {endpoint_name}")
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
        # Deploy the multi-model endpoint
        deploy_multi_model_endpoint(MODEL_S3_PATH, "embed_combined.py", ENDPOINT_NAME, instance_type)

    elif action == "delete":
        # Delete the endpoint and configuration
        delete_endpoint(ENDPOINT_NAME)

    elif action == "status":
        # Check endpoint status
        check_endpoint_status(ENDPOINT_NAME)

    else:
        logger.error("Invalid action. Use 'deploy', 'delete', or 'status'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Usage: python deploy_model.py <action>")
    else:
        main(sys.argv[1])
