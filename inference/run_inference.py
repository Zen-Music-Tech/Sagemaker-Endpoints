from sagemaker.predictor import Predictor
from dotenv import load_dotenv
import os
import json
import logging
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load unified endpoint name from .env
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "multi-model-endpoint")


def query_endpoint(predictor, payload, input_type="json"):
    """
    Query the unified SageMaker endpoint with the given payload.
    """
    try:
        predictor.content_type = f"application/{input_type}"
        logger.info(f"Payload sent to endpoint {predictor.endpoint_name}: {json.dumps(payload)}")
        
        start_time = time.time()
        logger.info(f"Sending request to endpoint: {predictor.endpoint_name}")
        response = predictor.predict(json.dumps(payload))
        end_time = time.time()
        
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Raw response from endpoint: {response}")
        
        # Deserialize and return JSON response
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error querying endpoint {predictor.endpoint_name}: {e}")
        return None


def infer_text(texts):
    """
    Perform inference on text inputs using the unified endpoint.
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    
    logger.info(f"Preparing payload for text inference: {texts}")
    payload = {"texts": texts}
    unified_predictor = Predictor(endpoint_name=ENDPOINT_NAME)
    
    logger.info(f"Initialized predictor for endpoint: {ENDPOINT_NAME}")
    return query_endpoint(unified_predictor, payload)


def infer_audio(file_urls):
    """
    Perform inference on audio inputs using the unified endpoint.
    """
    if not isinstance(file_urls, list):
        raise ValueError("Input file URLs must be a list of strings.")
    
    logger.info(f"Preparing payload for audio inference: {file_urls}")
    payload = {"fileUrls": file_urls}
    unified_predictor = Predictor(endpoint_name=ENDPOINT_NAME)
    
    logger.info(f"Initialized predictor for endpoint: {ENDPOINT_NAME}")
    return query_endpoint(unified_predictor, payload)


def main():
    # Example: Text inference
    texts = ["hello world", "this is a test"]
    logger.info(f"Running inference on texts: {texts}")
    text_results = infer_text(texts)
    logger.info(f"Text inference results: {text_results}")

    # Example: Audio inference
    file_urls = [
        "s3://tracks-dev/Ed Sheeran-Shape of You.mp3",
        "s3://tracks-dev/Ed Sheeran-Shape of You.mp3"
    ]
    logger.info(f"Running inference on audio files: {file_urls}")
    audio_results = infer_audio(file_urls)
    logger.info(f"Audio inference results: {audio_results}")


if __name__ == "__main__":
    main()
