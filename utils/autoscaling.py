import boto3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enable_autoscaling(endpoint_name, min_capacity, max_capacity, target_invocations):
    """
    Enable autoscaling for a SageMaker endpoint.
    """
    autoscaling_client = boto3.client("application-autoscaling")
    
    autoscaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )
    logger.info(f"Scalable target registered for endpoint: {endpoint_name}")
    
  
    autoscaling_client.put_scaling_policy(
        PolicyName="ScaleOnInvocations",
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": target_invocations,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleOutCooldown": 60,
            "ScaleInCooldown": 60,
        },
    )
    logger.info(f"Autoscaling policy configured for endpoint: {endpoint_name}")


if __name__ == "__main__":
    enable_autoscaling(endpoint_name="embedd-text", min_capacity=1, max_capacity=5, target_invocations=50.0)
    enable_autoscaling(endpoint_name="embedd-audio", min_capacity=1, max_capacity=5, target_invocations=50.0)
