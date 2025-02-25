import boto3
import sagemaker
from sagemaker.estimator import Estimator

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define your fine-tuning job parameters
role = 'arn:aws:iam::334061574913:role/VGI-AIML-SMK-SageMaker-SMStudioTeamExRole-Hackathon-2025-us-east-1'  # Your IAM role ARN
bucket = 'vgi-gis-prod-us-east-1-hackathon-2025-sandbox'
prefix = 'fine-tune-data'

# Create a dummy dataset and upload it to S3
dummy_data = 'dummy data'
s3 = boto3.client('s3')
s3.put_object(Bucket=bucket, Key=f'{prefix}/dummy-dataset.jsonl', Body=dummy_data)

# Define the estimator
estimator = Estimator(
    image_uri='anthropic.claude-3-haiku-20240307-v1:0',  # Your model image URI
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    volume_size=50,
    max_run=86400,
    input_mode='File',
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sagemaker_session
)

# Set hyperparameters
estimator.set_hyperparameters(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5
)

# Define the input data configuration
train_data = sagemaker.inputs.TrainingInput(
    s3_data=f's3://{bucket}/{prefix}/dummy-dataset.jsonl',
    content_type='application/jsonlines'
)

# Launch the training job
training_job_name = 'fine-tuning-job'  # Ensure this name follows the required pattern
estimator.fit({'train': train_data}, job_name=training_job_name)

print("Fine-tuning job created")
