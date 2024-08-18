""" SerializeImageData Function:
This Lambda function handles data preparation by copying an image object from an S3 bucket, 
encoding the image in base64 format, and then returning the encoded image data to the Step Function.
 The serialized image data is included as image_data in the event output.
"""
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Check if the necessary keys are in the event
    if 's3_key' not in event or 's3_bucket' not in event:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Missing 's3_key' or 's3_bucket' in event"})
        }
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    try:
        # Download the data from s3 to /tmp/image.png
        s3.download_file(bucket, key, '/tmp/image.png')
        
        # Read the data from the file and encode it in base64
        with open("/tmp/image.png", "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Pass the data back to the Step Function
        return {
            'statusCode': 200,
            'body': json.dumps({
                "image_data": image_data,
                "s3_bucket": bucket,
                "s3_key": key,
                "inferences": []
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
    

#------------------------------------------
""""
Image Classification Lambda Function:
This function processes the base64-encoded image data received from the first Lambda function (SerializeImageData).
\ It decodes the image, performs classification, and sends the inference results back to the Step Function.
"""
import os
import io
import boto3
import json
import base64

#from sagemaker.serializers import IdentitySerializer
#from sagemaker.predictor import Predictor
runtime=boto3.client('runtime.sagemaker')
# Fill this in with the name of your deployed model
ENDPOINT = "image-classifier-endpoint"  # Replace with your actual endpoint name

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])  # Decoding the image from base64
    response=runtime.invoke_endpoint(EndpointName=ENDPOINT,ContentType="image/png",Body=image)

    # Instantiate a Predictor
    #predictor = Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    #predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction
    #inferences = predictor.predict(image)
    inferences=json.loads(response['Body'].read().decode('utf_8'))
    # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
    event["inferences"]=inferences
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
#------------------------------------------------------


""""
    The third function is responsible for filtering out low-confidence inferences.
   It takes the inferences from the Lambda 2 function output and filters low-confidence inferences
   "above a certain threshold indicating success"
 """
import json

THRESHOLD = 0.9

def lambda_handler(event, context):
    
    # Parse the body from the event if it's a string
    body = event.get('body')
    if isinstance(body, str):
        body = json.loads(body)
    
    # Grab the inferences from the event
    inferences = body['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(x > THRESHOLD for x in inferences)
    
    # Log the inferences
    print(f"Inferences: {inferences}")
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
