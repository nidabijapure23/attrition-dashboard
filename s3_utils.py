import boto3
import pandas as pd
from io import BytesIO

def load_excel_from_s3(bucket, key, aws_access_key_id, aws_secret_access_key, region_name):
    """Load an Excel file from S3 and return as a pandas DataFrame."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_excel(BytesIO(obj['Body'].read()))
    return df

def save_excel_to_s3(df, bucket, key, aws_access_key_id, aws_secret_access_key, region_name):
    """Save a pandas DataFrame as an Excel file to S3."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    out_buffer = BytesIO()
    df.to_excel(out_buffer, index=False)
    out_buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=out_buffer.getvalue()) 