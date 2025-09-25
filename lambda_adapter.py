# filepath: lambda_adapter.py
import json
import base64
from production_api import app


def lambda_handler(event, context):
    """AWS Lambda handler for Flask app"""
    
    # Handle API Gateway event
    if "requestContext" in event:
        # Extract request data
        method = event.get("httpMethod", "GET")
        path = event.get("path", "/")
        headers = event.get("headers", {})
        body = event.get("body", "")
        query_params = event.get("queryStringParameters") or {}

        # Handle base64 encoded body
        if event.get("isBase64Encoded", False) and body:
            body = base64.b64decode(body).decode("utf-8")

        # Create Flask test client and make request
        with app.test_client() as client:
            try:
                # Convert query params to string
                query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
                
                response = client.open(
                    path=path,
                    method=method,
                    headers=headers,
                    data=body,
                    query_string=query_string,
                    content_type=headers.get("Content-Type", "application/json")
                )

                return {
                    "statusCode": response.status_code,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type",
                    },
                    "body": response.get_data(as_text=True),
                }
            except Exception as e:
                return {
                    "statusCode": 500,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": str(e)}),
                }

    # Handle direct Lambda invocation
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Sentiment Analysis API is running",
            "version": "2.0.0"
        }),
    }
