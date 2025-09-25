# filepath: lambda_adapter.py
import json
import base64
import os
from io import StringIO
import sys

# Set environment for Lambda
os.environ['FLASK_ENV'] = 'production'
os.environ['NLTK_DATA'] = '/tmp/nltk_data'

try:
    from production_api import app
except ImportError as e:
    print(f"Import error: {e}")
    app = None


def lambda_handler(event, context):
    """AWS Lambda handler for Flask app"""
    
    print(f"Event: {json.dumps(event, default=str)}")
    
    if not app:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Flask app failed to initialize"})
        }
    
    # Handle API Gateway event
    if "requestContext" in event:
        try:
            # Extract request data
            http_method = event.get("httpMethod", "GET")
            path = event.get("path", "/")
            headers = event.get("headers", {})
            body = event.get("body", "")
            query_params = event.get("queryStringParameters") or {}
            
            print(f"Processing {http_method} {path}")

            # Handle base64 encoded body
            if event.get("isBase64Encoded", False) and body:
                try:
                    body = base64.b64decode(body).decode("utf-8")
                except Exception as e:
                    print(f"Base64 decode error: {e}")
                    body = ""

            # Create Flask test request
            with app.test_client() as client:
                # Build query string
                query_string = ""
                if query_params:
                    query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
                
                # Make request to Flask app
                response = client.open(
                    path=path,
                    method=http_method,
                    headers=headers,
                    data=body if body else None,
                    query_string=query_string,
                    content_type=headers.get("content-type", "application/json")
                )

                # Get response body
                response_body = response.get_data(as_text=True)
                
                print(f"Flask response status: {response.status_code}")
                print(f"Flask response body: {response_body[:200]}...")

                return {
                    "statusCode": response.status_code,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type,Authorization,Accept",
                    },
                    "body": response_body,
                }

        except Exception as e:
            print(f"Lambda handler error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Internal server error",
                    "message": str(e),
                    "type": type(e).__name__
                }),
            }

    # Handle direct Lambda invocation
    else:
        try:
            # Test if the app works
            with app.test_client() as client:
                health_response = client.get('/health')
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": "Sentiment Analysis API is running",
                        "version": "2.0.0",
                        "health_check": health_response.get_data(as_text=True)
                    }),
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": "Direct invocation failed",
                    "message": str(e)
                }),
            }
