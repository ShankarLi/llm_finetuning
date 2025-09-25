# filepath: lambda_adapter.py
import json
import base64
from production_api import app


def lambda_handler(event, context):
    """AWS Lambda handler for Flask app"""

    # Handle API Gateway event
    if "httpMethod" in event:
        # Extract request data
        method = event["httpMethod"]
        path = event.get("path", "/")
        headers = event.get("headers", {})
        body = event.get("body", "")

        # Handle base64 encoded body
        if event.get("isBase64Encoded", False):
            body = base64.b64decode(body).decode("utf-8")

        # Create WSGI environ
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": event.get("queryStringParameters", ""),
            "CONTENT_TYPE": headers.get("Content-Type", ""),
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": body,
            "wsgi.url_scheme": "https",
            "HTTP_HOST": headers.get("Host", "localhost"),
        }

        # Add all headers
        for key, value in headers.items():
            key = key.upper().replace("-", "_")
            if key not in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                environ[f"HTTP_{key}"] = value

        # Mock start_response
        response_data = {}

        def start_response(status, response_headers):
            response_data["status"] = int(status.split()[0])
            response_data["headers"] = dict(response_headers)

        # Call Flask app
        with app.request_context(environ):
            try:
                response = app.full_dispatch_request()

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
        "body": json.dumps({"message": "Sentiment Analysis API is running"}),
    }
