swagger_config = {
    "openapi": "3.0.3",
    "info": {
        "title": "Production Sentiment Analysis API",
        "description": (
            "Advanced sentiment analysis API with machine learning optimization"
        ),
        "version": "2.0.0",
        "contact": {"email": "admin@sentimentapi.com"},
    },
    "servers": [
        {
            "url": "https://h405sd8nkc.execute-api.us-east-1.amazonaws.com", 
            "description": "Production AWS Lambda server"
        },
        {
            "url": "http://localhost:5002", 
            "description": "Local development server"
        },
        {
            "url": "http://127.0.0.1:5002", 
            "description": "Local loopback server"
        }
    ],
    "components": {
        "schemas": {
            "PredictionRequest": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "example": "This product is absolutely amazing!",
                        "description": "Text to analyze for sentiment"
                    }
                }
            },
            "PredictionResponse": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "example": "1",
                        "description": "Predicted sentiment class"
                    },
                    "confidence": {
                        "type": "number",
                        "format": "float",
                        "example": 0.95,
                        "description": "Prediction confidence score"
                    },
                    "model_version": {
                        "type": "string",
                        "example": "20250919_182637",
                        "description": "Model version identifier"
                    },
                    "probabilities": {
                        "type": "object",
                        "description": "Probability distribution across all classes",
                        "example": {
                            "0": 0.05,
                            "1": 0.95,
                            "2": 0.02
                        }
                    }
                }
            },
            "BatchPredictionRequest": {
                "type": "object",
                "required": ["texts"],
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "example": [
                            "Great product!",
                            "Terrible service",
                            "It's okay"
                        ],
                        "description": "Array of texts to analyze"
                    }
                }
            },
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "example": "healthy"
                    },
                    "model_loaded": {
                        "type": "boolean",
                        "example": true  # Changed from True to true
                    },
                    "model_path": {
                        "type": "string",
                        "example": "output/production_sentiment_model_20250919_182637.pkl"
                    },
                    "requests_served": {
                        "type": "integer",
                        "example": 1250
                    },
                    "model_version": {
                        "type": "string",
                        "example": "20250919_182637"
                    },
                    "uptime": {
                        "type": "string",
                        "example": "Available"
                    }
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "example": "No text provided"
                    }
                }
            }
        }
    },
    "tags": [
        {"name": "Prediction", "description": "Sentiment prediction operations"},
        {"name": "Monitoring", "description": "System monitoring and statistics"},
        {"name": "System", "description": "System health and information"},
    ],
    "paths": {
        "/predict": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Predict sentiment from text",
                "description": "Analyzes text and returns sentiment prediction with confidence scores",
                "operationId": "predictSentiment",
                "requestBody": {
                    "required": true,  # Changed from True to true
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/PredictionRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful prediction",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictionResponse"
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "Bad request - no text provided",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/batch-predict": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Batch sentiment prediction",
                "description": "Analyzes multiple texts in a single request for efficient processing",
                "operationId": "batchPredict",
                "requestBody": {
                    "required": true,  # Changed from True to true
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/BatchPredictionRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful batch prediction",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "predictions": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/PredictionResponse"
                                            }
                                        },
                                        "count": {
                                            "type": "integer",
                                            "example": 3,
                                            "description": "Number of predictions made"
                                        }
                                    }
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "Bad request",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    }
                },
            }
        },
        "/model-info": {
            "get": {
                "tags": ["Monitoring"],
                "summary": "Get model information",
                "description": "Returns comprehensive model metadata and performance metrics",
                "operationId": "modelInfo",
                "responses": {
                    "200": {
                        "description": "Model information retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_metadata": {
                                            "type": "object",
                                            "description": "Complete model metadata"
                                        },
                                        "requests_served": {
                                            "type": "integer",
                                            "example": 1250
                                        },
                                        "available_classes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "example": ["0", "1", "2"]
                                        },
                                        "model_performance": {
                                            "type": "object",
                                            "properties": {
                                                "test_accuracy": {"type": "number"},
                                                "test_f1_macro": {"type": "number"},
                                                "validation_f1": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
            }
        },
        "/stats": {
            "get": {
                "tags": ["Monitoring"],
                "summary": "Get prediction statistics",
                "description": "Returns usage statistics and prediction trends for monitoring",
                "operationId": "predictionStats",
                "responses": {
                    "200": {
                        "description": "Statistics retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_predictions": {
                                            "type": "integer",
                                            "example": 1250
                                        },
                                        "sentiment_distribution": {
                                            "type": "object",
                                            "example": {"0": 450, "1": 300, "2": 500}
                                        },
                                        "average_confidence": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 0.87
                                        },
                                        "recent_predictions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "timestamp": {"type": "string"},
                                                    "prediction": {"type": "string"},
                                                    "confidence": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
            }
        },
        "/health": {
            "get": {
                "tags": ["System"],
                "summary": "Health check",
                "description": "Returns comprehensive system health status and model availability",
                "operationId": "healthCheck",
                "responses": {
                    "200": {
                        "description": "Health status retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthResponse"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api-info": {
            "get": {
                "tags": ["System"],
                "summary": "API information",
                "description": "Returns API metadata and available endpoints",
                "operationId": "apiInfo",
                "responses": {
                    "200": {
                        "description": "API information retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "version": {"type": "string"},
                                        "description": {"type": "string"},
                                        "endpoints": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
}
