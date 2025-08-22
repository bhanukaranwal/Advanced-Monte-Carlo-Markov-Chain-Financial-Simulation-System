"""
API middleware for MCMF system
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import logging

from ..utils.exceptions import MCMFException
from ..utils.logging_config import MCMFLogger

logger = MCMFLogger("middleware")

class TimingMiddleware:
    """Request timing middleware"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    process_time = time.time() - start_time
                    message["headers"].append([
                        b"x-process-time",
                        str(process_time).encode()
                    ])
                await send(message)
                
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

class RequestIDMiddleware:
    """Request ID middleware"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())
            scope["request_id"] = request_id
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    message["headers"].append([
                        b"x-request-id",
                        request_id.encode()
                    ])
                await send(message)
                
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

class ErrorHandlingMiddleware:
    """Global error handling middleware"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                await send(message)
                
            try:
                await self.app(scope, receive, send_wrapper)
            except MCMFException as e:
                logger.log_error(e, {"request_path": scope.get("path")})
                response = JSONResponse(
                    status_code=400,
                    content={
                        "error": e.__class__.__name__,
                        "message": e.message,
                        "error_code": e.error_code
                    }
                )
                await response(scope, receive, send)
            except Exception as e:
                logger.log_error(e, {"request_path": scope.get("path")})
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": "InternalServerError",
                        "message": "An unexpected error occurred"
                    }
                )
                await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Request logging middleware"""
    start_time = time.time()
    
    logger.log_api_request(
        endpoint=str(request.url.path),
        method=request.method,
        user_id=getattr(request.state, 'user_id', None)
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
