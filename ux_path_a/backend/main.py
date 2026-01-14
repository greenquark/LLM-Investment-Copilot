"""
UX Path A Backend - FastAPI Application

This is the main entry point for the UX Path A backend server.
It provides the chat orchestrator API that integrates with OpenAI
and the Trading Copilot Platform tools.
"""

"""
UX Path A Backend - FastAPI Application

This is the main entry point for the UX Path A backend server.
It provides the chat orchestrator API that integrates with OpenAI
and the Trading Copilot Platform tools.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.api import chat, auth, health
from ux_path_a.backend.backend_core.config import settings

# Configure logging early
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable debug logging for specific modules if DEBUG is enabled
if settings.DEBUG:
    try:
        logging.getLogger("ux_path_a").setLevel(logging.DEBUG)
    except:
        logging.getLogger("backend").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)

# Create FastAPI app
app = FastAPI(
    title="UX Path A - Smart Trading Copilot API",
    description="Chat orchestrator for conversational market analysis",
    version="0.1.0",
)

# Simple test endpoint BEFORE middleware and routers
@app.get("/test-route-early")
async def test_route_early():
    """Simple test endpoint defined early."""
    print("🔍 TEST ROUTE EARLY CALLED!")
    return {"message": "Test route early works"}

# Route to list all registered routes - defined early
@app.get("/list-all-routes")
async def list_all_routes_early():
    """List all registered routes."""
    all_routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            all_routes.append({
                "path": route.path,
                "methods": getattr(route, 'methods', set()),
                "name": getattr(route, 'name', None)
            })
        elif hasattr(route, 'path_regex'):
            all_routes.append({"path_regex": str(route.path_regex)})
    return {"routes": all_routes, "total": len(all_routes)}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests for debugging."""
    import time
    import json
    import traceback
    
    
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    if settings.DEBUG:
        logger.debug(f"   Query params: {dict(request.query_params)}")
        # Log headers (but mask sensitive ones)
        headers_dict = dict(request.headers)
        if 'authorization' in headers_dict:
            headers_dict['authorization'] = 'Bearer ***'
        logger.debug(f"   Headers: {headers_dict}")
        
        # Log request body for POST/PUT (be careful with sensitive data)
        # Note: We can't read the body here as it's consumed, so we'll log it after processing
        if request.method in ["POST", "PUT", "PATCH"]:
            logger.debug(f"   Body: [Request body - check endpoint logs for details]")
    
    # Process request
    try:
        response = await call_next(request)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    return response

# #region agent log
# Verify middleware was registered
import json
try:
    middleware_info = {
        "has_middleware_stack": hasattr(app, "middleware_stack"),
    }
    if hasattr(app, "middleware_stack") and app.middleware_stack is not None:
        try:
            middleware_info["middleware_count"] = len(app.middleware_stack)
        except (TypeError, AttributeError):
            middleware_info["middleware_count"] = "unknown"
    else:
        middleware_info["middleware_count"] = 0
    with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"main.py:132","message":"middleware defined, checking app.middleware_stack","data":middleware_info,"timestamp":int(__import__('time').time()*1000)})+'\n')
except Exception as e:
    # Don't fail startup if logging fails
    pass
# #endregion

# Include routers
# #region agent log
import json
route_paths_before = [r.path for r in app.routes]
with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a', encoding='utf-8') as f:
    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"main.py:96","message":"before include_router","data":{"routes_before":route_paths_before,"health_router_routes":[r.path for r in health.router.routes]},"timestamp":int(__import__('time').time()*1000)})+'\n')
# #endregion

app.include_router(health.router, prefix="/api/health", tags=["health"])

# #region agent log
route_paths_after = [r.path for r in app.routes]
with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a', encoding='utf-8') as f:
    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"main.py:100","message":"after include_router health","data":{"routes_after":route_paths_after,"debug_route_exists":"/api/health/debug" in route_paths_after},"timestamp":int(__import__('time').time()*1000)})+'\n')
# #endregion

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

# Simple test endpoint to verify routing works
@app.get("/test-route")
async def test_route():
    """Simple test endpoint."""
    print("🔍 TEST ROUTE CALLED!")
    return {"message": "Test route works", "routes": [r.path for r in app.routes if hasattr(r, 'path')]}

# Route to list all registered routes (for debugging)
@app.get("/list-routes", name="list_routes")
async def list_routes():
    """List all registered routes for debugging."""
    all_routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            all_routes.append(route.path)
        elif hasattr(route, 'path_regex'):
            all_routes.append(str(route.path_regex))
    return {"routes": all_routes, "total": len(all_routes)}

# Custom 404 handler using a catch-all route approach
# FastAPI doesn't have a NotFound exception, so we'll handle it in middleware

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
    from ux_path_a.backend.backend_core.database import Base, engine
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
    
    logger.info("UX Path A Backend starting up...")
    logger.info(f"CORS origins: {settings.CORS_ORIGINS}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("UX Path A Backend shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
