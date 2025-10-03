import asyncio
import aiohttp.web
import json
import importlib.util
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlgorithmServer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.algorithms: Dict[str, Any] = {}
        self.load_algorithms()

    def load_algorithms(self):
        """Dynamically load algorithm modules from the algorithm directory"""
        algorithm_base_path = Path('algorithm')

        if not algorithm_base_path.exists():
            logger.error(f"Algorithm directory not found: {algorithm_base_path}")
            return

        # Scan for algorithm directories
        for algo_dir in algorithm_base_path.iterdir():
            if algo_dir.is_dir():
                algo_id = algo_dir.name.split('_')[0]  # Extract the numeric part
                script_path = algo_dir / 'script.py'

                if script_path.exists():
                    try:
                        # Load the module dynamically
                        spec = importlib.util.spec_from_file_location(
                            f"algo_{algo_id}",
                            script_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[f"algo_{algo_id}"] = module
                        spec.loader.exec_module(module)

                        # Check if the module has the required 'algo' function
                        if hasattr(module, 'algo'):
                            self.algorithms[algo_id] = module.algo
                            logger.info(f"Loaded algorithm: {algo_id} from {algo_dir.name}")
                        else:
                            logger.warning(f"Algorithm {algo_dir.name} does not have 'algo' function")

                    except Exception as e:
                        logger.error(f"Failed to load algorithm {algo_dir.name}: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"Script.py not found in {algo_dir}")

        logger.info(f"Loaded {len(self.algorithms)} algorithms: {list(self.algorithms.keys())}")

    def run_algorithm_sync(self, algo_id: str, message: str) -> str:
        """Run algorithm synchronously (will be executed in thread pool)"""
        try:
            if algo_id not in self.algorithms:
                return json.dumps({
                    'status': 'error',
                    'error_type': 'NotFound',
                    'message': f'Algorithm {algo_id} not found. Available: {list(self.algorithms.keys())}'
                }, indent=2)

            algo_func = self.algorithms[algo_id]
            result = algo_func(message)
            return result

        except Exception as e:
            logger.error(f"Error running algorithm {algo_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({
                'status': 'error',
                'error_type': 'ExecutionError',
                'message': f'Failed to execute algorithm: {str(e)}'
            }, indent=2)

    async def run_algorithm_async(self, algo_id: str, message: str) -> str:
        """Run algorithm asynchronously using thread pool executor"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.run_algorithm_sync,
            algo_id,
            message
        )
        return result

    async def handle_algorithm_request(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle incoming HTTP requests for algorithms"""
        algo_id = request.match_info.get('algo_id', '')

        # Log request
        logger.info(f"Received request for algorithm: {algo_id}")

        try:
            # Read request body
            if request.body_exists:
                body = await request.read()
                message = body.decode('utf-8')
            else:
                return aiohttp.web.json_response(
                    {
                        'status': 'error',
                        'error_type': 'BadRequest',
                        'message': 'Request body is required'
                    },
                    status=400
                )

            # Validate JSON format
            try:
                json.loads(message)
            except json.JSONDecodeError:
                return aiohttp.web.json_response(
                    {
                        'status': 'error',
                        'error_type': 'JSONDecodeError',
                        'message': 'Invalid JSON in request body'
                    },
                    status=400
                )

            # Execute algorithm
            result_json = await self.run_algorithm_async(algo_id, message)

            # Parse result to check status
            result_dict = json.loads(result_json)

            # Determine HTTP status code based on result
            if result_dict.get('status') == 'error':
                if result_dict.get('error_type') == 'NotFound':
                    status_code = 404
                else:
                    status_code = 400
            else:
                status_code = 200

            return aiohttp.web.Response(
                text=result_json,
                content_type='application/json',
                status=status_code
            )

        except Exception as e:
            logger.error(f"Unexpected error handling request: {str(e)}")
            logger.error(traceback.format_exc())
            return aiohttp.web.json_response(
                {
                    'status': 'error',
                    'error_type': 'ServerError',
                    'message': f'Internal server error: {str(e)}'
                },
                status=500
            )

    async def handle_health(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Health check endpoint"""
        return aiohttp.web.json_response({
            'status': 'healthy',
            'algorithms_loaded': list(self.algorithms.keys())
        })

    async def handle_root(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Root endpoint with API information"""
        return aiohttp.web.json_response({
            'name': 'Algorithm Server',
            'version': '1.0.0',
            'endpoints': {
                '/': 'API information',
                '/health': 'Health check',
                '/algorithm/{algo_id}': 'Execute algorithm (POST with JSON body)',
            },
            'available_algorithms': list(self.algorithms.keys()),
            'example_usage': {
                'method': 'POST',
                'url': '/algorithm/001',
                'body': '[1.0, 2.0, 3.0, 4.0, 5.0]',
                'content_type': 'application/json'
            }
        })

    def create_app(self) -> aiohttp.web.Application:
        """Create and configure the aiohttp application"""
        app = aiohttp.web.Application()

        # Add routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/health', self.handle_health)
        app.router.add_post('/algorithm/{algo_id}', self.handle_algorithm_request)

        # Add middleware for CORS if needed
        @aiohttp.web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        app.middlewares.append(cors_middleware)

        return app

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

async def main():
    """Main entry point"""
    server = AlgorithmServer()
    app = server.create_app()

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()

    site = aiohttp.web.TCPSite(runner, '0.0.0.0', 8080)

    logger.info("Starting Algorithm Server on http://0.0.0.0:8080")
    logger.info(f"Available algorithms: {list(server.algorithms.keys())}")

    try:
        await site.start()
        # Keep the server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.cleanup()
        await runner.cleanup()

if __name__ == '__main__':
    # Use uvloop for better performance if available
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("Using uvloop for better performance")
    except ImportError:
        logger.info("uvloop not available, using default event loop")

    asyncio.run(main())
