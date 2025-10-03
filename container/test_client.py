import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any

async def test_algorithm(session: aiohttp.ClientSession, algo_id: str, data: List[float]) -> Dict[str, Any]:
    """Test a single algorithm"""
    url = f"http://localhost:8080/algorithm/{algo_id}"

    start_time = time.time()
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time

            print(f"\n{'='*50}")
            print(f"Algorithm: {algo_id}")
            print(f"Status Code: {response.status}")
            print(f"Response Time: {elapsed:.3f}s")
            print(f"Result Status: {result.get('status', 'unknown')}")

            if result.get('status') == 'success':
                if 'input_stats' in result:
                    print(f"Input Stats: {json.dumps(result['input_stats'], indent=2)}")
                if 'trend_analysis' in result:
                    trend = result['trend_analysis']
                    print(f"Trend: {trend.get('direction')} ({trend.get('strength')})")
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")

            return result
    except Exception as e:
        print(f"Error testing algorithm {algo_id}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    # Sample data
    test_data = [
        100.5, 101.2, 99.8, 102.3, 103.1, 101.9, 104.5, 103.2, 105.8, 104.3,
        106.1, 107.2, 105.9, 108.3, 107.1, 109.5, 108.2, 110.8, 109.3, 111.5
    ]

    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        async with session.get("http://localhost:8080/health") as response:
            health = await response.json()
            print(f"Server Health: {health}")

        # Test root endpoint
        async with session.get("http://localhost:8080/") as response:
            info = await response.json()
            print(f"Available algorithms: {info.get('available_algorithms', [])}")

        # Test algorithms concurrently
        algorithms = ['001', '002', '003']
        tasks = [test_algorithm(session, algo_id, test_data) for algo_id in algorithms]

        print("\nTesting algorithms concurrently...")
        results = await asyncio.gather(*tasks)

        print(f"\n{'='*50}")
        print("Concurrent test completed!")
        print(f"Total algorithms tested: {len(results)}")
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"Successful: {successful}/{len(results)}")

async def stress_test():
    """Perform a stress test with many concurrent requests"""
    test_data = list(range(1, 21))  # Simple test data

    async with aiohttp.ClientSession() as session:
        print("\nStarting stress test...")
        start_time = time.time()

        # Create 100 concurrent requests
        tasks = []
        for i in range(100):
            algo_id = ['001', '002', '003'][i % 3]
            tasks.append(test_algorithm(session, algo_id, test_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        failed = len(results) - successful

        print(f"\n{'='*50}")
        print(f"Stress Test Results:")
        print(f"Total Requests: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Requests/second: {len(results)/elapsed:.2f}")

if __name__ == "__main__":
    print("Algorithm Server Test Client")
    print("="*50)

    # Run the concurrent test and then stress test
    asyncio.run(test_concurrent_requests())
    asyncio.run(stress_test())
