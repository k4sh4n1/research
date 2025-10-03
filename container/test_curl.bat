curl http://localhost:8000/health

curl http://localhost:8000/algorithms

curl -X POST http://localhost:8000/001 -H "Content-Type: application/json" -d '[100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]'

curl -X POST http://localhost:8000/002 -H "Content-Type: application/json" -d '[100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]'

curl -X POST http://localhost:8000/003 -H "Content-Type: application/json" -d '[100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]'
