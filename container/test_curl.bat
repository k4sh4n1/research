curl http://localhost:8080/

curl http://localhost:8080/health

curl -X POST http://localhost:8080/algorithm/001 -H "Content-Type: application/json" -d "{\"data\": [1, 2, 3, 4, 5]}"

curl -X POST http://localhost:8080/algorithm/002 -H "Content-Type: application/json" -d "{\"signal\": [1, 2, 3, 4, 5]}"

curl -X POST http://localhost:8080/algorithm/003 -H "Content-Type: application/json" -d "{\"signal\": [1, 2, 3, 4, 5]}"
