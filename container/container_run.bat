podman run -d --name algorithm-server -p 8080:8080 -v $(pwd)/algorithm:/app/algorithm:ro algorithm-server:latest
