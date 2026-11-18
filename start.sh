#!/bin/bash
ollama serve &
sleep 5
ollama pull qwen2.5:3b
uvicorn app:app --host 0.0.0.0 --port 7860