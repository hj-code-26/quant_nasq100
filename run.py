"""서버 실행: python3 run.py  (기본 http://localhost:8899)"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8899, reload=False)
