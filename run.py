"""서버 실행: python3 run.py  (기본 http://localhost:8899, PORT 로 변경 가능)"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="127.0.0.1",
                port=int(os.environ.get("PORT", 8899)), reload=False)
