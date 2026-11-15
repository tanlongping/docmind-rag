.PHONY: install install-llm run test lint fmt eval seed docker-up docker-down clean

PY ?= python

install:            ## 安装最小依赖并开发模式安装
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"

install-llm:        ## 额外安装 LangChain / OpenAI 兼容适配层
	$(PY) -m pip install -e ".[llm]"

run:                ## 本地启动（热重载）
	$(PY) -m uvicorn app.main:create_app --factory --reload --port 8000

test:               ## 跑全部单元测试
	$(PY) -m pytest -q

eval:               ## 跑离线检索评测（recall@k / MRR）
	$(PY) scripts/eval_retrieval.py

seed:               ## 写入演示数据
	$(PY) scripts/seed_demo.py

bench:              ## 简易并发压测
	$(PY) scripts/bench_api.py

lint:
	$(PY) -m ruff check app tests

fmt:
	$(PY) -m ruff check --fix app tests

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ docmind.db
