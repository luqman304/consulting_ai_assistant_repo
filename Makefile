.PHONY: run test lint docker

run:
	streamlit run app/app.py --server.port=8501

test:
	pytest -q

lint:
	flake8 app tests

docker:
	docker build -t consulting_ai_assistant:latest .
