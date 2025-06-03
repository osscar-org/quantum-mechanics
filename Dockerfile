FROM python:3.9-slim

RUN apt-get update && apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install lxml_html_clean

COPY . .

EXPOSE 8080

CMD ["voila",\
    "--Voila.ip=0.0.0.0", \
    "--port=8080",\ 
    "--no-browser",\ 
    "--Voila.config_file_paths=./",\ 
    "--Voila.log_level=10",\
    "--Voila.show_tracebacks=True",\
    "--MappingKernelManager.cull_interval=60",\
    "--MappingKernelManager.cull_idle_timeout=600",\
    "--MappingKernelManager.cull_busy=False",\
    "notebook/index.ipynb"]
