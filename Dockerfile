FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["voila",\
    "--Voila.ip=0.0.0.0",\
    "--classic-tree",\
    "--template=osscar",\
    "notebook/",\
    "--port=8080",\
    "--no-browser",\
    "--MappingKernelManager.cull_interval=60",\
    "--MappingKernelManager.cull_idle_timeout=120",\
    "--MappingKernelManager.cull_busy=True"]