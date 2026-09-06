FROM python:3

WORKDIR /app

COPY requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY ..

# PYTHONUNBUFFERED=1 -> Allows output to not be delayed, shown as soon as outputted
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "127.0.0.1", "--port", "8000"]