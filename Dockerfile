FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the de421.bsp ephemeris at build time so the container does not download it on every start
RUN python -c "from skyfield.api import load; load('de421.bsp')"

COPY eva_app.py .
COPY static ./static

# The script binds to 127.0.0.1, which cannot be reached from outside the container; change it to 0.0.0.0
RUN sed -i 's/server_name="127.0.0.1"/server_name="0.0.0.0"/' eva_app.py

EXPOSE 7860
CMD ["python", "eva_app.py"]
