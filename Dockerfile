FROM python:3.13-slim

COPY . /src/

RUN pip install --no-cache-dir --upgrade pip setuptools \
 && pip install --no-cache-dir /src \
 && rm -rf /src \
# smoke test
 && python -c 'import fronts'
