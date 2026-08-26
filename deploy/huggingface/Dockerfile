# Hugging Face Spaces retired the managed Streamlit SDK, so the console ships as
# a Docker Space instead. The only requirements HF imposes are that the app
# listens on the port declared as `app_port` in README.md (7860 here) and binds
# to 0.0.0.0.

FROM python:3.11-slim

# LightGBM links against libgomp at runtime and slim images do not ship it.
# Installed before dropping privileges, since apt needs root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000. Creating that user ourselves means the
# writable directories the pipeline expects (experiments/, logs/, data/) are
# owned by the process that needs them.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Dependencies first, so a code change does not invalidate the pip layer.
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

EXPOSE 7860

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
