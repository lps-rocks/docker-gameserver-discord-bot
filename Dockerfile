FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy bot files
COPY app/ /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create /config directory
RUN mkdir -p /config

# Run the bot
CMD ["python", "bot.py"]
