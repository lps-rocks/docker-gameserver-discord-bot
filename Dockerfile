# Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files to /app
COPY app/ /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Run the bot
CMD ["python", "bot.py"]
