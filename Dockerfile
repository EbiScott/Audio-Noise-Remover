FROM python:3.11-slim          
# Base Linux image with Python

# Install ffmpeg (audio conversion) and libsndfile (audio file reading)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app                   
# All your bot files live here

COPY requirements.txt .

# Install PyTorch (CPU only — smaller image, no GPU needed on Render)
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt   
# Install everything else

COPY . .                      
# Copy bot.py, audio_processor.py etc.

CMD ["python", "bot.py"]       
# Command Render runs to start your bot
```