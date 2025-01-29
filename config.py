import os

# Make sure you've set your OPENAI_API_KEY in your environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: path to Tesseract executable (if not in PATH)
TESSERACT_CMD = "/usr/bin/tesseract"