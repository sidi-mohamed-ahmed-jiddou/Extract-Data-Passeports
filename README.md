# Extract-Data-Passports

This project extracts data from passport images using OCR and question-answering model based on Hugging Face.


## Prerequisites
- Python 3.8
- pip

## Additional Dependencies

This project requires additional dependencies for OCR and question-answering models:

- OpenCV
- PaddleOCR
- Transformers

You can install these dependencies using pip:

```sh
pip install opencv-python-headless paddleocr transformers
```



## Getting Started

1. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install the dependencies:
    ```sh
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install opencv-python-headless paddleocr transformers
    ```

3. Run the Streamlit app:
    ```sh
    streamlit run main.py
    ```

4. Access the Streamlit app:
    Open your web browser and go to `http://localhost:8501`.

## Project Structure

- `requirements.txt`: Lists the Python dependencies.
- `main.py`: The main entry point for the Streamlit app.
