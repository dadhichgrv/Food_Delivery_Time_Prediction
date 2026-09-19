
# THese are instruction to create docker image and container

# Download this lightweight python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv 

# Copy requirements into working directory
COPY requirements-docker.txt ./

# Run requirements.txt file
RUN uv pip install --system -r requirements-docker.txt 

# COPY application code into container
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./power_transformer.pkl ./power_transformer.pkl
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py     
COPY ./run_information.json ./      

# EXPOSE port on which app will run
EXPOSE 8000

# Use uvicorn to run FastAPI app 
CMD ["uvicorn","app:app","--host", "0.0.0.0", "--port", "8000"]


# Command to run docker file 
# docker build -t food_delivery_time_prediction:latest .
# docker run --env-file .env --name delivery_time_pred -p 8000:8000 food_delivery_time_prediction
# When u get url , in the search bar write localhost:8000/docs

