FROM python:3.8
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./Model_Inference.py /app/Model_Inference.py
COPY ./Exp_large_code_t5-base /app/Exp_large_code_t5-base
RUN apt-get update -y

RUN pip install -r requirements.txt
EXPOSE 7860
CMD [ "python", "./Model_Inference.py"]
