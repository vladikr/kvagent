FROM python:3.11
#RUN mkdir -p /app
#RUN chown kvuser /app
#USER kvuser
USER root

WORKDIR /app
COPY requirements.txt .

RUN curl --location -k -v -o formattedInstTypesCollection.json 'https://www.dropbox.com/scl/fi/8o1u066dt2il5xyznpvp7/formattedInstTypesCollectionCopy.json?rlkey=4rxvzq1e3toacorr78wvemc9c&st=actfsd0f&dl=0'
RUN curl --location -k -v -o formattedPrefCollection.json 'https://www.dropbox.com/scl/fi/3q4cd4gl3pbm1t691ru6t/formattedPrefCollection.json?rlkey=crmv2utqrr3zv0cyqkjm3fyrf&st=gquho7tw&dl=0'

RUN pip install --upgrade pip
RUN pip install --default-timeout=100 --no-cache-dir --upgrade -r /app/requirements.txt
COPY app.py .
COPY kvagent.py .
COPY kvtypes.py .
EXPOSE  8501
ENV  HF_HUB_CACHE=/app/models/
ENTRYPOINT [ "streamlit", "run" ,"app.py" ]
