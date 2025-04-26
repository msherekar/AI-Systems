FROM python:3.11.7
 
RUN pip install opencv-python-headless
 
COPY  . .
 
CMD ["objectDetectionService.py"]

ENTRYPOINT [ "python" ]