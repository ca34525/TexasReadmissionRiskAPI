Synthea data: java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true

Build Image: docker build -t readmissions-app .

Launch the Jupyter Notebook Server: docker run --rm -p 8888:8888 -v .:/app readmissions-app jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

Access Notebook: file:///root/.local/share/jupyter/runtime/jpserver-1-open.html

