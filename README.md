
# Start
```
python -m venv venv
venv\Scripts\activate.bat
pip install requirements.txt#verificar
earthengine authenticate
```
Bash
```
echo "export PYTHONPATH=$PWD" >> dvc-venv/bin/activate
source dvc-venv/bin/activate
```

```
streamlit run
```

set PYTHONPATH=%cd%

dvc remote add s3-remote s3://cyano
