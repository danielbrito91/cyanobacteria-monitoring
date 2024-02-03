import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from io import StringIO
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

from urllib.request import urlopen

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()



def load_vigilancia(municipio: str = "PORTO ALEGRE", manancial:str = "GUAIBA") -> pd.DataFrame:
    """Realiza a leitura de dados baixados da Vigilancia para um dado manancial"""

    url = "https://sage.saude.gov.br/dados/sisagua/controle_mensal_demais_parametros.zip"
    
    r = urlopen(url).read()
    file = ZipFile(BytesIO(r))
    
    vigilancia = pd.read_csv(
            file.open("controle_mensal_demais_parametros.csv"),
            sep=";",
            decimal=",",
            encoding="latin-1",
            low_memory=False,
            parse_dates=["Data de preenchimento do relatório mensal", "Data da coleta"],
        )
    
    return vigilancia.loc[
        (vigilancia["Município"] == municipio)
        & (
            vigilancia["Nome do manancial superficial"] == manancial),
        :
    ]

if __name__ == '__main__':
    vigi = load_vigilancia()
    
    vigi.to_parquet( "s3://vigiagua/vigi.gzip", compression="gzip")   
