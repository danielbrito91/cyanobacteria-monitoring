# Monitoramento de cianobactérias no Lago Guaíba

![Desenvolvido em Python](https://img.shields.io/badge/-python-brightgreen)
![GEE](https://img.shields.io/badge/-GEE-brightgreen)
![XGBoost](https://img.shields.io/badge/-XGBoost-brightgreen)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-brightgreen)

## Índice

* [Índice](#índice)
* [Descrição do Projeto](#descrição-do-projeto)
* [Acesso ao projeto](#acesso-ao-projeto)
* [Tecnologias utilizadas](#tecnologias-utilizadas)
* [Referências](#referências)

Projeto desenvolvido na primeira etapa da mentoria [Alforriah](https://www.alforriah.com/).

Disponível em https://danielbrito91-cyanobacteria-monitoring-app-q9d98u.streamlit.app/.

## Descrição do Projeto
### Problema
As cianobactérias, popularmente conhecidas como "algas azuis", podem se proliferar de modo excessivo em reservatórios e corpos hídricos, especialmente aqueles locais com um regime lêntico, em um fenômeno conhecido como florações ("blooms") de algas. Esses eventos podem resultar em impactos ecônomicos e sanitários negativos<sup>[1](#referências)</sup>. Espera-se que, com o aquecimento global, ocorra um aumento na frequência e intensidade desses eventos<sup>[2](#referências)</sup>. O aumento das fontes de dados sobre de monitoramento de qualidade da água tem potencial de auxiliar o conhecimento da população e tomadores de decisão sobre o estado dos recursos hídricos frente a esse problema.

### Proposta
O objetivo deste trabalho foi se extrair uma série histórica de densidade de cianobactérias de recurso hídrico superficial a partir da análise de imagens de satélite. A área de estudo escolhida foi o Lago Guaíba, manancial da cidade de Porto Alegre, capital do Rio Grande do Sul.

### Implementação
O sistema calcula o índice NDVI (Normalized Difference Vegetation Index) e NDCI (Normalized Difference Chlorophyll Index) para o Lago Guaíba e os utiliza como preditores em uma regressão para se estimar as densidades de cianobactérias obtidas no monitoramento da qualidade da água realizado pelo setor de saúde (SISAGUA). Especificamente, foi realizada a análise de um dos pontos de captação da cidade (próximo ao par de coordenadas -30.012175, -51.215679). Trata-se de uma metodologia já adotada em alguns trabalhos<sup>[3, 4, 5](#referências)</sup>, especialmente para monitoramento de clorofila-a.


## Acesso ao projeto

Você poderá acessar o código fonte do projeto que foi organizado aos moldes do [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) com algumas pequenas adaptações.

## Tecnologias utilizadas

- ``Python``
    - ``Google Earth Engine``
    - ``Pandas``
    - ``plotly``
    - ``Scikit-learn``
    - ``Streamlit``
 
- ``AWS S3``

As seguintes bases de dados foram utilizadas no projeto:

- Série histórica Sentila-2A obtida no Google Earth Engine (European Space Agency): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2

- SISAGUA (Ministério da Saúde): https://dados.gov.br/dataset/sisagua-controle-mensal-resultado-de-analises


## Referências

[1] CETESB. Manual de cianobactérias planctônicas : legislação, orientações para o monitoramento e aspectos ambientais. 2013. https://cetesb.sp.gov.br/laboratorios/wp-content/uploads/sites/24/2015/01/manual-cianobacterias-2013.pdf

[2] Huisman, Jef; Codd, Geoffrey A.; Paerl, Hans W.; Ibelings, Bas W.; Verspagen, Jolanda M. H.; Visser, Petra M. 2018. Cyanobacterial blooms. Nature. https://www.nature.com/articles/s41579-018-0040-1

[3] Zhato, H. et al. Monitoring Cyanobacteria Bloom in Dianchi Lake Based on Ground-Based Multispectral Remote-Sensing Imaging: Preliminary Results. Remote Sensing. 2021 https://www.mdpi.com/2072-4292/13/19/3970

[4] Lobo, F.d.L.; Nagel, G.W.; Maciel, D.A.; Carvalho, L.A.S.d.; Martins, V.S.; Barbosa, C.C.F.; Novo, E.M.L.d.M. AlgaeMAp: Algae Bloom Monitoring Application for Inland Waters in Latin America. Remote Sens. 2021, 13, 2874. https://doi.org/10.3390/rs13152874

[5] Ventura, D. et al. Long-Term Series of Chlorophyll-a Concentration in Brazilian Semiarid Lakes from Modis Imagery. 2022. https://www.mdpi.com/2073-4441/14/3/400
