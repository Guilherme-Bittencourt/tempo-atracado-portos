# Dados

Coloque aqui os arquivos do Estatístico Aquaviário da ANTAQ (os `.zip` como vêm do site ou os `.txt` já extraídos). A busca é recursiva e não diferencia maiúsculas nem acentos no nome do arquivo.

Arquivos necessários, para cada ano (2022, 2023 e 2024):

| Arquivo | Obrigatório | Uso |
|---|---|---|
| `AAAAAtracacao.txt` | sim | berço, instalação, datas, tipo de navegação, armador |
| `AAAACarga.txt` | sim | peso bruto, TEU e natureza da carga |
| `AAAATemposAtracacao.txt` | não | só com `--origem-ta antaq`, que usa a coluna `TAtracado`. No padrão (como no artigo), o TA é a diferença entre as datas de desatracação e atracação |

Os `.txt` extraídos dos zips vão para `dados/extraidos/`. Nada desta pasta vai para o Git, exceto este arquivo.

Para reaproveitar modelos de uma execução anterior no Colab, coloque também aqui o `resultados.zip` daquela execução: os `.joblib` da pasta `modelos/` dentro dele são restaurados automaticamente.
