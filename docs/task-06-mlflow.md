# AgeLens | Task 06: MLflow Tracking

## Ambiente

MLflow 3.18.0, cliente e servidor. O Compose publica a interface apenas em http://127.0.0.1:15000; o volume Docker `mlflow_data` persiste o SQLite em `/state/mlflow.db` e os artefatos em `/state/artifacts`. O servidor entrega os artefatos via HTTP; o cliente nao precisa montar o volume. O Postgres do Compose continua separado. A telemetria do servidor esta desativada.

- `agelens-infra-smoke`: verificacoes sinteticas de infraestrutura, fora das comparacoes cientificas.
- `agelens-research-v1`: reservado aos experimentos de pesquisa, sem runs nesta etapa.

O servidor e local e nao e uma configuracao de producao. Nao expor a porta em rede sem autenticacao e TLS. Nao usar `docker compose down -v` para parar o servico, pois essa opcao remove volumes. `docker compose stop mlflow` preserva os registros.

## Reproduzir

Na raiz do projeto, com Docker Desktop e o ambiente CPU da task 03 preparados:

```powershell
docker compose up -d --wait mlflow
.\.venv\Scripts\python.exe -m scripts.verify_tracking
docker compose restart mlflow
docker compose up -d --wait mlflow
.\.venv\Scripts\python.exe -m scripts.verify_tracking --recheck
```

Sem `--recheck`, o script cria dois runs novos no experimento de infraestrutura: um `FINISHED` e um `FAILED` intencional, identificado pela tag `agelens.expected_failure=true`. A segunda chamada rele os mesmos IDs e artefatos no servidor reiniciado, sem criar runs ou gravar dados no MLflow. O comprovante local fica em [reports/task06/verification.json](../reports/task06/verification.json). Os IDs sao do volume local, nao URLs remotas permanentes.

## Contrato para os experimentos

`src/agelens/tracking.py` usa um cliente explicito para o servidor local; nao cai em um armazenamento por arquivos ou em um tracking remoto se o endereco estiver incorreto. `collect_provenance` confere o ponteiro e os metadados aprovados do cache FaRL e os hashes dos sete manifests. Cada run recebe um JSON com o hash do protocolo, do checkpoint, da versao do cache e dos manifests, os tamanhos dos splits, os hashes dos arquivos Python em `src/agelens` e `scripts`, Python e versoes das bibliotecas no cliente. O hash de codigo reflete o conteudo local, inclusive sem commit.

Nas proximas tarefas, usar o experimento de pesquisa, com o algoritmo, etapa, semente e particao identificados por run e com parametros e metricas explicitos. Fases de selecao e calibracao nao devem consultar o teste reservado. O helper nao habilita autologging. Nao registrar imagens, vetores faciais, manifest por linha, credenciais ou caminhos pessoais. A integridade da matriz e de seus indices continua sendo conferida no carregador da task 05.

## Validacao

O teste gravou e releu parametros, tags, o historico de uma metrica sintetica em dois steps e dois artefatos JSON. Conferiu o status dos runs, a separacao dos experimentos, a interface HTTP e a persistencia apos reiniciar o servico. Um run da primeira tentativa foi encerrado como `KILLED` apos uma incompatibilidade de codificacao do console Windows; ele tambem e sintetico e nao constitui resultado de pesquisa.

Nenhum modelo foi treinado, avaliado, registrado ou publicado. As particoes e o cache aprovado da task 05 permanecem inalterados.

Referencia: [documentacao oficial do MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
