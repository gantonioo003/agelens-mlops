# agelens-mlops
Projeto acadêmico de estimativa de idade e identificação de menores em imagens, utilizando FaRL, MLP e um pipeline integrado de MLOps com MLflow, MQTT, Airflow, API, dashboard e Docker.

## Etapas 01 a 12

- [Task 01: escopo e decisoes](docs/task-01-escopo.md)
- [Task 02: dados, licencas e protocolo experimental](docs/task-02-dados-e-protocolo.md)
- [Task 03: ambiente e infraestrutura inicial](docs/task-03-ambiente.md)
- [Task 04: auditoria integral e EDA interpretada](docs/task-04-auditoria-eda.md)
- [Notebook executado da auditoria e EDA](notebooks/04_auditoria_eda.ipynb)
- [Verificacao independente da task 04](reports/task04/verification.json)
- [Task 05: features FaRL congeladas e verificadas](docs/task-05-features-farl.md)
- [Notebook executado das features](notebooks/05_features_farl.ipynb)
- [Verificacao independente da task 05](reports/task05/verification.json)
- [Task 06: configuracao e verificacao do MLflow](docs/task-06-mlflow.md)
- [Comprovante da task 06](reports/task06/verification.json)
- [Task 07: comparacao de regressores e resultados](docs/task-07-comparacao.md)
- [Comprovante da task 07](reports/task07/verification.json)
- [Task 08: comparacao pareada R0 x R4](docs/task-08-r0-r4.md)
- [Comprovante da task 08](reports/task08/verification.json)
- [Task 09: contribuicao e comparacao R4 x R6](docs/task-09-r4-r6.md)
- [Comprovante da task 09](reports/task09/verification.json)
- [Task 10: simulador, MQTT e PostgreSQL](docs/task-10-ingestao.md)
- [Comprovante da task 10](reports/task10/verification.json)
- [Task 11: API de inferencia e recarga do modelo](docs/task-11-api.md)
- [Comprovante da task 11](reports/task11/verification.json)
- [Task 12: Airflow, inferencia agendada e dados rotulados](docs/task-12-airflow.md)
- [Comprovante da task 12](reports/task12/verification.json)
- [Fontes, condicoes de uso e procedencia dos dados](docs/data-sources.md)
- [Protocolo estruturado](configs/protocol-v1.json)

Projeto complementar individual de AM. UTKFace adquirido por espelho rastreado, com revisao e hash fixos; pesos FaRL obtidos do release oficial. Imagens, pesos, cache, credenciais e dados derivados individuais permanecem fora do Git.

A task 04 auditou tecnicamente as 23.708 imagens, colocou rotulos inconsistentes/conflitos exatos em quarentena e congelou 15.000 exemplos: treino 9.001, validacao de selecao 1.500, calibracao 1.499 e teste 3.000. Grupos de copias/similaridade nao cruzam particoes nem folds de treino. Isso nao comprova separacao de identidades. A EDA tem oito graficos interpretados; as distribuicoes de atributos sao analisadas somente no treino.

A task 05 extraiu as 15.000 representacoes FaRL de 512 dimensoes em FP32 na RTX 5050, sem TF32, aumento aleatorio ou ajuste de pesos. Cache privado, normalizacao L2 por imagem, splits/folds originais preservados e hashes vinculados ao protocolo. Cinquenta testes locais aprovados, 16 recomputacoes independentes e quatro conferencias em CPU validaram o resultado. O ambiente CPU anterior foi preservado; o ambiente CUDA separado e reproduzido por scripts/setup_features_gpu.ps1 e requirements-features-gpu.lock.

A task 06 configurou o tracking local com persistencia e verificou parametros, tags, historico de metricas e artefatos com runs sinteticos. A procedencia vincula cada run ao protocolo, aos manifests, ao codigo e ao cache aprovado.

A task 07 comparou mediana, KNN, Random Forest e MLP R0 com 19 runs de validacao cruzada e 12 ajustes finais no MLflow. O MLP teve o menor MAE em val_select: 4,65 anos (DP 0,05, tres sementes). O backbone FaRL nao foi treinado; teste e calibracao seguem reservados.

A task 08 comparou o R0 preservado com o R4 nas mesmas tres sementes. Em val_select, o MAE geral ficou praticamente igual; o F2 provisorio nos limiares de 18 e 21 anos caiu com R4. Os runs, checkpoints e o resumo pareado foram conferidos no MLflow. Teste e calibracao seguem reservados.

A task 09 avaliou a contribuicao R6 com duas novas cabecas. Em val_select, a media de F2 abaixo de 13 e de 16 anos caiu de 0,9420 (R4) para 0,9324 (R6), com delta negativo nas tres sementes. A comparacao foi conferida no MLflow; teste e calibracao continuam reservados.

## Reproduzir a task 04

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_data
.\.venv\Scripts\python.exe -m scripts.build_eda
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/private/task04-tests.xml
.\.venv\Scripts\python.exe -m scripts.verify_data_audit
```

O notebook repete a auditoria e exige manifests identicos. O protocolo e a politica estao em configs/protocol-v1.json e configs/audit-v1.json. Resultados agregados e figuras ficam em reports/task04; manifests individuais e pares de similaridade continuam em data/manifests, fora do Git. Antes de qualquer mudanca no recorte ou splits, versionar o protocolo; nao sobrescrever manifests congelados silenciosamente.

## Executar a base local

Requisitos: Windows de 64 bits com Python 3.12 e Docker Desktop em modo Linux. Na raiz do projeto:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python python
docker compose up -d --build --wait
```

MLflow: http://127.0.0.1:15000 . MQTT local: 127.0.0.1:18883. A inicializacao cria .env privado com senha aleatoria para o banco, sem sobrescrever uma configuracao existente.

Para adquirir os assets em outra maquina, ler as condicoes de uso e executar:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_assets --accept-research-terms
.\.venv\Scripts\python.exe -m scripts.check_assets
.\.venv\Scripts\python.exe -m scripts.verify_backbone
```

Para verificar o ambiente e os servicos:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m scripts.check_services
.\.venv\Scripts\python.exe -m pytest -q
docker compose --profile check run --rm workspace-check
```

O lock Windows documenta o ambiente cientifico local; as imagens Docker usam dependencias Linux proprias e nao incluem dados nem pesos. O Compose inclui broker, banco, MLflow e o consumidor de ingestao. O simulador sob demanda publica referencias de imagens pelo MQTT; o consumidor valida e grava os bytes e o estado pendente no PostgreSQL. A API local recarrega o R0 pelo MLflow. O perfil opcional de orquestracao executa o Airflow: o DAG operacional chama a API e persiste o resultado, e o DAG de treino prepara um snapshot privado apenas com os rotulos auditados. O dashboard fica para a etapa seguinte.

As portas estao restritas a localhost. A configuracao nao e adequada para publicacao na Internet sem autenticacao e TLS. Para parar sem apagar dados: `docker compose stop`.
