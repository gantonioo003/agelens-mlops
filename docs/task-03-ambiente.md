# AgeLens | Task 03
## Estrutura, ambiente e Docker inicial

Preparacao tecnica de 23/11/2026. Esta etapa nao executa EDA, busca de hiperparametros, treinamento ou avaliacao de desempenho.

### Escopo desta task
- Criar ambiente Python 3.12 separado em .venv, dependencias cientificas, pacote inicial e comandos de verificacao.
- Automatizar aquisicao e verificacao dos dados e pesos, mantendo imagens e segredos fora do Git.
- Preparar Docker Compose inicial com MQTT, PostgreSQL, MLflow e comando de checagem.
- Testar configuracao, extracao segura e retomada de downloads. Validar carregamento do backbone com uma entrada sintetica, sem extrair features do dataset.
- Airflow, simulador, API de inferencia, dashboard e pipeline completo continuam nas tasks posteriores; nao sao servicos ficticios ou funcionalidades ja entregues.

### Estrutura criada
- configs/: protocolo experimental e registro de assets com fontes e hashes.
- scripts/: setup, aquisicao, checagem de assets, ambiente, backbone e servicos.
- src/agelens/: pacote inicial do projeto. notebooks/: destino da EDA da task 04.
- infra/: configuracao local do broker MQTT. Dockerfile e compose.yaml: infraestrutura inicial.
- tests/: testes unitarios. .github/workflows/setup-check.yml: teste de configuracao em push/PR, sem baixar imagens e sem treinar.
- docs/: escopo, protocolo, fontes e instrucoes. data/, models/, runtime/ e artifacts/private/: somente locais, ignorados pelo Git.

---

# 01 | Como reproduzir
### Ambiente Windows
Usar Python 3.12 de 64 bits e Docker Desktop com containers Linux. Na raiz do repositorio: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python python.

Se o comando python nao apontar para uma instalacao funcional de Python 3.12, passar o caminho absoluto do executavel no parametro -Python. O ambiente ja preparado neste computador fica em .venv; nao copiar essa pasta para outra maquina.

O setup preserva credenciais existentes e gera .env privado se necessario. requirements-windows.lock registra as versoes do ambiente Windows verificado; requirements.txt declara as dependencias diretas. O lock Windows nao deve ser usado como lock Linux. O ambiente local usa PyTorch CPU, sem depender de GPU.

### Dados e pesos
- Ler docs/data-sources.md antes da aquisicao. Executar .venv/Scripts/python.exe -m scripts.prepare_assets --accept-research-terms.
- Para conferir o que ja esta baixado sem extrair de novo: adicionar --verify-only ao mesmo comando.
- Checagem de tamanho, hashes e amostra de imagens: .venv/Scripts/python.exe -m scripts.check_assets.
- Teste sintetico do FaRL: .venv/Scripts/python.exe -m scripts.verify_backbone. Isso nao e treinamento nem metrica de acuracia.
- Em rede lenta, scripts/download_large.py suporta retomada por HTTP Range, intervalos conferidos e concorrencia limitada. A verificacao SHA-256 posterior continua obrigatoria.

### Servicos locais
- Subir a base: docker compose up -d --build --wait.
- MLflow: http://127.0.0.1:15000 . Broker MQTT: 127.0.0.1:18883 . PostgreSQL e acessivel apenas na rede interna do Compose.
- Checar servicos reais: .venv/Scripts/python.exe -m scripts.check_services. Valida publicacao/recepcao MQTT, SELECT 1 no banco e endpoint HTTP de saude do MLflow.
- Checagem dentro da imagem: docker compose --profile check run --rm workspace-check.
- Parar preservando dados: docker compose stop. Volumes persistentes nao sao apagados automaticamente.

---

# 02 | Verificacao e limites
### Testes de desenvolvimento
- .venv/Scripts/python.exe -m pytest -q
- .venv/Scripts/python.exe -m ruff check scripts src tests
- .venv/Scripts/python.exe -m pip check
- docker compose config --quiet

Os testes unitarios cobrem protocolo sem uso do teste final, arquivos de tamanho/hash incorretos, limpeza de download falho, extracao normal, bloqueio de caminhos perigosos e links, e validacao dos intervalos de retomada. O workflow remoto so sera executado quando houver push; nao afirmar que o GitHub Actions passou antes disso.

### Comprovantes locais
Os comandos gravam acquisition.json, asset-check.json, environment.json, backbone-check.json e services-check.json em artifacts/private/. Esses comprovantes nao contem resultados de treino. O resumo agregado de verificacao deve ser mantido junto deste documento, sem caminhos pessoais ou segredos.

---

# 03 | Resultados observados
- Python local 3.12.14; ambiente cientifico instalado e pacote agelens-mlops 0.1.0 importavel. Dependencias sem conflitos segundo pip check. Versoes fixadas em requirements-windows.lock.
- PyTorch 2.8.0+cpu e torchvision 0.23.0+cpu; CUDA nao disponivel no ambiente instalado. MLflow 3.18.0 no cliente e servidor.
- FaRL oficial carregado com pesos congelados: saida sintetica de formato [1,512], finita, sem treinamento. Apenas modulos auxiliares do pre-treinamento mascarado foram ignorados; os pesos do encoder CLIP foram carregados estritamente, sem camadas aleatorias faltantes.
- UTKFace: arquivo com hash esperado, 23.708 JPGs; 23.705 nomes com rotulos interpretaveis, 3 inconsistentes e 32 imagens da amostra tecnica decodificadas. Ainda nao e auditoria completa.
- Dez testes unitarios aprovados; Ruff aprovado; configuracao Compose valida. O GitHub Actions ainda nao foi executado, pois nao houve push.
- Broker e banco saudaveis; publicacao/recepcao MQTT confirmada, SELECT 1 retornou 1 e saude HTTP do MLflow retornou 200.
- Imagens oficiais fixadas por digest; versoes Linux da infraestrutura em requirements-runtime.lock. Credenciais, imagens e pesos excluidos do Git e do contexto de build.

Status: tasks 01, 02 e 03 concluidas no escopo definido. Nao houve treinamento, extracao das features do dataset, commit ou push. A task 04 permanece por fazer.

### Seguranca e escopo de demonstracao
Portas expostas somente em loopback. MQTT anonimo e permitido apenas nesta rede local de desenvolvimento; antes de publicar qualquer servico, configurar autenticacao, TLS e autorizacao. O banco usa senha aleatoria local em .env. MLflow nao deve ser exposto na Internet sem controle de acesso.

O MLflow usa SQLite em volume persistente nesta infraestrutura inicial; o PostgreSQL e destinado a dados/eventos do pipeline. Manter a versao do MLflow alinhada entre cliente e servidor. A imagem cientifica completa e a integracao final de todos os servicos serao consolidadas antes da entrega, na task 14.

### Proxima etapa
Task 04: auditoria de todas as imagens, tratamento dos tres nomes com rotulos inconsistentes, duplicatas, cinco visualizacoes interpretadas e manifests de splits congelados. Nao iniciar modelos antes dessas verificacoes.
