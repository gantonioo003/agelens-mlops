# AgeLens | Task 11: API e recarga do modelo

## Modelo e ponto operacional

A selecao recuperou 18 runs verificados no MLflow (seis candidatos, tres sementes) e usou somente `val_select`. O R0 venceu com F2 provisorio medio de 0,7952 em <18, seguido de Random Forest (0,7472), KNN (0,6851), R4 (0,6812) e R6 (0,6605). A mediana foi inelegivel por prever todos como menores no ponto de recall alvo. Nao houve empate que exigisse medir custo.

A selecao e o checkpoint da semente de referencia 42 foram congelados antes de materializar `val_calibration`. Nessa particao separada, com 1.499 exemplos, foram congelados os seis limiares para cada uma das tres sementes. O servico usa apenas a semente 42. Em <18, nessa particao, teve 268 acertos entre 270 menores (recall 99,26%) e 401 falsos positivos entre 1.229 adultos (32,63%). Esse ponto privilegia recall de calibracao; nao demonstra seguranca, nem deve ser usado para verificar idade ou controlar acesso. O teste reservado nao foi usado.

A cada inicializacao a API baixa o checkpoint R0 pelo ID do run no MLflow, confere o SHA-256, carrega somente o estado PyTorch e verifica os pesos FaRL. A imagem e decodificada, orientada por EXIF, convertida a RGB e submetida a mesma transformacao CLIP 224 e normalizacao L2 das features. Tudo roda em CPU, sem treino. O score do R0 e o negativo da idade estimada, nao uma probabilidade.

## Contrato HTTP

| Rota | Retorno |
| --- | --- |
| `GET /health` | Prontidao e versao |
| `GET /v1/model` | Variante, semente, run e hashes dos checkpoints |
| `POST /v1/predict` | UUID operacional, idade estimada, score, corte e decisao por limiar, versao e tempo |
| `GET /v1/results/{id}` | Resultado persistido ou 404 |

Envie os bytes do arquivo diretamente com `Content-Type: image/png` ou `image/jpeg`, sem formulario. Limites: 2 MiB e 2.048 pixels por lado. Arquivo invalido recebe 422; tipo nao aceito, 415; corpo acima do limite, 413. O servico nao persiste imagens nem registra seu conteudo nos logs. Resultados e tempos ficam no SQLite privado `runtime/inference/results.sqlite3`; consultas ignoram resultados com mais de sete dias e novas gravacoes limpam registros expirados. A [task 12](task-12-airflow.md) adiciona o header opcional `X-Event-ID`, reutilizado nas retentativas, e registra o resultado do evento no PostgreSQL. O servico e local, nao tem autenticacao de producao e nao deve ser exposto na rede.

## Reproducao

Com a base cientifica da task 03, assets da task 05 e MLflow local disponivel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv\Scripts\python.exe -m services.inference.prepare
.\.venv\Scripts\python.exe -m services.inference.api
```

A API fica em [localhost:18080/docs](http://127.0.0.1:18080/docs). Em outro terminal:

```powershell
.\.venv\Scripts\python.exe -m tools.verify_inference
```

A conferencia via HTTP recarregou o mesmo modelo em outro processo: diferenca absoluta de 0,0 ano no exemplo sintetico e decisoes identicas nos seis limiares. A consulta por ID retornou o mesmo JSON; quatro entradas invalidas foram recusadas e o servico continuou saudavel. Os 82 testes locais passaram.

[Selecao congelada](../configs/deployment-selection-v1.json) | [Calibracao congelada](../configs/deployment-calibration-v1.json) | [Comprovante](../reports/task11/verification.json) | [Registro no MLflow](http://127.0.0.1:15000/#/experiments/1/runs/9f64d39db47b4b18a18d5b5bac053b73)
