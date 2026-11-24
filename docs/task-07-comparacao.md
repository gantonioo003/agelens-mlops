# AgeLens | Task 07: regressao de idade

## Desenho do experimento

Comparamos DummyRegressor (mediana), KNN, Random Forest e o MLP R0, que tem somente cabeca de idade. Todos recebem os mesmos vetores FaRL congelados de 512 dimensoes da task 05. O protocolo `configs/protocol-v1.json` nao foi reescrito.

- Busca: 9.001 amostras de treino, tres folds congelados, semente 42. Oito candidatos de KNN, seis de Random Forest e quatro de MLP, alem de uma referencia de mediana. Selecao por menor MAE medio; desempates por menor desvio-padrao amostral entre folds e ordem do candidato.
- Ajuste final: usar a configuracao vencedora de cada familia e todo o treino, nas sementes 17, 42 e 2026. Comparar nos mesmos 1.500 exemplos de `val_select`. O KNN e o Dummy sao deterministicos e nao ganham aleatoriedade artificial.
- Preprocessamento: KNN tem StandardScaler dentro do pipeline, ajustado so nas linhas de treino de cada ajuste. Random Forest usa a representacao original; MLP usa os vetores L2 da task 05, com amostragem inversa por bin de idade calculada apenas no treino de cada ajuste.
- MLP: AdamW, minibatches de 128, 102 logits com cross-entropy; idade prevista pela expectativa das probabilidades. Cada ajuste da busca usa exatamente 100 epocas, sem early stopping. Nos refits, a paciencia e 10, a melhoria minima e 0,001 ano, o limite e 100 epocas e o checkpoint selecionado pelo criterio de parada e restaurado.
- Teste e calibracao: zero linhas materializadas ou previstas. O hash integral da matriz, junto dos hashes de procedencia, confere integridade dos arquivos, sem usar as linhas reservadas para analise, treinamento, metricas ou escolha de limiares.

## Medidas e limites

O MAE mede erro absoluto de idade, em anos. Os resultados da validacao sao diagnosticos de selecao, nao estimativas finais. Para os limiares 12, 13, 15, 16, 18 e 21, registramos metricas de menoridade com o ponto bruto (`idade prevista < T`) e com um ponto provisoriamente escolhido em `val_select` para recall de pelo menos 0,99. Nesse segundo ponto os empates ficam juntos. Os mesmos exemplos geram e avaliam esses pontos provisorios; eles nao substituem a calibracao separada nem indicam recall de teste. Nenhum modelo de demonstracao foi escolhido definitivamente.

As tabelas em `reports/task07` contem apenas estatisticas agregadas e IDs de runs locais. A dispersao e desvio-padrao amostral entre as tres sementes, nao intervalo de confianca. O protocolo de separacao por grupos limita duplicatas conhecidas, mas nao comprova independencia de pessoas. Os resultados nao justificam uso para verificacao de idade no mundo real.

## Resultados

| Modelo | Configuracao selecionada nos folds | MAE CV (anos) | MAE em val_select (media +/- DP) | F2 provisorio abaixo de 18 |
| --- | --- | ---: | ---: | ---: |
| Mediana | Mediana do treino | 14,89 | 15,21 +/- 0,00 | 0,523 |
| KNN | k=11, distancia euclidiana, peso por distancia | 5,34 | 5,33 +/- 0,00 | 0,685 |
| Random Forest | 200 arvores, profundidade 24, minimo 3 por folha, max_features=1.0 | 5,81 | 5,69 +/- 0,01 | 0,747 |
| MLP R0 | lr=0.0003, weight_decay=0 | **4,88** | **4,65 +/- 0,05** | **0,795** |

O MLP teve o menor MAE nos folds e em `val_select`. Na validacao, errou em media 0,68 ano a menos que o KNN, mas a ordem desses algoritmos nao e a mesma para todas as metricas: o Random Forest perdeu para o KNN em MAE e ganhou em F2 provisorio abaixo de 18. O MLP atingiu recall provisorio medio de 99,26% nesse limiar, ao custo de 27,45% de falsos positivos entre adultos. A mediana atingiu 100% de recall apenas porque classificou todos como menores nesse ponto, com 100% de falsos positivos. Isso mostra por que recall, isoladamente, nao basta.

O melhor checkpoint do MLP ficou nas epocas 23, 28 e 39, para as sementes 17, 42 e 2026. A configuracao selecionada em R0 esta congelada para a comparacao posterior com R4 e R6; a escolha para demonstracao e a calibracao final permanecem para etapas futuras.

[Grafico](../reports/task07/comparison.png) · [Grade e resultados de CV](../reports/task07/cv_candidates.csv) · [Metricas por semente](../reports/task07/by_seed.csv) · [Resumo completo](../reports/task07/summary.json).

## Rastreamento e reproducao

Cada candidato e um run, com metricas dos folds separadas por step. Cada refit/seed e um run, associado ao candidato escolhido. O MLflow guarda parametros, metricas, historicos de epocas do MLP, procedencia e JSON agregado por run. Os checkpoints finais do Dummy, Random Forest e MLP tambem ficam no servidor local. Os modelos KNN guardam os vetores e rotulos de treino: os arquivos ficam somente em `artifacts/private/task07`, com SHA-256 registrado, e nao sao enviados ao MLflow, Git ou contexto Docker. Nenhum modelo vai para o Registry nesta etapa. Os arquivos privados nao devem ser compartilhados ou publicados; o OneDrive deve ter acesso restrito.

Com MLflow ativo conforme a task 06, execute na raiz do projeto. Para reproduzir a configuracao da RTX, instale no ambiente GPU da task 05 o cliente com `runtime/gpu-env/Scripts/python.exe -m pip install mlflow-skinny==3.18.0`. Use uma nova identificacao de campanha ao reproduzir com outro codigo, ambiente ou protocolo.

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_baselines --family dummy
.\.venv\Scripts\python.exe -m scripts.compare_baselines --family knn --jobs 2
.\.venv\Scripts\python.exe -m scripts.compare_baselines --family random_forest --jobs 12
.\runtime\gpu-env\Scripts\python.exe -m scripts.compare_baselines --family mlp --device cuda
.\.venv\Scripts\python.exe -m scripts.compare_baselines --verify
```

Cada etapa salva progresso apos o termino de um run. A verificacao rele os parametros e metricas do servidor, checa os artefatos e as ligacoes de procedencia, compara a grade de candidatos com o protocolo e confirma que todos os refits usam o vencedor dos folds de treino.

## Conferencia final

31 runs conferidos no MLflow: 19 de validacao cruzada (incluindo a referencia de mediana) e 12 refits. Os 12 modelos salvos reproduziram as previsoes em `val_select` apos recarregar (maior diferenca absoluta: 4,27e-14 ano). Os nove binarios enviados foram baixados e conferidos por SHA-256; os tres KNN ficaram apenas no armazenamento privado. 62 testes locais passaram. [Comprovante](../reports/task07/verification.json).
