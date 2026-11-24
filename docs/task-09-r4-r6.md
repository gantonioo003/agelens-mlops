# AgeLens | Task 09: contribuicao R6

## Experimento

A hipotese H1 pre-registrada era que acrescentar as cabecas de menoridade abaixo de 13 e 16 anos ao R4 aumentaria a media de F2 nesses dois limiares. O R6 foi treinado do zero nas sementes 17, 42 e 2026; o R4 foi recuperado dos checkpoints da task 08, sem retreino. Foram mantidos o cache FaRL congelado, as particoes (9.001 exemplos de treino e 1.500 de selecao), os hiperparametros escolhidos em R0, a inicializacao das camadas e cabecas comuns, e a ordem dos lotes enquanto os modelos treinavam. O melhor checkpoint foi escolhido por MAE em `val_select`, conforme o protocolo.

A mudanca foi adicionar duas cabecas mantendo o peso de 0,25 de cada BCE. Os novos intervalos mascarados foram [10,15) para T=13 e [13,19) para T=16. A mascara afeta apenas a perda da cabeca correspondente durante treino, nao a perda de idade nem a avaliacao.

## Resultados em val_select

Medias de tres sementes. Delta = R6 menos R4, calculado por par. A metrica principal e a media de F2 em <13 e <16 por semente. O F2 abaixo usa pontos provisorios calculados separadamente em `val_select` para recall >= 0,99, preservando empates. Em <13 e <16 o R4 pontua com o negativo da idade prevista; nos limiares comuns ele usa as cabecas dedicadas. O R6 usa as seis cabecas.

| Metrica | R4 | R6 | Delta |
| --- | ---: | ---: | ---: |
| **Media de F2 em <13 e <16** | **0,9420** | **0,9324** | **-0,0096** |
| F2 em <13 | 0,9386 | 0,9501 | +0,0115 |
| F2 em <16 | 0,9453 | 0,9147 | -0,0306 |
| F2 em <12 | 0,9604 | 0,9660 | +0,0056 |
| F2 em <15 | 0,9023 | 0,8886 | -0,0137 |
| F2 em <18 | 0,6812 | 0,6605 | -0,0207 |
| F2 em <21 | 0,7328 | 0,7218 | -0,0110 |
| MAE geral (anos) | 4,6316 | 4,6230 | -0,0086 |
| MAE de 10 a 21 anos (anos) | 3,6888 | 3,6921 | +0,0034 |

**H1 nao recebeu apoio em `val_select`.** Os deltas do desfecho principal foram -0,0135, -0,0091 e -0,0060; media -0,0096 e desvio-padrao amostral 0,0038. Com o mesmo recall provisorio entre variantes, a taxa de falsos positivos caiu de 4,95% para 3,85% em <13, mas subiu de 4,97% para 8,45% em <16. As quedas nos limiares comuns <15, <18 e <21 excederam a margem de 0,01 do protocolo. O MAE geral ficou praticamente estavel e nao excedeu a margem de aumento de 0,5 ano.

Tres sementes descrevem variacao de treino, nao significancia estatistica. A propria `val_select` fornece os pontos provisorios e as metricas, portanto estes numeros nao sao uma avaliacao independente. Nenhum modelo foi escolhido para demonstracao ou publicado no Registry. Teste e calibracao continuam reservados.

## Verificacao e reproducao

Foram conferidos seis checkpoints, a inicializacao comum, o hash da amostragem, as metricas, os historicos de perda e a parada antecipada. O R6 restaurou as epocas 55/65, 28/38 e 39/49 (melhor/executadas). O MLflow tem tres runs R6 pareados aos respectivos R4 e um [run de resumo](http://127.0.0.1:15000/#/experiments/1/runs/ddf402a85003408bb1d576db2044faae) com graficos, tabelas e evidencias comparados por SHA-256. 68 testes locais passaram. Nenhuma foto ou feature individual foi registrada.

Com o MLflow ativo, o cache aprovado, os runs R4 preservados e o ambiente CUDA da task 08:

```powershell
.\runtime\gpu-env\Scripts\python.exe -m scripts.compare_r6
.\runtime\gpu-env\Scripts\python.exe -m scripts.compare_r6 --verify
```

[Grafico](../reports/task09/comparison.png) | [Tabela por semente](../reports/task09/paired_by_seed.csv) | [Resumo](../reports/task09/paired_summary.csv) | [Desfecho](../reports/task09/outcome.json) | [Comprovante](../reports/task09/verification.json)
