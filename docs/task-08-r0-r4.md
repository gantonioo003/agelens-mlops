# AgeLens | Task 08: R0 versus R4

## Comparacao pareada

O R0 foi recuperado dos tres checkpoints verificados da task 07, sem retreino. O R4 foi treinado do zero na RTX com as mesmas 9.001 amostras de treino, 1.500 amostras de `val_select`, cache FaRL congelado e sementes 17, 42 e 2026. Tronco e cabeca de idade comecaram com os mesmos pesos por par; o gerador da amostragem por bin reproduziu a mesma ordem de lotes enquanto ambos treinavam. Foram mantidos AdamW, lr=0.0003, weight_decay=0, batch 128 e selecao do checkpoint pelo MAE em `val_select` (ate 100 epocas, paciencia 10, delta 0,001).

O R4 acrescenta quatro cabecas binarias (<12, <15, <18, <21), com alvo positivo estrito `idade < T`. A perda e `CE de idade + 0.25 * soma das quatro BCEs mascaradas`. Os intervalos [10,14), [12,18), [15,21) e [17,25) excluem exemplos apenas da BCE da cabeca correspondente no treino. Se nenhum exemplo do lote tiver mascara valida para uma cabeca, sua perda e zero. A perda de idade usa todo o lote; a avaliacao nao usa mascaras.

## Resultados em val_select

Media +/- desvio-padrao amostral de tres sementes. A coluna delta e a media das tres diferencas pareadas R4 menos R0.

| Metrica | R0 | R4 | Delta |
| --- | ---: | ---: | ---: |
| MAE de idade (anos) | 4,646 +/- 0,051 | 4,632 +/- 0,016 | -0,014 |
| MAE de 10 a 21 anos | 3,517 +/- 0,190 | 3,689 +/- 0,218 | +0,171 |
| F2 abaixo de 12 | 0,957 +/- 0,001 | 0,960 +/- 0,006 | +0,003 |
| F2 abaixo de 15 | 0,900 +/- 0,078 | 0,902 +/- 0,010 | +0,002 |
| F2 abaixo de 18 | 0,795 +/- 0,018 | 0,681 +/- 0,007 | -0,114 |
| F2 abaixo de 21 | 0,822 +/- 0,021 | 0,733 +/- 0,037 | -0,089 |

O R0 usa o negativo da idade prevista como score. O R4 usa a probabilidade da cabeca correspondente. Cada F2 da tabela usa um ponto de corte provisorio calculado na propria `val_select` para recall >= 0,99, mantendo empates juntos. As metricas sem ajuste (idade prevista < T no R0, probabilidade >= 0,5 no R4), contagens e dispersoes pareadas estao nos CSVs.

O MAE geral mudou pouco, e o erro na faixa de 10 a 21 anos aumentou. No limiar de 18 anos, ambos tiveram recall medio provisorio de 99,26%, mas a taxa de adultos classificados como menores subiu de 27,45% para 50,35%. Portanto, a adicao das cabecas nao trouxe ganho uniforme neste recorte. Estes resultados sao diagnosticos de selecao, nao estimativas finais ou evidencias de seguranca. Calibracao, teste reservado, escolha para demonstracao e contribuicao R6 ficaram fora desta etapa.

## Verificacao e reproducao

O R4 restaurou as epocas 43/53, 28/38 e 39/49 (melhor/executadas). Os seis checkpoints foram baixados e conferidos por SHA-256, e as metricas de validacao foram recalculadas. Foram conferidos hashes de procedencia, pesos iniciais compartilhados, ordem de amostragem, parametros, historicos de perda por cabeca e criterio de parada. 66 testes locais passaram. O protocolo e os arquivos de dados nao foram modificados; zero linhas de teste ou calibracao foram materializadas.

No MLflow, foram gravados tres runs de treino R4, cada um ligado ao R0 da mesma semente, e um [run de resumo pareado](http://127.0.0.1:15000/#/experiments/1/runs/da01085980954f5d91c4bf8ad986c42a) com metricas agregadas, tabelas, grafico e comprovante. Nenhum vetor individual ou imagem foi registrado. Nenhum modelo foi publicado no Registry.

Com o MLflow local ativo e o ambiente CUDA da task 05:

```powershell
.\runtime\gpu-env\Scripts\python.exe -m pip install -r requirements-task08.txt
.\runtime\gpu-env\Scripts\python.exe -m scripts.compare_r4
.\runtime\gpu-env\Scripts\python.exe -m scripts.compare_r4 --verify
```

[Grafico](../reports/task08/comparison.png) · [Tabela por semente](../reports/task08/paired_by_seed.csv) · [Resumo pareado](../reports/task08/paired_summary.csv) · [Comprovante](../reports/task08/verification.json)
