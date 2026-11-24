# AgeLens | Task 05

## Escopo

Extrair uma representacao FaRL por imagem do recorte congelado na task 04. Nao treinar o backbone, MLP, KNN ou Random Forest; nao ajustar scaler, PCA, limiares, hiperparametros ou calibracao. Features nao sao previsoes de idade e nao demonstram desempenho de classificacao.

## Representacao

- FaRL-Base-Patch16-LAIONFace20M-ep16 oficial, com checkpoint SHA-256 registrado no protocolo.
- Encoder visual carregado estritamente, preservando os pesos FP32 originais. Os modulos auxiliares de pre-treinamento mascarado nao sao usados na inferencia visual.
- Orientacao EXIF e RGB, seguidos pelo transform oficial do CLIP fixado por commit: resize bicubico, center crop 224 x 224, tensor e normalizacao pelas constantes oficiais.
- Nenhum crop, espelhamento ou aumento aleatorio. Modelo em eval, parametros sem gradiente e forward em inference_mode.
- Vetor projetado de 512 dimensoes, float32, normalizado por L2 individualmente. Essa normalizacao nao estima estatisticas no dataset e pode ser aplicada separadamente a treino e held-outs sem ajustar parametros nos conjuntos reservados.

## Correspondencia com os dados

Os manifests da task 04 sao verificados antes da extracao; nenhuma linha e embaralhada. Cada imagem tem seus bytes conferidos contra o SHA-256 auditado no momento da leitura.

O recorte mantem 15.000 exemplos: treino 9.001, val_select 1.500, val_calibration 1.499 e teste 3.000. O cache conserva sample_id, idade anotada, split, fold, grupo conservador e hash da imagem. Indices separados apontam para cada particao e para os tres folds de treino, sem duplicar ou redistribuir exemplos.

Nao se pode afirmar separacao por identidade: os grupos detectados sao apenas de duplicatas/similaridade, conforme as limitacoes da task 04.

## Cache e retomada

O cache fica em data/processed/farl_ep16, fora do Git e do contexto Docker. Features faciais e metadados individuais nao devem ser publicados. Somente resumo, hashes e comprovantes agregados vao para reports/task05.

O projeto esta numa pasta do OneDrive: .gitignore nao impede sincronizacao em nuvem. Os dados privados continuam sujeitos a controle de acesso, condicoes de uso e politica de retencao, inclusive nas copias sincronizadas.

A identidade do cache vincula configuracao, checkpoint, manifests, versoes das bibliotecas, plataforma e codigo da extracao. Cada bloco tem hash de integridade e progresso atomico; uma retomada so aceita blocos contiguos e inalterados. O ponteiro active.json so e publicado depois de uma matriz completa e validada. Cache corrompido ou de outra configuracao e recusado, nao usado silenciosamente.

## Comandos

O ambiente CPU da task 03 e preservado. A extracao usa um ambiente separado, runtime/gpu-env, com Python 3.12, PyTorch 2.8.0 + CUDA 12.8 e torchvision 0.23.0. As URLs oficiais e hashes dos wheels estao em requirements-features-gpu.txt; requirements-features-gpu.lock congela os 35 pacotes do ambiente efetivamente validado. E necessario driver NVIDIA compativel; nao ha fallback silencioso para CPU.

Na raiz do projeto:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_features_gpu.ps1
.\runtime\gpu-env\Scripts\python.exe -m scripts.extract_features --benchmark-only
.\runtime\gpu-env\Scripts\python.exe -m scripts.extract_features
.\runtime\gpu-env\Scripts\python.exe -m pytest -q --junitxml=artifacts/private/task05-tests.xml
.\runtime\gpu-env\Scripts\python.exe -m scripts.verify_features
.\.venv\Scripts\python.exe -m scripts.build_feature_notebook
```

O benchmark usa apenas imagens de treino para medir tempo, sem ajustar pesos. A verificacao final exige o cache registrado no protocolo; a primeira verificacao de publicacao usa explicitamente --before-freeze antes do registro dos hashes.

O forward CUDA permanece em FP32: sem autocast, quantizacao ou TF32, com algoritmos deterministicos e CUBLAS_WORKSPACE_CONFIG=:4096:8. A GPU reduz o tempo de execucao, nao melhora por si so a qualidade da representacao. Diferencas numericas CPU/GPU sao verificadas com tolerancia explicita. Trocar configuracao, dispositivo ou versoes gera outra identidade de cache e exige nova verificacao.

## Conferencias executadas

- Todas as linhas finitas, shape 15000 x 512, dtype float32 e normas L2 proximas de 1.
- Correspondencia integral de IDs, rotulos, splits e folds com os manifests congelados.
- Integridade dos arquivos do cache, metadados e bytes das 15.000 imagens selecionadas.
- Mesmo hash do estado do encoder antes/depois, parametros congelados, nenhum gradiente ou otimizador.
- Recarregamento independente do checkpoint e recomputacao de quatro imagens por particao: comparar inferencia individual com o cache em lote, admitindo apenas a tolerancia numerica FP32 declarada no comprovante.
- Repeticao individual identica na mesma plataforma; teste de corrupcao, desalinhamento, cache incompleto e retomada apos interrupcao.
- Conferencia cruzada de uma imagem por particao em CPU FP32, sem combinar caches de dispositivos diferentes.

## Status

Task 05 concluida em 24/11/2026, sem treinamento. Cache registrado em configs/protocol-v1.json e comprovante em reports/task05/verification.json, com completed=true.

| Conferencia | Resultado observado |
| --- | --- |
| Cache privado aprovado | d7ace2ec615fb9275113baae10414c1d |
| Matriz | 15.000 x 512, float32, todos os valores finitos |
| Normas L2 | 0,9999998212 a 1,0000001192 |
| Menor desvio padrao por dimensao | 0,0100709262; nenhuma dimensao constante |
| Dispositivo | RTX 5050 Laptop GPU, CUDA 12.8, FP32 sem TF32 |
| Lote escolhido | 4; maior throughput medido entre 4, 8, 16 e 32 no benchmark de inferencia |
| Tempo de extracao e escrita do cache | 197,69 segundos, sem contar preparacao do ambiente |
| Recomputacao independente | 16 imagens, quatro por particao; erro absoluto maximo 2,46e-7 |
| Conferencia CPU FP32 | 4 imagens, uma por particao; erro absoluto maximo 3,69e-7 |
| Tolerancia das comparacoes | atol=2e-6 e rtol=1e-5; sem afrouxamento apos os resultados |
| Repeticao individual na GPU | Identica bit a bit |
| Estado do encoder e precisao do checkpoint | Preservados; parametros congelados e gradientes ausentes |
| Integridade e alinhamento | 15.000 imagens, nove arquivos de cache, sete manifests e indices conferidos |
| Testes locais | 50 aprovados no ambiente CUDA |

Os resultados agregados tambem estao no notebook executado notebooks/05_features_farl.ipynb. O cache CPU parcial foi descartado; nenhuma feature de outra configuracao foi misturada ao cache aprovado. Nenhum modelo, scaler, PCA ou limiar foi ajustado, e nenhuma previsao de idade foi gerada. Nao houve commit ou push nesta etapa.

Proxima etapa apos a verificacao: task 06, configuracao do tracking MLflow, ainda antes dos treinamentos previstos nas tasks posteriores.
