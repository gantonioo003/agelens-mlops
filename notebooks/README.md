# Notebooks

04_auditoria_eda.ipynb executa a auditoria tecnica integral e a EDA com oito visualizacoes interpretadas. Possui outputs salvos, verifica a reproducao dos manifests e nao treina modelos. Os arquivos de imagem e manifests devem ser adquiridos/preparados localmente conforme o README; eles nao sao redistribuidos no Git.

05_features_farl.ipynb verifica e apresenta o cache aprovado no protocolo: 15.000 representacoes FP32 de 512 dimensoes, contagens por particao/fold e comprovantes de integridade e recomputacao. Tem outputs salvos, usa somente estatisticas agregadas e nao publica vetores faciais, linhas individuais ou imagens. A extracao e feita pela CLI em ambiente CUDA separado; o notebook apenas consome o cache privado e pode executar no ambiente CPU original. Nenhum modelo e treinado nesta etapa.
