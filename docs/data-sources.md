# Dados e pesos

## UTKFace

Fonte original: https://susanqq.github.io/UTKFace/ . O download oficial de faces recortadas retornou 404 em 23/11/2026.

Origem alternativa rastreada: https://huggingface.co/datasets/py97/UTKFace-Cropped . O responsavel declara que esta republicando as faces recortadas do UTKFace, aponta para a origem original e informa a indisponibilidade do link anterior. Trata-se de um espelho de terceiro, nao de distribuicao oficial ou de copia cuja identidade com o arquivo original tenha sido comprovada.

Usar revisao fixa e SHA-256 do Git LFS registrados em configs/assets.json. Esse hash confere a integridade em relacao ao espelho; nao prova consentimento das pessoas ou equivalencia exata com a distribuicao original. Conferir nomes e rotulos antes da EDA. A tag MIT do espelho nao substitui as restricoes originais: pesquisa nao comercial e direitos das imagens dos respectivos titulares.

Os rotulos sao estimativas revisadas, nao idade documental. Nao publicar rostos, imagens ou manifests individuais no GitHub. Demonstracao local restrita a finalidade academica, sem captura de pessoas reais e sem decisao automatizada de acesso. Dados brutos, features e pesos devem permanecer em pastas ignoradas pelo Git e ser removidos quando nao forem mais necessarios ao projeto, respeitando orientacoes institucionais.

## FaRL e CLIP

Checkpoint oficial: https://github.com/FacePerceiver/FaRL/releases/tag/pretrained_weights . FaRL ep16, 652.165.542 bytes. Download concluido; SHA-256 observado e fixado: 67b4d6bcad0cd4bdaa302741e05e92eb82ecb334ef92446555b3abd3e3cd3862. O release nao publicou um digest na API consultada; esse hash permite repetir a verificacao local, mas nao e uma assinatura criptografica do autor.

Codigo FaRL: https://github.com/FacePerceiver/FaRL (MIT). Codigo CLIP: https://github.com/openai/CLIP (MIT), fixado por commit em requirements.txt. Preservar licencas de terceiros quando reutilizar seu codigo. Nao presumir que licencas de codigo abrangem imagens de pre-treinamento.

## Aquisicao reproduzivel

Depois de ler estas condicoes: `python -m scripts.prepare_assets --accept-research-terms`.

O script confere hashes e tamanhos, rejeita caminhos perigosos e links no arquivo compactado, limita a extracao e salva comprovantes locais em artifacts/private/acquisition.json. Ele nao cria splits, extrai features, publica arquivos ou treina modelos.
