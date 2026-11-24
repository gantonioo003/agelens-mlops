# AgeLens | Task 02
## Dados, licencas, hipotese e protocolo v1

Registrado em 23/11/2026, antes de treinamento ou acesso a resultados. Este e um protocolo de reproducao reduzida, nao uma promessa de repetir os numeros do artigo.

### Decisoes desta etapa
- Dataset principal: UTKFace. Backbone: FaRL ViT-B/16 ep16, congelado. Comparacao: baseline constante, KNN, Random Forest e MLP; reproducao com quatro cabecas; contribuicao com seis cabecas.
- Sementes de treinamento: 17, 42 e 2026. Semente de selecao de amostra e fallback de split: 42. Mesmo conjunto de teste em todas as rodadas.
- Protocolo estruturado em configs/protocol-v1.json. Ajustes motivados por dados/hardware deverao ser registrados antes dos experimentos, com nova versao; nunca em funcao do resultado no teste.

### Status real
- Fontes oficiais e restricoes consultadas. Metodo central e adaptacoes identificados. Hipotese e regras de avaliacao registradas.
- Link original recortado indisponivel; acesso resolvido com o espelho py97/UTKFace-Cropped, revisao fixa e restricoes originais mantidas. Arquivo de 106.634.631 bytes, SHA-256 2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7, conferido contra o Git LFS do espelho.
- Extraidas 23.708 imagens JPG: 23.705 com nomes/rotulos interpretaveis, 3 inconsistentes e 23.686 candidatas na faixa 0 a 101 antes de deduplicar. As 32 imagens da amostra tecnica abriram corretamente. Isso nao substitui a auditoria completa da task 04.
- Pesos FaRL ep16 oficiais baixados e verificados por tamanho e hash local fixado em configs/assets.json. Carregamento e teste sintetico documentados na task 03; nenhum treinamento foi realizado.
- A task 02 esta concluida quanto a fontes, licencas, hipotese e protocolo. Antes dos experimentos, a task 04 ainda precisa auditar todas as imagens e congelar os manifests; isso e uma etapa prevista, nao uma pendencia de autorizacao ou de acesso.

---

# 01 | Dados e uso responsavel
### Fontes verificadas em 23/11/2026
- UTKFace: https://susanqq.github.io/UTKFace/ e https://github.com/aicip/UTKFace/blob/master/README.md . Uso restrito a pesquisa nao comercial; direitos das imagens permanecem com seus titulares. Os rotulos foram estimados e revisados por humanos, portanto nao equivalem a idade documental.
- FaRL: https://github.com/FacePerceiver/FaRL . Codigo sob MIT; preservar avisos quando reutilizar codigo. Verificar condicoes aplicaveis ao checkpoint antes de redistribuir; nao estender automaticamente a licenca do codigo a imagens ou outros materiais.
- Pesos: https://api.github.com/repos/FacePerceiver/FaRL/releases/tags/pretrained_weights . Checkpoint FaRL-Base-Patch16-LAIONFace20M-ep16.pth baixado, 652.165.542 bytes, hash local registrado. O digest nao foi publicado pelo autor na API consultada; nao declarar assinatura do autor.
- Splits de referencia: https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases . Codigo MIT; anotacoes seguem as condicoes dos datasets originais. Nao sao imagens nem autorizacao de redistribuicao.

### Origem alternativa documentada
Espelho: https://huggingface.co/datasets/py97/UTKFace-Cropped . O responsavel declara republicacao do UTKFace e aponta para a fonte original. Revisao e80f21fb21fa3631380c3cb634b79be3ac2f4cba e hash registrados em configs/assets.json. Nomes e rotulos foram conferidos na checagem inicial. O hash confirma a copia do espelho, nao equivalencia independente ao arquivo original indisponivel. A tag MIT do espelho nao substitui os direitos das imagens nem a restricao de pesquisa nao comercial. Detalhes em docs/data-sources.md.

### Recorte e auditoria planejados
- Imagens faciais RGB legiveis, idade inteira de 0 a 101 anos, para manter a cabeca de idade do artigo. Idades 102 a 116 ficam fora deste recorte, sem serem declaradas rotulos invalidos.
- Limite inicial: 15.000 imagens validas no total, ou todas se houver menos. Amostragem deterministica proporcional aos estratos de idade e ao split, sem balancear artificialmente validacao/teste. Registrar tamanho final apos deduplicacao.
- Contar exclusoes por motivo: ilegivel, rotulo ausente/inconsistente, fora do recorte, duplicata exata e conflito de rotulos. Nao excluir exemplos por erro alto do modelo.
- Manifesto local: sample_id, caminho relativo, idade, source, image_sha256, grupo de duplicatas, identidade quando realmente disponivel, split e motivo de exclusao. Sem identificadores de pessoas inventados.
- Revisar duplicatas exatas e semelhantes antes de congelar splits. Nomes e horarios dos arquivos nao provam identidade. Sem metadados confiaveis, nao afirmar separacao por pessoa.
- Imagens, manifests individuais, features, modelos e registros com caminhos privados ficam fora do Git. Publicar apenas codigo, protocolos e estatisticas agregadas sem rostos identificaveis.

---

# 02 | Splits e preparo sem vazamento
### Particao preferencial
Usar o split 0 de Paplham e Franc se os arquivos puderem ser associados de forma verificavel: folders 0 a 5 para treino, 6 e 7 para validacao, 8 e 9 para teste. Conferir especificamente os metadados UTKFace; a descricao geral do benchmark nao comprova que cada identidade esteja conhecida neste dataset.

### Fallback previamente definido
Se nao houver correspondencia confiavel, registrar a adaptacao antes de treinar: holdout 60% treino, 20% validacao, 20% teste, estratificado por faixas [0,4), [4,8), [8,12), [12,13), [13,15), [15,16), [16,18), [18,21), [21,25), [25,40), [40,60), [60,102), semente 42. Agrupar duplicatas/identidades conhecidas e preservar grupos inteiros; o tamanho real pode variar. Fundir estratos vizinhos insuficientes sem olhar predicoes. Nao prometer subject-exclusive se faltarem identidades.

### Papel de cada conjunto
- Treino: ajustar parametros, transformacoes e balanceamento. Validacao cruzada de tres folds somente dentro do treino; usar grupos quando existirem e estratos quando forem viaveis.
- Validacao: dividir uma vez em val_select e val_calibration (50%/50%, semente 42, preservando grupos). A primeira serve para early stopping e selecao; a segunda apenas para calibrar limiares depois de selecionar modelos.
- Teste: nao usar para escolher amostra, hiperparametros, epoca ou limiares. Executar a avaliacao final depois de congelar as decisoes. Manter as mesmas amostras para todos os modelos e sementes.
- Guardar manifests e hashes localmente. Publicar apenas resumo e hashes dos manifests, sem caminhos privados ou imagens. Qualquer incompatibilidade de splits deve ser corrigida antes do treinamento.

### Features e transformacoes
FaRL congelado em modo de avaliacao, entrada 224 x 224 conforme preprocessamento oficial e vetor projetado de 512 dimensoes com normalizacao L2. Cache de uma representacao deterministica por imagem. Nenhum gradiente no backbone. A ausencia de aumentos aleatorios por epoca e uma adaptacao explicita para permitir cache unico.

StandardScaler para KNN, ajustado dentro de cada fold e depois apenas no treino definitivo. Random Forest sem escalonamento. MLP sobre features L2. Nao encaixar PCA ou normalizadores em dados de validacao/teste.

---

# 03 | Experimentos registrados antes de rodar
### Hipotese principal H1
Adicionar cabecas para menor de 13 e 16 anos ao MLP de idade + 12/15/18/21 pode aumentar a media de F2 em 13 e 16, em relacao as decisoes obtidas da idade prevista pelo MLP de quatro cabecas, mantendo as demais condicoes.

Comparacao primaria: media de F2(13) e F2(16), com limiares calibrados para recall >= 0,99 na validacao. Desfechos secundarios: F2 em 12/15/18/21, MAE geral e MAE na faixa de 10 a 21 anos. Essa faixa e apenas analise secundaria, nao filtro do teste principal.

### Condicoes pareadas
- R0: idade somente. R4: idade + menor de 12, 15, 18 e 21. R6: R4 + menor de 13 e 16. R0 versus R4 reproduz a comparacao central; R4 versus R6 testa H1.
- Mesmos dados, features, splits, batch size, otimizador, budget, regra de parada e sementes. Inicializar o tronco, a cabeca de idade e as cabecas comuns com os mesmos valores por semente; controlar a ordem dos minibatches independentemente da criacao de cabecas extras.
- MLP independente 512 -> 256 -> 512, ReLU; cabeca de idade com 102 logits, cross-entropy, idade prevista pela expectativa de 0 a 101. Cabecas binarias com logits; probabilidade positiva corresponde a menor.
- Amostragem de treino equilibrada nos 12 bins do artigo (limites 4,8,12,16,20,24,28,32,36,42,50), peso por amostra inverso a contagem do bin. Mesma regra para R0/R4/R6; nunca balancear o teste.
- BCE binaria estavel em logits (sem reponderacao adicional e gamma=0), configuracao presente nas ablacoes do artigo. Age-gap relativo f=6/5: para a cabeca T, mascarar apenas sua perda em floor(T/f) <= idade < floor(T*f). Nao remover essas amostras da perda de idade nem da avaliacao.
- Perda: CE_idade + 0,25 * soma das perdas binarias, cada uma com media sobre seus exemplos nao mascarados. Se uma cabeca nao tiver exemplos validos no minibatch, sua parcela e zero. Peso 0,25 de cada cabeca comum permanece fixo ao adicionar 13/16; nao renormalizar pelo numero de cabecas. Esta agregacao e uma escolha documentada da reproducao reduzida.
- AdamW, ate 100 epocas, batch 128; early stopping por MAE em val_select, paciencia 10, melhoria minima 0,001 ano. Reusar a mesma configuracao selecionada em R0 para R4/R6. Registrar epocas realmente executadas.

---

# 04 | Comparacao dos tres algoritmos
DummyRegressor com mediana do treino como referencia. KNNRegressor, RandomForestRegressor e MLP de idade como algoritmos distintos. Todos estimam idade e permitem decisao de menoridade por score -idade prevista; nao comparar um classificador isolado com um regressor em tabelas de MAE.

Busca em tres folds no treino, selecao por menor MAE medio: KNN, k=[3,5,11,21], weights=[uniform,distance], p=2 (8 candidatos); RF, n_estimators=200, max_depth=[null,12,24], min_samples_leaf=[1,3], max_features=1.0 (6); MLP, lr=[0,001;0,0003], weight_decay=[0;0,0001] (4), arquitetura fixa. Semente 42 na busca. Desempate: menor dispersao de MAE e depois primeiro candidato na ordem declarada. Nos folds do MLP, usar epocas fixas 100, sem calibrar ou consultar o teste; no treino final, usar a regra de parada comum. Leave-one-out fica fora por custo desproporcional; justificar no relatorio.

---

# 05 | Metricas, selecao e interpretacao
### Decisao e calibracao
Para cada T, rotulo verdadeiro = 1 se idade < T; idade igual a T e adulto para aquele limiar. Score alto deve significar maior evidencia de menoridade. Cabecas: sigmoid(logit). Modelos de idade: -idade prevista.

Depois da selecao, por modelo, semente e T, escolher em val_calibration o maior limiar q que ainda atinja recall >= 0,99, usando predicao positiva quando score >= q. Empates de score sao mantidos juntos. Se nao houver positivos e negativos suficientes ou houver scores nao finitos, nao certificar o ponto operacional. Aplicar q inalterado ao teste. Incluir como referencia o ponto sem calibracao: idade prevista < T ou probabilidade >= 0,5.

### Metricas a registrar
- MAE em anos; precisao, recall, F1, F2 = 5TP/(5TP+4FN+FP), FPR, FNR e matriz de confusao com contagens por limiar. FNR = FN/(TP+FN) = 1 - recall.
- Reportar F2 e recall realmente obtidos no teste no ponto escolhido na validacao. Nao denominar esse valor F2 a recall 0,99 no teste se o recall observado nao for 0,99.
- Media e desvio-padrao amostral entre as tres sementes, e diferencas pareadas R6-R4. KNN deterministico pode repetir resultados; nao inventar variacao de inicializacao.
- Para FNR, contagens e intervalo binomial de 95%; n pequeno nao demonstra seguranca. Como referencia matematica, com zero erros sao necessarios pelo menos 299 menores independentes para limite superior unilateral exato de 95% <= 1%. Duplicatas ou dependencia por pessoa invalidam essa interpretacao simples.
- Sem teste externo, nao afirmar robustez a mudanca de dominio; sem identidades conhecidas, discutir possivel sobreposicao de pessoas. Resultados valem para o recorte observado, nao para plataformas reais.

### Escolha para demonstracao
Antes de val_calibration e do teste: comparar candidatos por F2 sob recall alvo em val_select, com limiares temporarios estimados apenas nessa particao. Exigir alvo atingido nas tres sementes; ordenar por media de F2 no limiar 18, depois menor FPR medio, menor MAE e menor custo. Esses numeros sao de selecao, nao estimativas finais imparciais.

Modelo selecionado e regra congelados antes de calibrar no conjunto separado. Se o recall final cair, reportar a queda; nao escolher outro modelo pelo teste. Se nenhum for elegivel ou todas as decisoes forem trivialmente positivas, demonstrar apenas como prototipo e explicitar a falha.

### Conclusao da contribuicao
Relatar a diferenca primaria media e sua dispersao. Considerar troca relevante de desempenho se houver queda media de F2 > 0,01 em algum limiar comum ou aumento de MAE > 0,5 ano. Estes sao criterios academicos predefinidos, nao tolerancias de seguranca. Tres sementes nao provam significancia estatistica nem viabilidade de verificacao etaria. Resultado negativo tambem conclui o experimento.

---

# 06 | Evidencias e proximos passos
### O que foi produzido
Escopo atualizado, hipotese H1 previa, protocolo estruturado, plano de splits/tuning/calibracao, registro de fontes e regras de privacidade. Acesso resolvido, imagens e pesos baixados com hashes conferidos; checagem inicial documentada. Nenhum treinamento, resultado de desempenho de modelo ou commit/push foi realizado.

### Requisitos das etapas seguintes
- Aquisicao de imagens e pesos encerrada com comprovantes locais em artifacts/private/acquisition.json e asset-check.json. Configuracao publica contem origem, tamanho e hashes, sem imagens.
- Teste de carregamento do FaRL e do ambiente na task 03; extracao de features somente na task 05.
- Na task 04, verificar compatibilidade de metadados oficiais com as imagens e executar auditoria completa/deduplicacao. O tamanho 15.000 e um teto planejado, nao tamanho final observado.
- Medir hardware, memoria e tempo em pequeno teste tecnico sem usar o teste final. A consulta CIM a RAM/GPU nao foi permitida neste ambiente; nao foi presumida uma GPU.
- Registrar manifesto final e confirmar suporte por classe para cada limiar. Se o recorte for insuficiente, revisar a amostra antes de treinar e versionar o protocolo.

### Checklist das etapas seguintes
- Task 03: organizar ambiente, dependencias, aquisicao rastreavel e estrutura inicial do projeto.
- Task 04: auditoria/EDA com cinco graficos interpretados e splits congelados; eventuais revisoes do protocolo registradas antes dos experimentos.
- Task 05: extrair features FaRL e verificar cache sem vazamento; somente depois iniciar MLflow e modelos nas tasks correspondentes.

### Fidelidade e adaptacoes a relatar
O artigo recebido em tema_03.pdf usa varios datasets, alinhamento e aumentos, splits de referencia e avaliacoes ASORES/ASWIFT. AgeLens usara somente UTKFace e cache deterministico, sem esses benchmarks externos nem promessas de reproduzir os numeros publicados. Preserva FaRL congelado, MLP compacto, expectativa da idade, cabecas de menoridade, amostragem por idade e comparacao controlada. Nao usar a formula de focal loss reponderada sem esclarecer sua ambiguidade de notacao; o protocolo fixa a alternativa BCE ja presente nas ablacoees.

### Referencias metodologicas
- Gaul et al., Underage Detection through a Multi-Task and MultiAge Approach for Screening Minors in Unconstrained Imagery; PDF recebido, secoes 4, 5.3, 5.4 e tabelas 6-8.
- FaRL oficial: https://github.com/FacePerceiver/FaRL e https://github.com/FacePerceiver/FaRL/blob/main/LICENSE
- UTKFace oficial: https://susanqq.github.io/UTKFace/
- Splits oficiais: https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases/blob/main/README.md

Status: task 02 concluida. Fontes, licencas, acesso, hipotese e protocolo registrados; nenhum experimento treinado. Auditoria completa e congelamento dos manifests permanecem no escopo da task 04, antes do treinamento.
