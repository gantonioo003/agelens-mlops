# AgeLens | Task 01
## Escopo e alinhamento com o professor

Documento de planejamento - 23/11/2026. Nenhum treinamento realizado nesta etapa.

### O que ja esta definido
- Projeto complementar de Aprendizagem de Maquina; o participante nao cursa Projeto 6.
- Formato: individual; o aluno informou que o professor ja permite essa modalidade.
- Conta GitHub: gantonioo003. Nome completo para relatorio e cadastro: a informar.
- Repositorio confirmado: https://github.com/gantonioo003/agelens-mlops
- Pasta local: agelens-mlops, na Area de Trabalho do OneDrive.
- Tema escolhido pelo aluno: Tema 1, estimativa de idade e identificacao de menores com FaRL congelado e MLP multitarefa. No material recebido, o artigo deste tema esta em tema_03.pdf.

### Objetivo proposto
Reproduzir em escala reduzida a comparacao entre estimar apenas a idade e estimar idade junto com classificacoes de menoridade. Depois, testar uma contribuicao propria e demonstrar os modelos em um pipeline academico integrado de MLOps.

### O que este projeto nao promete
- Nao determinar idade real com certeza nem servir como mecanismo de verificacao de idade ou bloqueio de acesso.
- Nao coletar fotos de criancas, usar cameras reais ou disponibilizar monitoramento publico.
- Nao garantir recall de 99% ou melhora com a contribuicao; os resultados serao medidos e discutidos, inclusive se forem negativos.
- Implementar por etapas; nao treinar antes de validar acesso aos dados, splits e protocolo.

---

# 01 | Recorte cientifico proposto
### Dados e viabilidade
- Usar UTKFace como dataset principal, em reproducao reduzida, conforme a liberdade de datasets disponiveis prevista no enunciado. CLAP2016 e FG-NET nao integram o escopo minimo; sem teste externo nao afirmar generalizacao entre datasets.
- O protocolo da task 02 define o recorte; o acesso e a auditoria dos arquivos continuam sendo requisitos antes de treinar.
- Separar treino, validacao e teste antes de ajustar modelos; investigar duplicatas e separar identidades quando os metadados permitirem. Documentar limites quando nao houver identificadores.
- Nao publicar imagens ou pesos sem verificar as respectivas condicoes de uso.

### Reproducao do artigo
- Extrair uma vez as representacoes de um backbone FaRL pre-treinado e congelado.
- Comparar um MLP que estima somente idade com um MLP que estima idade e possui cabecas binarias para menor de 12, 15, 18 e 21 anos.
- Artigo conferido na task 02: idade por expectativa de uma distribuicao de classes, MLP independente e amostragem por idade. Registrar adaptacoes da reproducao reduzida; nao apresentar simplificacoes como reproducao exata.

### Contribuicao proposta
Hipotese a registrar antes de rodar: adicionar cabecas para menor de 13 e 16 anos pode melhorar a identificacao de menores perto desses limiares, sem degradar de forma relevante os limiares originais.

Mudar apenas o conjunto de cabecas: conservar backbone, dados, splits, sementes, regras de treinamento e selecao. Comparar tambem nos limiares comuns aos modelos. Nos limiares novos, comparar com a idade prevista pelo baseline convertida em decisao binaria.

### Comparacao adicional exigida na disciplina
Decisao: KNN, Random Forest e MLP usando as mesmas features FaRL e comparacao de estimativa de idade e deteccao de menores. Incluir baseline constante. As variantes do MLP nao serao contadas como algoritmos distintos.

---

# 02 | Protocolo e limites
### Regras propostas de avaliacao
- Idade: MAE. Menoridade: precisao, recall, F2, matriz de confusao e taxa de falso adulto por limiar etario, com menor como classe positiva.
- Falso adulto significa um menor classificado como adulto; reportar a proporcao sobre os menores avaliados.
- Escolher limiares de decisao na validacao para buscar recall de 0,99; medir e reportar o recall realmente obtido no teste, sem reajustar pelo teste.
- Reportar quantidades de menores/adultos e incerteza: em amostras pequenas, nao tratar uma estimativa de 1% de erro como garantia.
- Proposta de variacao: tres sementes de treinamento previamente fixadas, mantendo o mesmo teste reservado e as mesmas condicoes entre modelos. Documentar a variacao efetivamente introduzida por cada semente.
- Usar validacao cruzada somente no conjunto de desenvolvimento para comparacoes e tuning. O teste fica reservado para a avaliacao final.
- Justificar busca de hiperparametros e seu custo; justificar por que leave-one-out e ou nao aplicavel, sem executa-lo automaticamente.

### Criterio de escolha do modelo
Definir na task 02 uma regra antes dos resultados: privilegiar a deteccao de menores, sem ignorar falsos positivos, variacao e custo de inferencia. Escolher usando validacao; usar teste apenas para relatar desempenho final. Nao promover automaticamente o modelo de maior numero isolado.

### Fora do escopo inicial
- Treinar o backbone do zero ou fazer fine-tuning completo.
- Ataques cosmeticos, YOLO e treinamento zero-shot dos outros temas.
- Inferir atributos sensiveis adicionais ou coletar imagens de pessoas para demonstracao.
- Publicar o servico para decisoes reais sobre menores.

---

# 03 | Requisitos e evidencias de entrega
### Dados, modelos e rastreabilidade
- EDA e preparacao: notebook com estatisticas, decisoes de limpeza e pelo menos cinco visualizacoes interpretadas.
- Modelagem: baseline, tres algoritmos distintos, tuning justificado, avaliacao e discussao de estabilidade, generalizacao e sobreajuste.
- Reproducao e contribuicao: comparacoes controladas, hipotese previa, tres sementes ou alternativa aprovada de incerteza e analise dos resultados.
- MLflow: parametros, metricas, artefatos e modelos salvos; runs separados e ligados aos dados, splits e versoes de codigo utilizados.

### Integracao obrigatoria proposta
- Dispositivo simulado publica por MQTT identificadores de imagens de um conjunto de demonstracao autorizado, em armazenamento compartilhado; nao ha captura de pessoas reais.
- Consumidor persiste eventos e referencias dos arquivos em banco/armazenamento.
- Airflow automatiza ingestao, validacao, preprocessamento e preparacao dos dados para treinamento, alem da orquestracao de inferencia conforme o fluxo aprovado.
- API carrega o modelo selecionado e produz predicoes; dashboard consome resultados reais da API, apresenta metricas dos experimentos e estado operacional.
- Docker Compose integra broker, persistencia, Airflow, MLflow, API e dashboard. Uma vez preparados dados e pesos conforme README, um comando inicia a solucao.
- Predicoes em dados novos nao viram rotulos verdadeiros; metricas supervisionadas exigem rotulos de referencia.

### Comunicacao e entrega
- Relatorio final em PDF no formato SBC, sem alterar o template; este documento de planejamento nao e o relatorio final.
- README com nome completo, GitHub, descricao, instalacao e execucao reproduziveis; manter Google Sites entre as entregas previstas na especificacao.
- Preparar apresentacao e demonstracao integradas; formato e duracao a confirmar.
- Entrega anunciada: somente link do repositorio e relatorio PDF, ate 11/12/2026 as 23h59. Confirmar eventuais atualizacoes no Classroom.

---

# 04 | Decisoes resolvidas em 23/11/2026
O aluno informou que o trabalho individual ja era permitido. Revisando o enunciado, a reproducao reduzida, o uso de datasets disponiveis, as tres sementes e a contribuicao com cabecas para 13 e 16 anos ja estao contemplados. A especificacao tambem admite dispositivo simulado. Nao ha necessidade de reenviar uma lista de perguntas sobre essas permissoes.

Adotado: projeto complementar individual, Tema 1, UTKFace, FaRL congelado, tres algoritmos, reproducao idade versus idade + quatro cabecas e contribuicao com duas cabecas adicionais. Mantidos todos os requisitos de integracao e comunicacao.

Nao foi recebida nesta conversa uma nova resposta do professor aprovando um protocolo detalhado. Essas decisoes decorrem do enunciado e da informacao do aluno, nao de uma aprovacao adicional inventada.

As duvidas administrativas sobre pesos (90% no documento), formato da apresentacao e nome completo podem ser esclarecidas durante o projeto; nao bloqueiam a task 02. Se o tema ainda nao tiver sido comunicado ao professor, observar a orientacao geral de alinhar novos projetos, sem transformar decisoes ja permitidas em novos pedidos de autorizacao.

---

# 05 | Fechamento da task 01
### Confirmado com o aluno
- Tema 1 escolhido; projeto complementar de AM; modalidade individual permitida, conforme informado pelo aluno.
- Repositorio e pasta local identificados.
- Nenhum treinamento, download de dataset ou alteracao de implementacao nesta etapa.

### Pendencias que nao impedem o planejamento
- Nome completo do participante e confirmacao do cadastro administrativo.
- Esclarecimentos de pesos da avaliacao e detalhes da apresentacao, sem suspender o desenvolvimento.
- Acesso aos dados resolvido por espelho rastreado, mantendo as restricoes originais. Evidencias tecnicas ficam nas tasks 02 e 03; auditoria completa e treinamento sao etapas posteriores.

### Criterio para iniciar a task 02
Escopo suficiente para iniciar a task 02: definir dados, licencas, protocolo, sementes e hipotese. Nao e necessario solicitar novamente permissoes ja previstas no enunciado.

### Status
Task 01 concluida quanto ao escopo e permissoes discutidas. Nao depende de novos pedidos de autorizacao para escolhas ja previstas no enunciado. Tasks 02 e 03 documentam o protocolo e a preparacao tecnica. Nenhum commit ou push realizado nesta etapa.

### Materiais que fundamentam este planejamento
- Especificacao de Projeto de AM disponibilizada pelo professor: requisitos tecnicos, integracao, rubrica e README.
- Comunicados de entrega, definicao de grupos e temas sugeridos, reproduzidos pelo aluno nesta conversa.
- Artigo do Tema 1: Underage Detection through a Multi-Task and MultiAge Approach for Screening Minors in Unconstrained Imagery (arquivo local tema_03.pdf).
- Template SBC disponibilizado pelo professor para o relatorio final.

Este documento organiza requisitos e propostas; nao representa aprovacao do professor, resultado experimental ou validacao de disponibilidade dos datasets.
