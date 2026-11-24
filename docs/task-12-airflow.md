# AgeLens | Task 12: processamento automatico

## Fluxos

| DAG | Agenda | Entrada | Saida |
| --- | --- | --- | --- |
| `agelens_operational_inference` | A cada minuto | Eventos pendentes no PostgreSQL, recebidos pelo MQTT | Resultado e estado final com o mesmo ID do MQTT |
| `agelens_prepare_labeled_train` | Diaria | Manifest auditado de treino e features congeladas | Snapshot privado com features, idades anotadas e folds de treino |

O DAG operacional valida o tipo, o tamanho e o hash da imagem armazenada. Envia os bytes para a API local, que decodifica a imagem, extrai as features FaRL e aplica o modelo R0 aprovado. O PostgreSQL registra a estimativa, as seis decisoes, a versao, o run do MLflow, o tempo e o horario. O Airflow guarda apenas contagens agregadas em XCom, nao imagens ou previsoes individuais.

O DAG de treino e independente: seleciona somente as 9.001 linhas de treino, salva `train.npz` com `features`, `age_years` e `folds` e confere os hashes na reexecucao. A exportacao fica em `/opt/airflow/prepared/<cache-id>/`, no volume privado do Airflow. Nenhum evento operacional, previsao, rotulo do teste ou da calibracao entra no snapshot. O hash integral da matriz e conferido unicamente para integridade.

## Reprocessamento

Cada evento e reservado por ate 120 segundos com um token de processamento. Uma transacao grava o estado `completed` e o resultado juntos. Falhas transitorias voltam a `pending` com intervalo de 15 ou 30 segundos antes de ficarem elegiveis no proximo ciclo. A terceira falha, ou uma entrada rejeitada definitivamente, fica visivel como `failed` com motivo. Em caso de interrupcao do worker, a reserva expira e outro ciclo pode retomar.

A API aceita `X-Event-ID`: o mesmo ID com os mesmos bytes e tipo retorna o resultado existente, inclusive depois de um timeout de quem chamou; o mesmo ID com outro conteudo recebe 409. A tabela `inference_results` tambem e unica por evento. Os IDs da avaliacao continuam separados. A limpeza de eventos antigos da task 10 remove seus resultados no PostgreSQL em cascata.

## Executar localmente

Inicie a API da task 11 em um terminal e deixe-a ativa. O Airflow no Docker Desktop acessa a API pelo host, sem abrir novas portas na rede. Em outro terminal, com a base e o cache das tasks anteriores disponiveis:

```powershell
docker compose up -d --build --wait airflow
.\.venv\Scripts\python.exe -m tools.verify_orchestration
```

[Airflow local](http://127.0.0.1:18090) usa o usuario `agelens`. A senha gerada fica no arquivo privado `/opt/airflow/simple_auth_manager_passwords.json.generated`; nao compartilhe a senha ou os logs de inicializacao. Este e um ambiente de demonstracao local, nao uma implantacao de producao. [Airflow standalone](https://airflow.apache.org/docs/apache-airflow/3.1.8/start.html) · [Gerenciador de autenticacao local](https://airflow.apache.org/docs/apache-airflow/3.1.8/core-concepts/auth-manager/simple/index.html).

O verificador publica uma imagem sintetica e aguarda o DAG agendado, sem acionar uma execucao manual. Depois confirma o resultado no PostgreSQL, repete a mensagem, verifica que nao houve duplicata e confere o snapshot privado do treino. [Comprovante](../reports/task12/verification.json).

Na verificacao, o evento novo foi concluido em uma tentativa e teve exatamente um resultado persistido, com o mesmo UUID no MQTT, na API e no PostgreSQL. Foram testados tambem o reaproveitamento do snapshot, a rejeicao de um replay com bytes diferentes e, em tabela temporaria do PostgreSQL, o atraso de retentativa e a recuperacao de reservas expiradas. 84 testes locais passaram.
