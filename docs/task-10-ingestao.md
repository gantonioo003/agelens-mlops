# AgeLens | Task 10: simulador, MQTT e persistencia

## Fluxo local

O simulador le um arquivo preparado no inbox, calcula seu SHA-256 e publica uma referencia no topico `agelens/v1/images` com QoS 1 e sem mensagem retida. A referencia e um nome de arquivo no volume compartilhado, nao um caminho do Windows. O consumidor le o mesmo volume em modo somente leitura, valida a mensagem e a imagem, e grava os bytes, metadados e estado inicial `pending` em `ingest_events` no PostgreSQL. A API e a inferencia ficam para as etapas seguintes.

O consumidor usa uma sessao MQTT persistente e confirma a mensagem somente depois do commit no PostgreSQL. Uma repeticao com o mesmo UUID e dados iguais nao cria outro registro; um UUID repetido com conteudo diferente e rejeitado. Falhas de validacao entram em `ingest_rejections` como hash da mensagem e codigo do motivo, sem payload. Uma falha de gravacao nao recebe ACK. [Referencia do cliente MQTT](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).

## Contrato v1

| Campo | Formato |
| --- | --- |
| `version` | Inteiro 1 |
| `event_id` | UUID v4 canonico; reutilizar apenas para a mesma mensagem |
| `observed_at` | ISO 8601 UTC, no formato `YYYY-MM-DDTHH:MM:SSZ` |
| `source` | Identificador do simulador, como `sim-device-01` |
| `image_ref` | Somente o nome do arquivo no inbox, sem diretorios |
| `image_sha256` | SHA-256 hexadecimal em minusculas |

O JSON tem limite de 2 KiB. O arquivo deve ser JPEG ou PNG estatico, ate 2 MiB e 2.048 pixels por lado. A extensao deve corresponder ao formato, e os bytes devem corresponder ao hash. Datas anteriores a sete dias ou mais de cinco minutos no futuro sao rejeitadas para novos registros. Nao incluir idade, nome de pessoa ou outros dados pessoais na mensagem, na origem ou no nome de arquivo.

## Executar e conferir

Na raiz do projeto, com Docker Desktop ativo e `.env` da task 03:

```powershell
New-Item -ItemType Directory -Force runtime/ingest/inbox | Out-Null
docker compose up -d --build --wait ingestion
.\.venv\Scripts\python.exe -m tools.verify_ingestion
```

O verificador cria uma imagem sintetica, publica pelo simulador, compara os bytes no banco, verifica duplicata e rejeicoes e testa uma publicacao com o consumidor parado. Depois religa o consumidor e reinicia o broker para testar a retomada. O [comprovante](../reports/task10/verification.json) contem tres IDs sinteticos do teste, sem imagens. O teste encerra com os servicos ativos e remove sua copia de staging; os registros sinteticos continuam no banco local.

Para enviar um exemplo proprio, coloque um JPEG/PNG autorizado e fora dos conjuntos de avaliacao no inbox (por exemplo, `sample.png`) e execute:

```powershell
docker compose run --rm -T simulator python -m services.ingest.simulator --image sample.png
docker compose logs --tail 30 ingestion
docker compose exec -T database psql -U agelens -d agelens -c "SELECT event_id, received_at, status FROM ingest_events ORDER BY received_at DESC LIMIT 10"
```

A publicacao retorna o ID quando o broker a confirma. A consulta ao banco mostra quando a persistencia realmente aconteceu.

## Acesso e retencao

O broker de demonstracao esta limitado a localhost e aceita clientes locais anonimos. PostgreSQL nao publica porta no host. O inbox fica fora do Git e da imagem Docker, mas esta em uma pasta que pode sincronizar via OneDrive: utilize apenas imagens autorizadas para esse armazenamento, preferencialmente sinteticas. Nao exponha broker, inbox, banco ou logs a terceiros. Uma instalacao compartilhada exigiria autenticacao, ACL e TLS.

A politica local e de sete dias para eventos e rejeicoes. A limpeza e manual: `docker compose exec -T ingestion python -m services.ingest.worker --purge-expired`. O comando nao apaga arquivos do inbox. Remova as copias de staging manualmente somente apos confirmar a gravacao e restrinja o acesso ao volume e aos backups. Logs do consumidor contem estado, UUID ou motivo fixo; nao sao registro de imagens. A ingestao e de demonstracao, sem treino, acesso ao teste ou calibracao.
