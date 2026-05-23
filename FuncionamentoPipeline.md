base_ficticia.csv
      ↓
  Script 01        → organiza e estrutura o corpus
      ↓
  Script 02        → limpa, tokeniza e lematiza os textos
      ↓
  Script 03        → análise exploratória: você entende o corpus antes de modelar
      ↓
  Script 04        → treina o LDA, escolhe o melhor número de tópicos, salva o modelo
      ↓
  Script 05        → treina o BERTopic com BERTimbau, salva o modelo
      ↓
  Script 06        → treina o classificador de gravidade (SVM + fine-tuning BERTimbau), salva os modelos
      ↓
  Script 08        → valida todos os modelos com métricas (F1, acurácia, C_v)
      ↓
  modelos salvos em disco → prontos para a fase de produção

---


documento novo (PDF ou CSV do SIPIA)
      ↓
  anonimização (Presidio)
      ↓
  pré-processamento (mesmo script 02, só a função preprocessar())
      ↓
  modelo LDA carregado → extrai tópico do documento
  modelo BERTopic carregado → extrai tópico semântico
  modelo SVM/BERTimbau carregado → classifica gravidade e sentimento
      ↓
  resultado armazenado no banco de dados
      ↓
  dashboard atualizado automaticamente


---


Etapa 2 — Rotulação manual dos dados reais
Essa é a etapa mais importante e que mais impacta a qualidade dos modelos. Quando os documentos reais chegarem do SIPIA, antes de treinar qualquer modelo, você vai precisar rotular manualmente uma amostra.
O processo é o seguinte:
1. Selecione entre 150 e 200 documentos reais de forma aleatória e estratificada — garantindo representatividade de diferentes Conselhos Tutelares, períodos e tipos de caso.
2. Para cada documento, um especialista (você junto com os conselheiros tutelares) define:

A categoria do caso (negligência, violência, evasão etc.)
O nível de gravidade (alta, média, baixa)
Se há marcador racial explícito ou implícito no texto

3. Esses documentos rotulados vão para uma planilha com a estrutura:
id | texto | categoria | gravidade | marcador_racial
4. Essa planilha se torna o conjunto de treinamento supervisionado dos modelos de classificação.
O número mínimo recomendado por classe é 50 exemplos. Abaixo disso os modelos não generalizam bem. Então para gravidade (alta, média, baixa) você precisa de pelo menos 150 exemplos rotulados no total.

Etapa 3 — Retreinar com os dados reais rotulados
Com os documentos reais rotulados em mãos, você repete toda a pipeline substituindo a base fictícia pela base real. Os modelos não supervisionados (LDA e BERTopic) vão ser retreinados em todo o corpus real — não apenas nos rotulados. Os modelos supervisionados (SVM e BERTimbau fine-tuned) vão ser treinados apenas com os exemplos rotulados.
A divisão recomendada para os dados rotulados é:
80% para treino → o modelo aprende
10% para validação → você ajusta os parâmetros
10% para teste final → você mede o desempenho real
Nunca use o conjunto de teste para ajustar o modelo — ele deve ser visto apenas uma vez, no final, para medir o desempenho real sem viés.