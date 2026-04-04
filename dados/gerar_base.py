import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

conselhos = [
    "Conselho Tutelar Centro",
    "Conselho Tutelar Norte",
    "Conselho Tutelar Sul"
]

tipos_doc = ["Relatório de Atendimento", "Ata de Reunião", "Termo de Ocorrência"]

relatos = [
    # Negligência
    "Criança de 7 anos encontrada sozinha em casa por três dias. Mãe ausentou-se sem deixar alimento ou responsável. Vizinhos acionaram o conselho. Criança apresentava sinais de desnutrição.",
    "Adolescente de 14 anos relatou que não frequenta escola há dois meses. Responsável afirmou não ter condições de levá-lo. Residência sem condições básicas de higiene.",
    "Menor de 5 anos sem vacinação em dia. Família não compareceu a nenhuma consulta pediátrica no último ano. Agente de saúde encaminhou o caso.",
    "Criança de 9 anos trabalhando em feira junto ao pai. Não estava frequentando escola regularmente. Situação de trabalho infantil identificada.",
    "Responsável relatou dificuldade em manter alimentação regular dos três filhos menores. Renda familiar insuficiente. Solicitado encaminhamento ao CRAS.",

    # Violência doméstica
    "Adolescente de 15 anos relatou agressão física pelo padrasto. Marcas visíveis no braço esquerdo. Mãe presente e confirma os fatos. Boletim de ocorrência registrado.",
    "Criança de 8 anos apresentou hematomas na região das costas. Professora da escola acionou o conselho após suspeita de maus-tratos. Pais negam agressão.",
    "Menor relatou violência verbal e psicológica constante por parte da genitora. Professores relataram mudança brusca de comportamento nas últimas semanas.",
    "Vizinhos relataram gritos e barulhos frequentes durante a madrugada. Criança de 6 anos encontrada em estado de agitação e medo extremo na abordagem.",
    "Adolescente de 16 anos fugiu de casa após episódio de violência. Abrigado temporariamente por familiar. Relata histórico de agressões físicas recorrentes.",

    # Evasão escolar
    "Aluno de 11 anos não comparece à escola há 30 dias. Escola tentou contato com família sem sucesso. Equipe do conselho realizou visita domiciliar.",
    "Adolescente de 13 anos abandonou escola após bullying relatado. Família ciente mas sem iniciativa para resolução. Encaminhado para mediação escolar.",
    "Criança de 10 anos matriculada mas com frequência abaixo de 50% no semestre. Responsável alega que criança se recusa a ir. Suspeita de conflito com colegas.",
    "Menor de 12 anos foi retirado da escola pelo pai para trabalhar na roça durante período de colheita. Família de baixa renda em zona rural.",
    "Adolescente grávida de 14 anos afastou-se da escola por vergonha. Escola não ofereceu suporte adequado. Encaminhada à assistência social.",

    # Vulnerabilidade social / racial
    "Família negra em situação de extrema pobreza relatou dificuldade de acesso a serviços públicos. Crianças sem documentação básica. Encaminhamento ao CRAS e cartório.",
    "Adolescente negro de 17 anos abordado repetidamente pela polícia sem motivo aparente. Família relata histórico de discriminação racial na escola e no bairro.",
    "Criança quilombola sem acesso à escola. Comunidade distante do centro urbano. Ausência de transporte escolar. Caso encaminhado à secretaria de educação.",
    "Família migrante com três filhos menores. Crianças sem matrícula escolar por dificuldade de comprovar residência. Apoio para regularização documental solicitado.",
    "Mãe solo negra relatou que filho foi recusado em creche sem justificativa formal. Suspeita de discriminação. Caso encaminhado para apuração.",

    # Saúde mental / abuso sexual
    "Criança de 9 anos apresenta comportamento regressivo, choro excessivo e recusa a frequentar casa do avô paterno. Suspeita de abuso sexual. Caso encaminhado ao CREAS.",
    "Adolescente de 16 anos com histórico de automutilação. Escola acionou conselho após identificar marcas no braço. Encaminhado ao serviço de saúde mental.",
    "Criança relatou toque inadequado por parte de vizinho adulto. Responsável presente e apoiando a criança. Boletim de ocorrência registrado. Medida protetiva solicitada.",
    "Adolescente de 15 anos apresenta sintomas de depressão e ansiedade. Sem acompanhamento psicológico. Família com dificuldade de acesso ao CAPS.",
    "Menor relata abuso sexual pelo genitor. Mãe presente e acredita na criança. Criança afastada do lar e abrigada. Inquérito policial instaurado.",

    # Medidas aplicadas
    "Após investigação, aplicada medida protetiva de afastamento do agressor do lar. Criança permanece com a mãe. Acompanhamento quinzenal estabelecido.",
    "Família encaminhada ao Programa Bolsa Família e ao CRAS para suporte psicossocial. Crianças reintegradas à escola. Monitoramento mensal previsto.",
    "Adolescente encaminhado ao programa de medida socioeducativa em meio aberto após ato infracional. Acompanhamento pelo CREAS iniciado.",
    "Criança retirada do ambiente familiar após confirmação de maus-tratos graves. Acolhimento institucional realizado. Processo judicial em andamento.",
    "Reunião com escola, família e equipe do CRAS para elaboração de plano de atendimento individualizado. Frequência escolar retomada após mediação."
]

categorias = [
    "negligência", "negligência", "negligência", "trabalho infantil", "negligência",
    "violência doméstica", "violência doméstica", "violência doméstica", "violência doméstica", "violência doméstica",
    "evasão escolar", "evasão escolar", "evasão escolar", "trabalho infantil", "evasão escolar",
    "vulnerabilidade social", "racismo", "vulnerabilidade social", "vulnerabilidade social", "racismo",
    "abuso sexual", "saúde mental", "abuso sexual", "saúde mental", "abuso sexual",
    "medida protetiva", "encaminhamento social", "medida socioeducativa", "acolhimento institucional", "mediação"
]

gravidades = [
    "alta", "média", "média", "média", "média",
    "alta", "alta", "média", "alta", "alta",
    "média", "média", "média", "média", "alta",
    "média", "média", "baixa", "baixa", "média",
    "alta", "alta", "alta", "alta", "alta",
    "alta", "média", "alta", "alta", "média"
]

data_base = datetime(2024, 1, 1)
registros = []
for i in range(30):
    data = data_base + timedelta(days=random.randint(0, 365))
    registros.append({
        "id": f"CT-{1000 + i}",
        "data": data.strftime("%Y-%m-%d"),
        "conselho": random.choice(conselhos),
        "tipo_documento": random.choice(tipos_doc),
        "categoria": categorias[i],
        "gravidade": gravidades[i],
        "raca_cor": random.choice(["parda", "preta", "branca", "indígena", "amarela"]),
        "faixa_etaria": random.choice(["0-5", "6-11", "12-14", "15-17"]),
        "texto": relatos[i]
    })

df = pd.DataFrame(registros)
df.to_csv("dados/base_ficticia.csv", index=False, encoding="utf-8")
print(f"Base criada com {len(df)} registros.")
print(df[["id", "categoria", "gravidade", "tipo_documento"]].to_string())