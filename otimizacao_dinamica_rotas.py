# ==========================================================
# TCC — OTIMIZAÇÃO DINÂMICA DE ROTAS OPERACIONAIS
# ACO + BALANCEAMENTO OPERACIONAL + HTML INTERATIVO
# ==========================================================

# ==========================================================
# IMPORTAÇÕES
# ==========================================================

import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import json
import warnings
import time

'''
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load

file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "new-york-city/nyc-fire-incident-dispatch-data",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
'''

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURAÇÕES OSMNX
# ==========================================================

ox.settings.use_cache = True
ox.settings.log_console = False

# ==========================================================
# CAMINHOS
# ==========================================================

PATH_SHAPE = "caminho df carregado do kaggle"

PATH_BASE = "-"

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

PLACE = "New York City, New York, USA"

N_DIAS = 5

VELOCIDADE_MEDIA = 40

TEMPO_BUFFER_MINUTOS = 45

JORNADA_MAXIMA_HORAS = 8

JORNADA_MAXIMA_SEGUNDOS = (
    JORNADA_MAXIMA_HORAS * 3600
)

JORNADA_ALVO_HORAS = 7.95

JORNADA_ALVO_SEGUNDOS = (
    JORNADA_ALVO_HORAS * 3600
)

JORNADA_MINIMA_DESEJADA_HORAS = 7.75

JORNADA_MINIMA_DESEJADA_SEGUNDOS = (
    JORNADA_MINIMA_DESEJADA_HORAS * 3600
)

COLUNA_ID_OCORRENCIA = "STARFIRE_INCIDENT_ID"

ACO_FORMIGAS = 30

ACO_ITERACOES = 60

ACO_ALPHA = 1

ACO_BETA = 2

ACO_EVAPORACAO = 0.35

ACO_Q = 100


def log_etapa(mensagem):

    agora = pd.Timestamp.now().strftime("%H:%M:%S")

    print(f"[{agora}] {mensagem}", flush=True)

# ==========================================================
# LEITURA SHAPEFILE
# ==========================================================

print("\nLENDO SHAPEFILE...")

inicio = time.perf_counter()

gdf = gpd.read_file(PATH_SHAPE)

log_etapa(
    f"Shapefile lido: {len(gdf)} registros em "
    f"{time.perf_counter() - inicio:.1f}s"
)

gdf["zip"] = (
    gdf["ZCTA5CE20"]
    .astype(str)
    .str.zfill(5)
)

gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

# ==========================================================
# LEITURA BASE
# ==========================================================

print("\nLENDO BASE...")

inicio = time.perf_counter()

base = pd.read_excel(PATH_BASE)

log_etapa(
    f"Base Excel lida: {len(base)} registros em "
    f"{time.perf_counter() - inicio:.1f}s"
)

base["linha_base_excel"] = base.index + 2

if COLUNA_ID_OCORRENCIA not in base.columns:
    COLUNA_ID_OCORRENCIA = "linha_base_excel"

# ==========================================================
# DATAS
# ==========================================================

print("\nTRATANDO DATAS...")

base["INCIDENT_DATETIME"] = pd.to_datetime(
    base["INCIDENT_DATETIME"],
    errors="coerce"
)

base["FIRST_ON_SCENE_DATETIME"] = pd.to_datetime(
    base["FIRST_ON_SCENE_DATETIME"],
    errors="coerce"
)

base["INCIDENT_CLOSE_DATETIME"] = pd.to_datetime(
    base["INCIDENT_CLOSE_DATETIME"],
    errors="coerce"
)

# ==========================================================
# FILTRO
# ==========================================================

base = base.loc[
    base["INCIDENT_DATETIME"] <= "2019-01-05"
]

# ==========================================================
# LIMPEZA
# ==========================================================

base = base.dropna(
    subset=[
        "INCIDENT_DATETIME",
        "FIRST_ON_SCENE_DATETIME",
        "INCIDENT_CLOSE_DATETIME"
    ]
)

# ==========================================================
# DIA
# ==========================================================

base["dia"] = (
    base["INCIDENT_DATETIME"]
    .dt.date
    .astype(str)
)

# ==========================================================
# TEMPO SERVIÇO
# ==========================================================

base["tempo_servico"] = (

    base["INCIDENT_CLOSE_DATETIME"]

    -

    base["FIRST_ON_SCENE_DATETIME"]

).dt.total_seconds()

# ==========================================================
# LIMPEZA
# ==========================================================

print("\nLIMPANDO DADOS...")

base = base.dropna(
    subset=[
        "ZIPCODE",
        "tempo_servico"
    ]
)

base = base[
    base["tempo_servico"] > 0
]

base = base[
    base["tempo_servico"] <= 43200
]

# ==========================================================
# ZIP
# ==========================================================

base["zip"] = (
    base["ZIPCODE"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(5)
)

# mantém apenas 1 zip por dia
base = base.drop_duplicates(
    subset=["dia", "zip"],
    keep="first"
)

# ==========================================================
# MERGE GEO
# ==========================================================

print("\nFAZENDO MERGE GEO...")

base = base.merge(

    gdf[
        ["zip", "lat", "lon"]
    ],

    on="zip",

    how="left"
)

base = base.dropna(
    subset=["lat", "lon"]
)

base.to_excel(r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/base_tratada.xlsx")

print(f"\nREGISTROS GEO VÁLIDOS: {len(base)}")

# ==========================================================
# DIAS
# ==========================================================

dias = sorted(
    base["dia"].unique()
)[:N_DIAS]

log_etapa(
    f"{len(dias)} dias selecionados para processamento"
)

print("\nDIAS PROCESSADOS:")

for d in dias:
    print(d)

# ==========================================================
# MALHA VIÁRIA
# ==========================================================

print("\nBAIXANDO MALHA VIÁRIA...")

inicio = time.perf_counter()

G = ox.graph_from_place(

    PLACE,

    network_type="drive",

    simplify=True,

    retain_all=False
)

log_etapa(
    f"Malha carregada: {len(G.nodes)} nodes, {len(G.edges)} arestas em "
    f"{time.perf_counter() - inicio:.1f}s"
)

print("✔ MALHA CARREGADA")

# ==========================================================
# DISTÂNCIA
# ==========================================================

def distancia_km(lat1, lon1, lat2, lon2):

    return np.sqrt(

        (lat1 - lat2) ** 2 +

        (lon1 - lon2) ** 2

    ) * 111

# ==========================================================
# MATRIZ DISTÂNCIA
# ==========================================================

def criar_matriz(coords):

    n = len(coords)

    matriz = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            if i != j:

                matriz[i][j] = distancia_km(

                    coords[i][0],
                    coords[i][1],

                    coords[j][0],
                    coords[j][1]
                )

    return matriz

# ==========================================================
# FUNCOES DE ROTA
# ==========================================================

def distancia_rota(path, matriz):

    distancia = 0

    for i in range(len(path) - 1):

        distancia += matriz[
            path[i]
        ][path[i + 1]]

    return distancia


def melhorar_rota_2opt(path, matriz):

    melhor_path = path.copy()

    melhor_distancia = distancia_rota(
        melhor_path,
        matriz
    )

    melhorou = True

    while melhorou:

        melhorou = False

        for i in range(1, len(melhor_path) - 2):

            for j in range(i + 1, len(melhor_path)):

                if j - i == 1:
                    continue

                novo_path = (
                    melhor_path[:i]
                    +
                    melhor_path[i:j][::-1]
                    +
                    melhor_path[j:]
                )

                nova_distancia = distancia_rota(
                    novo_path,
                    matriz
                )

                if nova_distancia < melhor_distancia:

                    melhor_path = novo_path

                    melhor_distancia = nova_distancia

                    melhorou = True

        if melhorou:
            continue

    return melhor_path

# ==========================================================
# ACO
# ==========================================================

def otimizar_rota_aco(matriz):

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    log_etapa(
        f"ACO iniciado: {n} pontos, {ACO_FORMIGAS} formigas, "
        f"{ACO_ITERACOES} iteracoes"
    )

    inicio_aco = time.perf_counter()

    matriz_segura = matriz.copy()

    matriz_segura[matriz_segura == 0] = 0.001

    feromonio = np.ones((n, n))

    heuristica = 1 / matriz_segura

    melhor_path = list(range(n))

    melhor_distancia = distancia_rota(
        melhor_path,
        matriz
    )

    for iteracao in range(ACO_ITERACOES):

        rotas_iteracao = []

        for _ in range(ACO_FORMIGAS):

            path = [0]

            visitados = set(path)

            atual = 0

            while len(path) < n:

                nao_visitados = [
                    i for i in range(n)
                    if i not in visitados
                ]

                pesos = np.array([

                    (
                        feromonio[atual][i] ** ACO_ALPHA
                    )
                    *
                    (
                        heuristica[atual][i] ** ACO_BETA
                    )

                    for i in nao_visitados
                ])

                soma_pesos = pesos.sum()

                if soma_pesos == 0:
                    proximo = nao_visitados[0]
                else:
                    probabilidades = pesos / soma_pesos

                    proximo = np.random.choice(
                        nao_visitados,
                        p=probabilidades
                    )

                path.append(int(proximo))

                visitados.add(int(proximo))

                atual = int(proximo)

            distancia = distancia_rota(
                path,
                matriz
            )

            rotas_iteracao.append((
                path,
                distancia
            ))

            if distancia < melhor_distancia:

                melhor_path = path

                melhor_distancia = distancia

        feromonio *= (1 - ACO_EVAPORACAO)

        for path, distancia in rotas_iteracao:

            deposito = ACO_Q / max(distancia, 0.001)

            for i in range(len(path) - 1):

                a = path[i]
                b = path[i + 1]

                feromonio[a][b] += deposito

                feromonio[b][a] += deposito

        if (
            (iteracao + 1) % 10 == 0
            or
            iteracao + 1 == ACO_ITERACOES
        ):

            log_etapa(
                f"ACO iteracao {iteracao + 1}/{ACO_ITERACOES} "
                f"- melhor distancia {melhor_distancia:.2f} km"
            )

    log_etapa(
        f"ACO concluido em {time.perf_counter() - inicio_aco:.1f}s"
    )

    return melhor_path

# ==========================================================
# ACO EXCLUSIVO
# ==========================================================

def otimizar_rota_aco_exclusivo(coords):

    n = len(coords)

    if n <= 2:
        return list(range(n)), "direta"

    log_etapa(f"Criando matriz de distancia para {n} pontos")

    inicio = time.perf_counter()

    matriz = criar_matriz(coords)

    log_etapa(
        f"Matriz criada em {time.perf_counter() - inicio:.1f}s"
    )

    path_aco = otimizar_rota_aco(matriz)

    log_etapa("Aplicando 2OPT")

    inicio = time.perf_counter()

    path_aco = melhorar_rota_2opt(
        path_aco,
        matriz
    )

    log_etapa(
        f"2OPT concluido em {time.perf_counter() - inicio:.1f}s"
    )

    return path_aco, "ACO+2OPT"

# ==========================================================
# BALANCEAMENTO OPERACIONAL
# ==========================================================

def calcular_tempo_total_equipe(servicos):

    if len(servicos) == 0:
        return 0

    servicos = sorted(
        servicos,
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    tempo_total = 0

    for i, servico in enumerate(servicos):

        tempo_total += servico["tempo_servico"]

        if i > 0:

            anterior = servicos[i - 1]

            dist = distancia_km(

                anterior["lat"],
                anterior["lon"],

                servico["lat"],
                servico["lon"]
            )

            tempo_total += (

                dist / VELOCIDADE_MEDIA

            ) * 3600

    return tempo_total


def estimar_tempo_rota_balanceamento(servicos):

    if len(servicos) == 0:
        return 0

    tempo_servicos = sum(
        servico["tempo_servico"] for servico in servicos
    )

    if len(servicos) == 1:
        return tempo_servicos

    nao_visitados = set(
        range(len(servicos))
    )

    atual = min(
        nao_visitados,
        key=lambda indice: servicos[indice]["INCIDENT_DATETIME"]
    )

    nao_visitados.remove(atual)

    distancia_total = 0

    while len(nao_visitados) > 0:

        proximo = min(
            nao_visitados,
            key=lambda indice: distancia_km(
                servicos[atual]["lat"],
                servicos[atual]["lon"],
                servicos[indice]["lat"],
                servicos[indice]["lon"]
            )
        )

        distancia_total += distancia_km(
            servicos[atual]["lat"],
            servicos[atual]["lon"],
            servicos[proximo]["lat"],
            servicos[proximo]["lon"]
        )

        atual = proximo

        nao_visitados.remove(atual)

    tempo_deslocamento = (
        distancia_total / VELOCIDADE_MEDIA
    ) * 3600

    return tempo_servicos + tempo_deslocamento


def equipe_consegue_atender(servicos):

    if len(servicos) == 0:
        return True

    servicos = sorted(
        servicos,
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    tempo_total = estimar_tempo_rota_balanceamento(servicos)

    return tempo_total <= JORNADA_MAXIMA_SEGUNDOS


def atualizar_estado_equipe(equipe):

    equipe["servicos"] = sorted(
        equipe["servicos"],
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    equipe["tempo_total"] = estimar_tempo_rota_balanceamento(
        equipe["servicos"]
    )

    ultimo = equipe["servicos"][-1]

    equipe["disponivel_em"] = (

        ultimo["INCIDENT_DATETIME"]

        +

        pd.to_timedelta(
            ultimo["tempo_servico"],
            unit="s"
        )
    )


def consolidar_equipes(equipes):

    houve_movimento = True
    movimentos = 0

    while houve_movimento:

        houve_movimento = False

        equipes = [
            equipe for equipe in equipes
            if len(equipe["servicos"]) > 0
        ]

        equipes = sorted(
            equipes,
            key=lambda equipe: equipe["tempo_total"]
        )

        for equipe_origem in equipes:

            if equipe_origem["tempo_total"] >= JORNADA_MINIMA_DESEJADA_SEGUNDOS:
                continue

            for servico in list(equipe_origem["servicos"]):

                servicos_origem_restantes = [
                    servico_origem for servico_origem in equipe_origem["servicos"]
                    if servico_origem is not servico
                ]

                tempo_origem_restante = estimar_tempo_rota_balanceamento(
                    servicos_origem_restantes
                )

                candidatos = []

                for equipe_destino in equipes:

                    if equipe_destino is equipe_origem:
                        continue

                    servicos_teste = (
                        equipe_destino["servicos"]
                        +
                        [servico]
                    )

                    tempo_total_teste = estimar_tempo_rota_balanceamento(
                        servicos_teste
                    )

                    if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                        continue

                    deficit_antes = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            equipe_origem["tempo_total"]
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            equipe_destino["tempo_total"]
                        )
                    )

                    deficit_depois = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_origem_restante
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_total_teste
                        )
                    )

                    if (
                        len(servicos_origem_restantes) > 0
                        and
                        deficit_depois >= deficit_antes
                    ):
                        continue

                    dist = distancia_km(

                        equipe_destino["servicos"][-1]["lat"],
                        equipe_destino["servicos"][-1]["lon"],

                        servico["lat"],
                        servico["lon"]
                    )

                    score = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS - tempo_total_teste
                        ),
                        abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                        dist
                    )

                    candidatos.append(
                        (
                            score,
                            equipe_destino
                        )
                    )

                candidatos = sorted(
                    candidatos,
                    key=lambda candidato: candidato[0]
                )

                for _, equipe_destino in candidatos:

                    servicos_teste = (
                        equipe_destino["servicos"]
                        +
                        [servico]
                    )

                    if equipe_consegue_atender(servicos_teste):

                        equipe_destino["servicos"].append(servico)

                        for posicao, servico_origem in enumerate(
                            equipe_origem["servicos"]
                        ):

                            if servico_origem is servico:

                                del equipe_origem["servicos"][posicao]

                                break

                        atualizar_estado_equipe(equipe_destino)

                        if len(equipe_origem["servicos"]) > 0:
                            atualizar_estado_equipe(equipe_origem)
                        else:
                            equipe_origem["tempo_total"] = 0

                        movimentos += 1

                        houve_movimento = True

                        break

                if houve_movimento:
                    break

            if houve_movimento:
                break

    log_etapa(
        f"Consolidacao operacional: {movimentos} ocorrencias realocadas"
    )

    return sorted(
        equipes,
        key=lambda equipe: equipe["tempo_total"],
        reverse=True
    )


def criar_equipe(servico):

    return {

        "tempo_total": servico["tempo_servico"],

        "disponivel_em": (

            servico["INCIDENT_DATETIME"]

            +

            pd.to_timedelta(
                servico["tempo_servico"],
                unit="s"
            )
        ),

        "servicos": [
            servico
        ]
    }


def alocar_equipes_balanceado(df):

    inicio_balanceamento = time.perf_counter()

    servicos = [

        linha for _, linha in df.sort_values(
            by=[
                "tempo_servico",
                "INCIDENT_DATETIME"
            ],
            ascending=[
                False,
                True
            ]
        ).iterrows()
    ]

    total_servicos = len(servicos)

    log_etapa(
        f"Balanceamento iniciado: {total_servicos} ocorrencias pendentes"
    )

    equipes = []

    for indice, servico in enumerate(servicos, start=1):

        melhor_equipe = None
        melhor_score = None

        for equipe in equipes:

            estimativa_minima = (
                equipe["tempo_total"]
                +
                servico["tempo_servico"]
            )

            if estimativa_minima > JORNADA_MAXIMA_SEGUNDOS:
                continue

            servicos_teste = (
                equipe["servicos"]
                +
                [servico]
            )

            tempo_total_teste = estimar_tempo_rota_balanceamento(
                servicos_teste
            )

            if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                continue

            ultimo = equipe["servicos"][-1]

            dist = distancia_km(

                ultimo["lat"],
                ultimo["lon"],

                servico["lat"],
                servico["lon"]
            )

            deficit_jornada = max(
                0,
                JORNADA_ALVO_SEGUNDOS - tempo_total_teste
            )

            score = (
                deficit_jornada,
                abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                dist
            )

            if (
                melhor_score is None
                or
                score < melhor_score
            ):

                melhor_equipe = equipe
                melhor_score = score

        if melhor_equipe is None:

            equipes.append(
                criar_equipe(servico)
            )

        else:

            melhor_equipe["servicos"].append(servico)

            atualizar_estado_equipe(melhor_equipe)

        if (
            indice % 100 == 0
            or
            indice == total_servicos
        ):

            log_etapa(
                f"Balanceamento: {indice}/{total_servicos} ocorrencias, "
                f"{len(equipes)} equipes criadas"
            )

    log_etapa(
        f"Consolidando equipes subutilizadas abaixo de "
        f"{JORNADA_MINIMA_DESEJADA_HORAS:.2f}h"
    )

    equipes = consolidar_equipes(equipes)

    log_etapa(
        f"Balanceamento concluido: {len(equipes)} equipes em "
        f"{time.perf_counter() - inicio_balanceamento:.1f}s"
    )

    return equipes

# ==========================================================
# ESTRUTURAS
# ==========================================================

rotas_json = []

metricas = []

sequencias = []

# ==========================================================
# PROCESSAMENTO
# ==========================================================

for dia in dias:

    inicio_dia = time.perf_counter()

    print("\n================================")
    print(f"PROCESSANDO DIA {dia}")
    print("================================")

    df_dia = base[
        base["dia"] == dia
    ].copy()

    if len(df_dia) < 2:
        continue

    log_etapa(
        f"Dia {dia}: {len(df_dia)} ocorrencias para processar"
    )

    print("\nASSOCIANDO NODES...")

    inicio = time.perf_counter()

    df_dia["node"] = ox.distance.nearest_nodes(

        G,

        X=df_dia["lon"],

        Y=df_dia["lat"]
    )

    log_etapa(
        f"Dia {dia}: nodes associados em "
        f"{time.perf_counter() - inicio:.1f}s"
    )

    equipes = alocar_equipes_balanceado(df_dia)

    print(
        f"\nEQUIPES CRIADAS: {len(equipes)}"
    )

    for equipe_id, equipe in enumerate(equipes):

        print("\n----------------------")
        print(f"EQUIPE {equipe_id}")
        print("----------------------")

        df_eq = pd.DataFrame(
            equipe["servicos"]
        )

        if len(df_eq) < 2:
            log_etapa(
                f"Dia {dia} equipe {equipe_id}: ignorada "
                f"por ter apenas {len(df_eq)} ocorrencia"
            )
            continue

        df_eq = df_eq.reset_index(drop=True)

        coords = list(

            zip(
                df_eq["lat"],
                df_eq["lon"]
            )
        )

        log_etapa(
            f"Dia {dia} equipe {equipe_id}: "
            f"{len(df_eq)} ocorrencias, "
            f"tempo alocado {equipe['tempo_total'] / 3600:.2f}h"
        )

        print("OTIMIZANDO ROTA COM ACO...")

        inicio = time.perf_counter()

        path, metodo_otimizacao = otimizar_rota_aco_exclusivo(coords)

        log_etapa(
            f"Dia {dia} equipe {equipe_id}: ACO finalizado em "
            f"{time.perf_counter() - inicio:.1f}s"
        )

        distancia_total = 0

        for i in range(len(path) - 1):

            a = path[i]
            b = path[i + 1]

            distancia_total += distancia_km(

                coords[a][0],
                coords[a][1],

                coords[b][0],
                coords[b][1]
            )

        tempo_deslocamento = (

            distancia_total

            / VELOCIDADE_MEDIA

        ) * 3600

        tempo_servicos = (
            df_eq["tempo_servico"].sum()
        )

        tempo_buffer_total = (

            len(df_eq)

            *

            TEMPO_BUFFER_MINUTOS

            * 60
        )

        tempo_total = (

            tempo_deslocamento

            +

            tempo_servicos
        )

        tempo_total_com_buffer = (

            tempo_total

            +

            tempo_buffer_total
        )

        jornada_horas = (
            tempo_total / 3600
        )

        rota_viavel = (
            tempo_total <=
            JORNADA_MAXIMA_SEGUNDOS
        )

        rota_coords = []

        log_etapa(
            f"Dia {dia} equipe {equipe_id}: calculando "
            f"{max(len(path) - 1, 0)} trechos na malha viaria"
        )

        inicio = time.perf_counter()

        for i in range(len(path) - 1):

            origem = df_eq.iloc[
                path[i]
            ]["node"]

            destino = df_eq.iloc[
                path[i + 1]
            ]["node"]

            try:

                route = ox.shortest_path(

                    G,

                    origem,

                    destino,

                    weight="length"
                )

                if route is None:
                    continue

                segmento = []

                for node in route:

                    segmento.append([

                        G.nodes[node]["y"],
                        G.nodes[node]["x"]

                    ])

                if len(rota_coords) > 0:
                    segmento = segmento[1:]

                rota_coords.extend(
                    segmento
                )

            except Exception:

                continue

            if (
                (i + 1) % 10 == 0
                or
                i + 1 == len(path) - 1
            ):

                log_etapa(
                    f"Dia {dia} equipe {equipe_id}: "
                    f"{i + 1}/{len(path) - 1} trechos calculados"
                )

        log_etapa(
            f"Dia {dia} equipe {equipe_id}: trechos finalizados em "
            f"{time.perf_counter() - inicio:.1f}s"
        )

        if len(rota_coords) == 0:

            rota_coords = [

                [float(lat), float(lon)]

                for lat, lon in zip(
                    df_eq["lat"],
                    df_eq["lon"]
                )
            ]

        pontos = []

        for ordem, idx in enumerate(path):

            linha = df_eq.iloc[idx]

            if ordem == 0:
                tempo_deslocamento_atendimento = 0
            else:
                idx_anterior = path[ordem - 1]

                dist_atendimento = distancia_km(

                    coords[idx_anterior][0],
                    coords[idx_anterior][1],

                    coords[idx][0],
                    coords[idx][1]
                )

                tempo_deslocamento_atendimento = (

                    dist_atendimento

                    / VELOCIDADE_MEDIA

                ) * 3600

            pontos.append({

                "ordem": ordem + 1,

                "zip": str(
                    linha["zip"]
                ),

                "lat": float(
                    linha["lat"]
                ),

                "lon": float(
                    linha["lon"]
                )
            })

            sequencias.append({

                "dia": dia,

                "equipe": equipe_id,

                "ordem_atendimento": ordem + 1,

                "metodo_otimizacao": metodo_otimizacao,

                "id_ocorrencia": linha[
                    COLUNA_ID_OCORRENCIA
                ],

                "linha_base_excel": linha[
                    "linha_base_excel"
                ],

                "zip": linha["zip"],

                "lat": linha["lat"],

                "lon": linha["lon"],

                "tempo_servico": linha[
                    "tempo_servico"
                ],

                "tempo_deslocamento": tempo_deslocamento_atendimento,

                "tempo_deslocamento_minutos": (
                    tempo_deslocamento_atendimento / 60
                )
            })

        rotas_json.append({

            "dia": dia,

            "equipe": str(equipe_id),

            "metodo_otimizacao": metodo_otimizacao,

            "coords": rota_coords,

            "pontos": pontos,

            "distancia_km": round(
                distancia_total,
                2
            )
        })

        metricas.append({

            "dia": dia,

            "equipe": equipe_id,

            "metodo_otimizacao": metodo_otimizacao,

            "distancia_total_km": round(
                distancia_total,
                2
            ),

            "tempo_total_horas": round(
                jornada_horas,
                2
            ),

            "tempo_total_com_buffer_horas": round(
                tempo_total_com_buffer / 3600,
                2
            ),

            "tempo_atendimento_horas": round(
                tempo_servicos / 3600,
                2
            ),

            "tempo_deslocamento_horas": round(
                tempo_deslocamento / 3600,
                2
            ),

            "quantidade_servicos": len(df_eq),

            "rota_viavel": rota_viavel
        })

    log_etapa(
        f"Dia {dia} concluido em {time.perf_counter() - inicio_dia:.1f}s"
    )

# ==========================================================
# EXPORTAÇÃO
# ==========================================================

print("\nEXPORTANDO EXCEL...")

inicio = time.perf_counter()

df_metricas = pd.DataFrame(metricas)

df_metricas.to_excel(
    "metricas_operacionais.xlsx",
    index=False
)

pd.DataFrame(sequencias).to_excel(
    "sequencia_atendimentos.xlsx",
    index=False
)

log_etapa(
    f"Excel exportado em {time.perf_counter() - inicio:.1f}s"
)

print("✔ EXPORTAÇÕES FINALIZADAS")

# ==========================================================
# HTML INTERATIVO
# ==========================================================

print("\nCRIANDO HTML...")

inicio = time.perf_counter()

centro_lat = base["lat"].mean()
centro_lon = base["lon"].mean()

html = f"""
<!DOCTYPE html>
<html>

<head>

<meta charset='utf-8'/>

<title>Rotas Operacionais ACO</title>

<link rel='stylesheet'
href='https://unpkg.com/leaflet/dist/leaflet.css'/>

<script
src='https://unpkg.com/leaflet/dist/leaflet.js'>
</script>

<style>

body {{
    margin: 0;
}}

#map {{
    width: 100%;
    height: 100vh;
}}

.controls {{

    position: absolute;

    top: 10px;
    left: 10px;

    z-index: 9999;

    background: white;

    padding: 12px;

    border-radius: 10px;

    box-shadow: 0 0 15px rgba(0,0,0,0.3);

    font-family: Arial;
}}

.legend {{
    margin-top: 10px;
}}

#equipes {{
    width: 150px;
}}

.team-actions {{
    margin-top: 6px;
}}

.legend-item {{
    margin-bottom: 5px;
}}

.legend-color {{

    display: inline-block;

    width: 12px;
    height: 12px;

    margin-right: 5px;
}}

</style>

</head>

<body>

<div class='controls'>

<h3>Rotas Operacionais</h3>

<label>Dia:</label>

<select id='dia' onchange='atualizarEquipes()'></select>

<br><br>

<label>Equipes:</label>

<br>

<select id='equipes' multiple size='8'></select>

<div class='team-actions'>

<button onclick='selecionarTodasEquipes()'>
Todas
</button>

<button onclick='limparEquipes()'>
Limpar
</button>

</div>

<br><br>

<button onclick='plotar()'>
Plotar Rotas
</button>

<div class='legend' id='legend'></div>

</div>

<div id='map'></div>

<script>

var rotas = {json.dumps(rotas_json)};

var cores = [

    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "black",
    "brown",
    "pink",
    "darkred",
    "cadetblue",
    "darkgreen",
    "gray",
    "gold"
];

var map = L.map('map').setView(
    [{centro_lat}, {centro_lon}],
    10
);

L.tileLayer(
    'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{
        maxZoom: 19
    }}
).addTo(map);

var dias = [...new Set(
    rotas.map(r => r.dia)
)];

dias.forEach(d => {{

    var option =
    document.createElement('option');

    option.value = d;
    option.text = d;

    document
    .getElementById('dia')
    .appendChild(option);
}});

function atualizarEquipes() {{

    var dia = document
        .getElementById('dia').value;

    var selectEquipes = document
        .getElementById('equipes');

    selectEquipes.innerHTML = "";

    var equipesDia = [...new Set(
        rotas
            .filter(r => r.dia == dia)
            .map(r => r.equipe)
    )];

    equipesDia.forEach(equipe => {{

        var option =
        document.createElement('option');

        option.value = equipe;
        option.text = "Equipe " + equipe;
        option.selected = true;

        selectEquipes.appendChild(option);
    }});
}}

atualizarEquipes();

function selecionarTodasEquipes() {{

    Array.from(
        document.getElementById('equipes').options
    ).forEach(option => {{
        option.selected = true;
    }});
}}

function limparEquipes() {{

    Array.from(
        document.getElementById('equipes').options
    ).forEach(option => {{
        option.selected = false;
    }});
}}

var layersRotas = [];

var layersMarkers = [];

function limparMapa() {{

    layersRotas.forEach(l => {{
        map.removeLayer(l);
    }});

    layersMarkers.forEach(l => {{
        map.removeLayer(l);
    }});

    layersRotas = [];
    layersMarkers = [];
}}

function plotar() {{

    limparMapa();

    document.getElementById(
        "legend"
    ).innerHTML = "";

    var dia = document
        .getElementById('dia').value;

    var equipesSelecionadas = Array.from(
        document.getElementById('equipes').selectedOptions
    ).map(option => option.value);

    var rotasDia = rotas.filter(
        r => (
            r.dia == dia
            &&
            equipesSelecionadas.includes(r.equipe)
        )
    );

    if (rotasDia.length == 0) {{

        alert("Nenhuma rota encontrada para a seleção");

        return;
    }}

    rotasDia.forEach((rota, idx) => {{

        var cor = cores[
            idx % cores.length
        ];

        var poly = L.polyline(

            rota.coords,

            {{
                color: cor,
                weight: 5
            }}

        ).addTo(map);

        layersRotas.push(poly);

        rota.pontos.forEach(p => {{

            var marker = L.circleMarker(

                [p.lat, p.lon],

                {{
                    radius: 6,
                    color: cor,
                    fillColor: cor,
                    fillOpacity: 1
                }}

            ).addTo(map);

            marker.bindPopup(

                "<b>Equipe:</b> " +

                rota.equipe +

                "<br>" +

                "<b>Ordem:</b> " +

                p.ordem +

                "<br>" +

                "<b>ZIP:</b> " +

                p.zip
            );

            layersMarkers.push(marker);
        }});

        var legenda = document.createElement(
            "div"
        );

        legenda.className = "legend-item";

        legenda.innerHTML =

            "<span class='legend-color' " +

            "style='background:" + cor + "'></span>" +

            "Equipe " + rota.equipe;

        document.getElementById(
            "legend"
        ).appendChild(legenda);

    }});

    var grupo = L.featureGroup(
        layersRotas
    );

    map.fitBounds(
        grupo.getBounds()
    );
}}

</script>

</body>
</html>
"""

with open(

    "rotas_interativas.html",

    "w",

    encoding="utf-8"

) as f:

    f.write(html)

print("✔ rotas_interativas.html")

log_etapa(
    f"HTML criado em {time.perf_counter() - inicio:.1f}s"
)
