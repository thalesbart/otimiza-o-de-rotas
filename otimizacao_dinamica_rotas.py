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
import math
import os
from scipy.stats import wilcoxon

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURAÇÕES OSMNX
# ==========================================================

ox.settings.use_cache = True
ox.settings.log_console = False

# ==========================================================
# CAMINHOS
# ==========================================================

PATH_SHAPE = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/tl_2023_us_zcta520/tl_2023_us_zcta520.shp"

PATH_BASE = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/archive/fire-incident-dispatch-data.xlsx"

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

PLACE = "New York City, New York, USA"

N_DIAS = 30

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

PESO_DISTANCIA = 1.0

PESO_JORNADA_EXCEDIDA = 3.0

PESO_OCIOSIDADE = 0.8

PESO_DESBALANCEAMENTO = 1.5

NIVEL_CONFIANCA_Z = 1.96


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

if len(dias) < N_DIAS:

    log_etapa(
        f"AVISO: apenas {len(dias)} dias validos encontrados "
        f"apos limpeza/merge geo; solicitado N_DIAS={N_DIAS}"
    )

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


def caminho_fifo(df_eq):

    return list(
        df_eq.sort_values("INCIDENT_DATETIME").index
    )


def caminho_nearest_neighbor(matriz, inicio=0):

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    nao_visitados = set(range(n))

    atual = inicio

    path = [atual]

    nao_visitados.remove(atual)

    while len(nao_visitados) > 0:

        proximo = min(
            nao_visitados,
            key=lambda indice: matriz[atual][indice]
        )

        path.append(proximo)

        nao_visitados.remove(proximo)

        atual = proximo

    return path


def avaliar_caminho(path, df_eq, coords, matriz):

    distancia_total = distancia_rota(
        path,
        matriz
    )

    tempo_deslocamento = (
        distancia_total / VELOCIDADE_MEDIA
    ) * 3600

    tempo_servicos = df_eq["tempo_servico"].sum()

    tempo_total = tempo_servicos + tempo_deslocamento

    return {
        "distancia_total_km": distancia_total,
        "tempo_deslocamento": tempo_deslocamento,
        "tempo_servicos": tempo_servicos,
        "tempo_total": tempo_total,
        "jornada_horas": tempo_total / 3600,
        "rota_viavel": tempo_total <= JORNADA_MAXIMA_SEGUNDOS
    }


def calcular_funcao_objetivo(
    distancia_km_total,
    jornada_excedida_horas,
    ociosidade_horas,
    desbalanceamento_horas
):

    return (
        PESO_DISTANCIA * distancia_km_total
        +
        PESO_JORNADA_EXCEDIDA * jornada_excedida_horas
        +
        PESO_OCIOSIDADE * ociosidade_horas
        +
        PESO_DESBALANCEAMENTO * desbalanceamento_horas
    )


def avaliar_baselines_rotas(df_eq, coords):

    matriz = criar_matriz(coords)

    path_fifo = caminho_fifo(df_eq)

    path_nn = caminho_nearest_neighbor(matriz)

    path_aco = otimizar_rota_aco(matriz)

    path_aco_2opt = melhorar_rota_2opt(
        path_aco,
        matriz
    )

    baselines = {
        "FIFO": path_fifo,
        "Nearest Neighbor": path_nn,
        "ACO sem 2OPT": path_aco,
        "ACO + 2OPT": path_aco_2opt
    }

    resultados = {}

    for metodo, path in baselines.items():

        resultados[metodo] = {
            "path": path,
            "metricas": avaliar_caminho(
                path,
                df_eq,
                coords,
                matriz
            )
        }

    return resultados, matriz

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


def absorver_equipes_subutilizadas(equipes):

    equipes = [
        equipe for equipe in equipes
        if len(equipe["servicos"]) > 0
    ]

    absorvidas = 0

    houve_absorcao = True

    while houve_absorcao:

        houve_absorcao = False

        for indice_origem, equipe_origem in sorted(
            list(enumerate(equipes)),
            key=lambda item: item[1]["tempo_total"]
        ):

            if equipe_origem["tempo_total"] >= JORNADA_MINIMA_DESEJADA_SEGUNDOS:
                continue

            if len(equipes) <= 1:
                break

            estado_teste = {
                indice: list(equipe["servicos"])
                for indice, equipe in enumerate(equipes)
                if indice != indice_origem
            }

            servicos_origem = sorted(
                equipe_origem["servicos"],
                key=lambda servico: servico["tempo_servico"],
                reverse=True
            )

            plano_viavel = True

            for servico in servicos_origem:

                candidatos = []

                for indice_destino, servicos_destino in estado_teste.items():

                    servicos_teste = (
                        servicos_destino
                        +
                        [servico]
                    )

                    tempo_total_teste = estimar_tempo_rota_balanceamento(
                        servicos_teste
                    )

                    if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                        continue

                    if len(servicos_destino) > 0:

                        ultimo = servicos_destino[-1]

                        dist = distancia_km(
                            ultimo["lat"],
                            ultimo["lon"],
                            servico["lat"],
                            servico["lon"]
                        )

                    else:

                        dist = 0

                    score = (
                        max(
                            0,
                            JORNADA_ALVO_SEGUNDOS - tempo_total_teste
                        ),
                        abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                        dist,
                        len(servicos_destino)
                    )

                    candidatos.append(
                        (
                            score,
                            indice_destino,
                            tempo_total_teste
                        )
                    )

                if len(candidatos) == 0:

                    plano_viavel = False

                    break

                _, melhor_destino, _ = min(
                    candidatos,
                    key=lambda candidato: candidato[0]
                )

                estado_teste[melhor_destino].append(servico)

            if not plano_viavel:
                continue

            for indice_destino, servicos_destino in estado_teste.items():

                equipes[indice_destino]["servicos"] = servicos_destino

                atualizar_estado_equipe(
                    equipes[indice_destino]
                )

            equipe_origem["servicos"] = []
            equipe_origem["tempo_total"] = 0

            absorvidas += 1
            houve_absorcao = True

            equipes = [
                equipe for equipe in equipes
                if len(equipe["servicos"]) > 0
            ]

            break

    return equipes, absorvidas


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

            folga_jornada = max(
                0,
                JORNADA_MAXIMA_SEGUNDOS - tempo_total_teste
            )

            score = (
                deficit_jornada,
                abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                folga_jornada,
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

    equipes, equipes_absorvidas = absorver_equipes_subutilizadas(
        equipes
    )

    log_etapa(
        f"Compactacao inicial: {equipes_absorvidas} equipes absorvidas"
    )

    equipes = consolidar_equipes(equipes)

    equipes, equipes_absorvidas = absorver_equipes_subutilizadas(
        equipes
    )

    log_etapa(
        f"Compactacao final: {equipes_absorvidas} equipes absorvidas"
    )

    log_etapa(
        f"Balanceamento concluido: {len(equipes)} equipes em "
        f"{time.perf_counter() - inicio_balanceamento:.1f}s"
    )

    return equipes


def alocar_equipes_fifo(df):

    equipes = []

    servicos = [
        linha for _, linha in df.sort_values(
            "INCIDENT_DATETIME"
        ).iterrows()
    ]

    for servico in servicos:

        alocado = False

        for equipe in equipes:

            servicos_teste = (
                equipe["servicos"]
                +
                [servico]
            )

            if equipe_consegue_atender(servicos_teste):

                equipe["servicos"].append(servico)

                atualizar_estado_equipe(equipe)

                alocado = True

                break

        if not alocado:

            equipes.append(
                criar_equipe(servico)
            )

    return equipes


def calcular_metricas_balanceamento(equipes):

    cargas = np.array([
        estimar_tempo_rota_balanceamento(equipe["servicos"]) / 3600
        for equipe in equipes
        if len(equipe["servicos"]) > 0
    ])

    if len(cargas) == 0:
        cargas = np.array([0])

    media_carga = cargas.mean()

    desvio_carga = cargas.std(ddof=0)

    coeficiente_variacao = (
        desvio_carga / media_carga
        if media_carga > 0
        else 0
    )

    tempo_ocioso = np.maximum(
        0,
        JORNADA_MAXIMA_HORAS - cargas
    )

    diferenca_carga = cargas.max() - cargas.min()

    jornada_excedida = np.maximum(
        0,
        cargas - JORNADA_MAXIMA_HORAS
    )

    equipes_baixa_alocacao = (
        cargas < JORNADA_MINIMA_DESEJADA_HORAS
    ).sum()

    return {
        "equipes_usadas": len(cargas),
        "carga_media_horas": media_carga,
        "carga_desvio_padrao_horas": desvio_carga,
        "coeficiente_variacao_carga": coeficiente_variacao,
        "tempo_ocioso_total_horas": tempo_ocioso.sum(),
        "tempo_ocioso_medio_horas": tempo_ocioso.mean(),
        "diferenca_mais_menos_carregada_horas": diferenca_carga,
        "jornada_excedida_total_horas": jornada_excedida.sum(),
        "equipes_baixa_alocacao": equipes_baixa_alocacao,
        "percentual_equipes_baixa_alocacao": (
            equipes_baixa_alocacao / len(cargas) * 100
            if len(cargas) > 0
            else 0
        ),
        "cargas_horas": cargas.tolist()
    }


def intervalo_confianca_95(valores):

    valores = pd.Series(valores).dropna()

    if len(valores) == 0:
        return 0

    if len(valores) == 1:
        return 0

    return (
        NIVEL_CONFIANCA_Z
        *
        valores.std(ddof=1)
        /
        math.sqrt(len(valores))
    )


def resumir_por_metodo(df, coluna_metodo):

    registros = []

    if len(df) == 0:
        return pd.DataFrame()

    metricas_resumo = [
        "distancia_total_km",
        "tempo_deslocamento_horas",
        "carga_media_horas",
        "equipes_usadas",
        "carga_desvio_padrao_horas",
        "coeficiente_variacao_carga",
        "tempo_ocioso_total_horas",
        "diferenca_mais_menos_carregada_horas",
        "equipes_baixa_alocacao",
        "percentual_equipes_baixa_alocacao",
        "funcao_objetivo"
    ]

    for metodo, grupo in df.groupby(coluna_metodo):

        registro = {
            coluna_metodo: metodo,
            "dias": grupo["dia"].nunique()
        }

        for metrica in metricas_resumo:

            if metrica not in grupo.columns:
                continue

            registro[f"{metrica}_media"] = grupo[metrica].mean()
            registro[f"{metrica}_desvio_padrao"] = grupo[metrica].std(ddof=1)
            registro[f"{metrica}_ic95"] = intervalo_confianca_95(
                grupo[metrica]
            )

        registros.append(registro)

    return pd.DataFrame(registros)

# ==========================================================
# ESTRUTURAS
# ==========================================================

rotas_json = []

metricas = []

sequencias = []

metricas_baselines = []

metricas_balanceamento = []

metricas_ablation = []

cargas_equipes_boxplot = []

comparativo_2opt = []

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

    equipes_fifo = alocar_equipes_fifo(df_dia)

    balanceamento_fifo = calcular_metricas_balanceamento(
        equipes_fifo
    )

    equipes = alocar_equipes_balanceado(df_dia)

    balanceamento_final = calcular_metricas_balanceamento(
        equipes
    )

    for nome_cenario, dados_balanceamento in [
        ("Sem balanceamento", balanceamento_fifo),
        ("Com balanceamento", balanceamento_final)
    ]:

        metricas_balanceamento.append({
            "dia": dia,
            "cenario": nome_cenario,
            "equipes_usadas": dados_balanceamento["equipes_usadas"],
            "carga_media_horas": dados_balanceamento["carga_media_horas"],
            "carga_desvio_padrao_horas": dados_balanceamento[
                "carga_desvio_padrao_horas"
            ],
            "coeficiente_variacao_carga": dados_balanceamento[
                "coeficiente_variacao_carga"
            ],
            "tempo_ocioso_total_horas": dados_balanceamento[
                "tempo_ocioso_total_horas"
            ],
            "tempo_ocioso_medio_horas": dados_balanceamento[
                "tempo_ocioso_medio_horas"
            ],
            "diferenca_mais_menos_carregada_horas": dados_balanceamento[
                "diferenca_mais_menos_carregada_horas"
            ],
            "jornada_excedida_total_horas": dados_balanceamento[
                "jornada_excedida_total_horas"
            ],
            "equipes_baixa_alocacao": dados_balanceamento[
                "equipes_baixa_alocacao"
            ],
            "percentual_equipes_baixa_alocacao": dados_balanceamento[
                "percentual_equipes_baixa_alocacao"
            ]
        })

        for carga in dados_balanceamento["cargas_horas"]:

            cargas_equipes_boxplot.append({
                "dia": dia,
                "cenario": nome_cenario,
                "carga_horas": carga
            })

    acumulado_baselines_dia = {}

    sem_balanceamento_fifo_dia = {
        "distancia_total_km": 0,
        "tempo_deslocamento_horas": 0,
        "tempo_total_horas": 0,
        "quantidade_servicos": 0
    }

    for equipe_fifo in equipes_fifo:

        df_eq_fifo = pd.DataFrame(
            equipe_fifo["servicos"]
        ).reset_index(drop=True)

        coords_fifo = list(
            zip(
                df_eq_fifo["lat"],
                df_eq_fifo["lon"]
            )
        )

        matriz_fifo = criar_matriz(coords_fifo)

        path_fifo = caminho_fifo(df_eq_fifo)

        metrica_fifo = avaliar_caminho(
            path_fifo,
            df_eq_fifo,
            coords_fifo,
            matriz_fifo
        )

        sem_balanceamento_fifo_dia["distancia_total_km"] += (
            metrica_fifo["distancia_total_km"]
        )

        sem_balanceamento_fifo_dia["tempo_deslocamento_horas"] += (
            metrica_fifo["tempo_deslocamento"] / 3600
        )

        sem_balanceamento_fifo_dia["tempo_total_horas"] += (
            metrica_fifo["tempo_total"] / 3600
        )

        sem_balanceamento_fifo_dia["quantidade_servicos"] += len(df_eq_fifo)

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

            df_eq = df_eq.reset_index(drop=True)

            coords = list(
                zip(
                    df_eq["lat"],
                    df_eq["lon"]
                )
            )

            matriz = criar_matriz(coords)

            resultados_baselines = {
                metodo: {
                    "path": list(range(len(df_eq))),
                    "metricas": avaliar_caminho(
                        list(range(len(df_eq))),
                        df_eq,
                        coords,
                        matriz
                    )
                }
                for metodo in [
                    "FIFO",
                    "Nearest Neighbor",
                    "ACO sem 2OPT",
                    "ACO + 2OPT"
                ]
            }

            for metodo_baseline, dados_baseline in resultados_baselines.items():

                metricas_baseline = dados_baseline["metricas"]

                if metodo_baseline not in acumulado_baselines_dia:

                    acumulado_baselines_dia[metodo_baseline] = {
                        "distancia_total_km": 0,
                        "tempo_deslocamento_horas": 0,
                        "tempo_total_horas": 0,
                        "quantidade_servicos": 0
                    }

                acumulado_baselines_dia[metodo_baseline][
                    "tempo_total_horas"
                ] += (
                    metricas_baseline["tempo_total"] / 3600
                )

                acumulado_baselines_dia[metodo_baseline][
                    "quantidade_servicos"
                ] += len(df_eq)

                metricas_baselines.append({
                    "dia": dia,
                    "equipe": equipe_id,
                    "metodo": metodo_baseline,
                    "distancia_total_km": 0,
                    "tempo_deslocamento_horas": 0,
                    "tempo_total_horas": round(
                        metricas_baseline["tempo_total"] / 3600,
                        2
                    ),
                    "quantidade_servicos": len(df_eq),
                    "rota_viavel": metricas_baseline["rota_viavel"]
                })

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

        print("CALCULANDO BASELINES DE ROTA...")

        inicio = time.perf_counter()

        resultados_baselines, matriz = avaliar_baselines_rotas(
            df_eq,
            coords
        )

        metodo_otimizacao = "ACO + 2OPT"

        path = resultados_baselines[
            metodo_otimizacao
        ]["path"]

        log_etapa(
            f"Dia {dia} equipe {equipe_id}: baselines finalizados em "
            f"{time.perf_counter() - inicio:.1f}s"
        )

        for metodo_baseline, dados_baseline in resultados_baselines.items():

            metricas_baseline = dados_baseline["metricas"]

            if metodo_baseline not in acumulado_baselines_dia:

                acumulado_baselines_dia[metodo_baseline] = {
                    "distancia_total_km": 0,
                    "tempo_deslocamento_horas": 0,
                    "tempo_total_horas": 0,
                    "quantidade_servicos": 0
                }

            acumulado_baselines_dia[metodo_baseline]["distancia_total_km"] += (
                metricas_baseline["distancia_total_km"]
            )

            acumulado_baselines_dia[metodo_baseline][
                "tempo_deslocamento_horas"
            ] += (
                metricas_baseline["tempo_deslocamento"] / 3600
            )

            acumulado_baselines_dia[metodo_baseline]["tempo_total_horas"] += (
                metricas_baseline["tempo_total"] / 3600
            )

            acumulado_baselines_dia[metodo_baseline][
                "quantidade_servicos"
            ] += len(df_eq)

            metricas_baselines.append({
                "dia": dia,
                "equipe": equipe_id,
                "metodo": metodo_baseline,
                "distancia_total_km": round(
                    metricas_baseline["distancia_total_km"],
                    2
                ),
                "tempo_deslocamento_horas": round(
                    metricas_baseline["tempo_deslocamento"] / 3600,
                    2
                ),
                "tempo_total_horas": round(
                    metricas_baseline["tempo_total"] / 3600,
                    2
                ),
                "quantidade_servicos": len(df_eq),
                "rota_viavel": metricas_baseline["rota_viavel"]
            })

        distancia_aco = resultados_baselines[
            "ACO sem 2OPT"
        ]["metricas"]["distancia_total_km"]

        distancia_2opt = resultados_baselines[
            "ACO + 2OPT"
        ]["metricas"]["distancia_total_km"]

        comparativo_2opt.append({
            "dia": dia,
            "equipe": equipe_id,
            "antes_2opt_km": distancia_aco,
            "depois_2opt_km": distancia_2opt,
            "ganho_2opt_percentual": (
                (distancia_aco - distancia_2opt)
                / distancia_aco
                * 100
                if distancia_aco > 0
                else 0
            )
        })

        metrica_final = resultados_baselines[
            metodo_otimizacao
        ]["metricas"]

        distancia_total = metrica_final["distancia_total_km"]

        tempo_deslocamento = metrica_final["tempo_deslocamento"]

        tempo_servicos = (
            df_eq["tempo_servico"].sum()
        )

        tempo_buffer_total = (

            len(df_eq)

            *

            TEMPO_BUFFER_MINUTOS

            * 60
        )

        tempo_total = metrica_final["tempo_total"]

        tempo_total_com_buffer = (

            tempo_total

            +

            tempo_buffer_total
        )

        jornada_horas = metrica_final["jornada_horas"]

        rota_viavel = metrica_final["rota_viavel"]

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

    referencia_fifo = acumulado_baselines_dia.get(
        "FIFO",
        {}
    )

    for metodo_baseline, totais in acumulado_baselines_dia.items():

        distancia_ref = referencia_fifo.get(
            "distancia_total_km",
            0
        )

        tempo_ref = referencia_fifo.get(
            "tempo_deslocamento_horas",
            0
        )

        reducao_distancia = (
            (distancia_ref - totais["distancia_total_km"])
            / distancia_ref
            * 100
            if distancia_ref > 0
            else 0
        )

        reducao_tempo = (
            (tempo_ref - totais["tempo_deslocamento_horas"])
            / tempo_ref
            * 100
            if tempo_ref > 0
            else 0
        )

        objetivo = calcular_funcao_objetivo(
            totais["distancia_total_km"],
            balanceamento_final["jornada_excedida_total_horas"],
            balanceamento_final["tempo_ocioso_total_horas"],
            balanceamento_final["diferenca_mais_menos_carregada_horas"]
        )

        metricas_ablation.append({
            "dia": dia,
            "cenario": f"Balanceamento + {metodo_baseline}",
            "metodo_rota": metodo_baseline,
            "distancia_total_km": totais["distancia_total_km"],
            "tempo_deslocamento_horas": totais[
                "tempo_deslocamento_horas"
            ],
            "tempo_total_horas": totais["tempo_total_horas"],
            "quantidade_servicos": totais["quantidade_servicos"],
            "equipes_usadas": balanceamento_final["equipes_usadas"],
            "carga_media_horas": balanceamento_final["carga_media_horas"],
            "carga_desvio_padrao_horas": balanceamento_final[
                "carga_desvio_padrao_horas"
            ],
            "coeficiente_variacao_carga": balanceamento_final[
                "coeficiente_variacao_carga"
            ],
            "tempo_ocioso_total_horas": balanceamento_final[
                "tempo_ocioso_total_horas"
            ],
            "diferenca_mais_menos_carregada_horas": balanceamento_final[
                "diferenca_mais_menos_carregada_horas"
            ],
            "jornada_excedida_total_horas": balanceamento_final[
                "jornada_excedida_total_horas"
            ],
            "equipes_baixa_alocacao": balanceamento_final[
                "equipes_baixa_alocacao"
            ],
            "percentual_equipes_baixa_alocacao": balanceamento_final[
                "percentual_equipes_baixa_alocacao"
            ],
            "reducao_distancia_vs_fifo_percentual": reducao_distancia,
            "reducao_tempo_deslocamento_vs_fifo_percentual": reducao_tempo,
            "funcao_objetivo": objetivo
        })

    objetivo_sem_balanceamento = calcular_funcao_objetivo(
        sem_balanceamento_fifo_dia["distancia_total_km"],
        balanceamento_fifo["jornada_excedida_total_horas"],
        balanceamento_fifo["tempo_ocioso_total_horas"],
        balanceamento_fifo[
            "diferenca_mais_menos_carregada_horas"
        ]
    )

    metricas_ablation.append({
        "dia": dia,
        "cenario": "Sem balanceamento + FIFO",
        "metodo_rota": "FIFO",
        "distancia_total_km": sem_balanceamento_fifo_dia[
            "distancia_total_km"
        ],
        "tempo_deslocamento_horas": sem_balanceamento_fifo_dia[
            "tempo_deslocamento_horas"
        ],
        "tempo_total_horas": sem_balanceamento_fifo_dia[
            "tempo_total_horas"
        ],
        "quantidade_servicos": sem_balanceamento_fifo_dia[
            "quantidade_servicos"
        ],
        "equipes_usadas": balanceamento_fifo["equipes_usadas"],
        "carga_media_horas": balanceamento_fifo["carga_media_horas"],
        "carga_desvio_padrao_horas": balanceamento_fifo[
            "carga_desvio_padrao_horas"
        ],
        "coeficiente_variacao_carga": balanceamento_fifo[
            "coeficiente_variacao_carga"
        ],
        "tempo_ocioso_total_horas": balanceamento_fifo[
            "tempo_ocioso_total_horas"
        ],
        "diferenca_mais_menos_carregada_horas": balanceamento_fifo[
            "diferenca_mais_menos_carregada_horas"
        ],
        "jornada_excedida_total_horas": balanceamento_fifo[
            "jornada_excedida_total_horas"
        ],
        "equipes_baixa_alocacao": balanceamento_fifo[
            "equipes_baixa_alocacao"
        ],
        "percentual_equipes_baixa_alocacao": balanceamento_fifo[
            "percentual_equipes_baixa_alocacao"
        ],
        "reducao_distancia_vs_fifo_percentual": 0,
        "reducao_tempo_deslocamento_vs_fifo_percentual": 0,
        "funcao_objetivo": objetivo_sem_balanceamento
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

df_baselines = pd.DataFrame(metricas_baselines)

df_balanceamento = pd.DataFrame(metricas_balanceamento)

# ==========================================================
# RESUMO ESTATISTICO DO BALANCEAMENTO
# ==========================================================

resumo_balanceamento = pd.DataFrame()

if len(df_balanceamento) > 0:

    resumo_balanceamento_completo = resumir_por_metodo(
        df_balanceamento,
        "cenario"
    )

    metricas_resumo_balanceamento = [
        "equipes_usadas",
        "carga_media_horas",
        "carga_desvio_padrao_horas",
        "coeficiente_variacao_carga",
        "tempo_ocioso_total_horas",
        "diferenca_mais_menos_carregada_horas"
    ]

    colunas_resumo_balanceamento = ["cenario", "dias"]

    for metrica in metricas_resumo_balanceamento:

        colunas_resumo_balanceamento += [
            f"{metrica}_media",
            f"{metrica}_desvio_padrao",
            f"{metrica}_ic95"
        ]

    resumo_balanceamento = resumo_balanceamento_completo[
        [
            coluna
            for coluna in colunas_resumo_balanceamento
            if coluna in resumo_balanceamento_completo.columns
        ]
    ]

df_ablation = pd.DataFrame(metricas_ablation)

# ==========================================================
# TESTES ESTATISTICOS - WILCOXON
# ==========================================================

registros_wilcoxon = []

if len(df_ablation) > 0:

    pivot_distancia = df_ablation[
        df_ablation["cenario"].isin([
            "Balanceamento + FIFO",
            "Balanceamento + ACO + 2OPT"
        ])
    ].pivot(
        index="dia",
        columns="cenario",
        values="distancia_total_km"
    ).dropna()

    if (
        "Balanceamento + FIFO" in pivot_distancia.columns
        and
        "Balanceamento + ACO + 2OPT" in pivot_distancia.columns
        and
        len(pivot_distancia) > 0
    ):

        try:

            estatistica, p_value = wilcoxon(
                pivot_distancia["Balanceamento + FIFO"],
                pivot_distancia["Balanceamento + ACO + 2OPT"]
            )

            registros_wilcoxon.append({
                "comparacao": "FIFO vs ACO + 2OPT (distancia_total_km)",
                "estatistica": estatistica,
                "p_value": p_value
            })

        except ValueError as erro:

            log_etapa(
                f"Wilcoxon FIFO vs ACO + 2OPT nao calculado: {erro}"
            )

if len(df_balanceamento) > 0:

    pivot_equipes = df_balanceamento.pivot(
        index="dia",
        columns="cenario",
        values="equipes_usadas"
    ).dropna()

    if (
        "Sem balanceamento" in pivot_equipes.columns
        and
        "Com balanceamento" in pivot_equipes.columns
        and
        len(pivot_equipes) > 0
    ):

        try:

            estatistica, p_value = wilcoxon(
                pivot_equipes["Sem balanceamento"],
                pivot_equipes["Com balanceamento"]
            )

            registros_wilcoxon.append({
                "comparacao": (
                    "Sem balanceamento vs Com balanceamento "
                    "(equipes_usadas)"
                ),
                "estatistica": estatistica,
                "p_value": p_value
            })

        except ValueError as erro:

            log_etapa(
                "Wilcoxon Sem balanceamento vs Com balanceamento "
                f"nao calculado: {erro}"
            )

teste_wilcoxon = pd.DataFrame(registros_wilcoxon)

if len(teste_wilcoxon) > 0:

    teste_wilcoxon["interpretacao"] = teste_wilcoxon["p_value"].apply(
        lambda p: (
            "Diferenca estatisticamente significativa (p < 0.05)"
            if p < 0.05
            else "Sem diferenca estatisticamente significativa (p >= 0.05)"
        )
    )

df_cargas_boxplot = pd.DataFrame(cargas_equipes_boxplot)

df_2opt = pd.DataFrame(comparativo_2opt)

df_cargas_boxplot_reduzido = pd.DataFrame()

if len(df_cargas_boxplot) > 0:

    limites_carga = (
        df_cargas_boxplot
        .groupby("cenario")["carga_horas"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(columns={
            0.05: "p05",
            0.95: "p95"
        })
    )

    df_cargas_boxplot_reduzido = df_cargas_boxplot.merge(
        limites_carga,
        left_on="cenario",
        right_index=True,
        how="left"
    )

    df_cargas_boxplot_reduzido["carga_horas_winsorizada"] = (
        df_cargas_boxplot_reduzido["carga_horas"]
        .clip(
            lower=df_cargas_boxplot_reduzido["p05"],
            upper=df_cargas_boxplot_reduzido["p95"],
            axis=0
        )
    )

if (
    len(df_ablation) > 0
    and
    "cenario" in df_ablation.columns
):

    resumo_baselines = resumir_por_metodo(
        df_ablation[
            df_ablation["cenario"].str.startswith("Balanceamento +")
        ],
        "metodo_rota"
    ).rename(
        columns={
            "metodo_rota": "metodo",
            "distancia_total_km_desvio_padrao": (
                "distancia_total_km_desvio"
            ),
            "tempo_deslocamento_horas_desvio_padrao": (
                "tempo_deslocamento_horas_desvio"
            )
        }
    )

    colunas_resumo_baselines = [
        "metodo",
        "distancia_total_km_media",
        "distancia_total_km_desvio",
        "distancia_total_km_ic95",
        "tempo_deslocamento_horas_media",
        "tempo_deslocamento_horas_desvio",
        "tempo_deslocamento_horas_ic95",
        "equipes_usadas_media"
    ]

    resumo_baselines = resumo_baselines[
        [
            coluna
            for coluna in colunas_resumo_baselines
            if coluna in resumo_baselines.columns
        ]
    ]

    df_resumo_baselines = resumo_baselines

    df_resumo_ablation = resumir_por_metodo(
        df_ablation,
        "cenario"
    )

else:

    resumo_baselines = pd.DataFrame()

    df_resumo_baselines = resumo_baselines

    df_resumo_ablation = pd.DataFrame()

# ==========================================================
# COMPARATIVO DE GANHOS (FIFO COMO BASELINE)
# ==========================================================

comparativo_ganhos = pd.DataFrame()

if (
    len(df_resumo_baselines) > 0
    and
    "metodo" in df_resumo_baselines.columns
):

    referencia_fifo_baseline = df_resumo_baselines.loc[
        df_resumo_baselines["metodo"] == "FIFO"
    ]

    if len(referencia_fifo_baseline) > 0:

        distancia_fifo = referencia_fifo_baseline[
            "distancia_total_km_media"
        ].iloc[0]

        tempo_fifo = referencia_fifo_baseline[
            "tempo_deslocamento_horas_media"
        ].iloc[0]

        registros_comparativo_ganhos = []

        for _, linha in df_resumo_baselines.iterrows():

            distancia_metodo = linha["distancia_total_km_media"]

            tempo_metodo = linha["tempo_deslocamento_horas_media"]

            ganho_distancia_percentual = (
                (distancia_fifo - distancia_metodo)
                / distancia_fifo
                * 100
                if distancia_fifo > 0
                else 0
            )

            ganho_tempo_deslocamento_percentual = (
                (tempo_fifo - tempo_metodo)
                / tempo_fifo
                * 100
                if tempo_fifo > 0
                else 0
            )

            registros_comparativo_ganhos.append({
                "metodo": linha["metodo"],
                "ganho_distancia_percentual": ganho_distancia_percentual,
                "ganho_tempo_deslocamento_percentual": (
                    ganho_tempo_deslocamento_percentual
                )
            })

        comparativo_ganhos = pd.DataFrame(registros_comparativo_ganhos)

# ==========================================================
# TABELA DE ABLATION STUDY
# ==========================================================

ablation_study = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenarios_ablation_study = [
        "Sem balanceamento + FIFO",
        "Balanceamento + Nearest Neighbor",
        "Balanceamento + ACO sem 2OPT",
        "Balanceamento + ACO + 2OPT"
    ]

    ablation_study = df_resumo_ablation[
        df_resumo_ablation["cenario"].isin(cenarios_ablation_study)
    ][
        [
            "cenario",
            "funcao_objetivo_media",
            "distancia_total_km_media",
            "tempo_deslocamento_horas_media",
            "equipes_usadas_media"
        ]
    ].copy()

    ablation_study["cenario"] = pd.Categorical(
        ablation_study["cenario"],
        categories=cenarios_ablation_study,
        ordered=True
    )

    ablation_study = ablation_study.sort_values(
        "cenario"
    ).reset_index(drop=True)

    referencia_ablation_study = ablation_study.loc[
        ablation_study["cenario"] == "Sem balanceamento + FIFO"
    ]

    if len(referencia_ablation_study) > 0:

        objetivo_baseline_ablation_study = referencia_ablation_study[
            "funcao_objetivo_media"
        ].iloc[0]

        ablation_study["ganho_percentual_funcao_objetivo"] = (
            (
                objetivo_baseline_ablation_study
                -
                ablation_study["funcao_objetivo_media"]
            )
            /
            objetivo_baseline_ablation_study
            *
            100
            if objetivo_baseline_ablation_study > 0
            else 0
        )

# ==========================================================
# EXPERIMENTO: EFEITO ISOLADO DO BALANCEAMENTO (FIFO)
# ==========================================================

ganho_balanceamento = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_sem_balanceamento = "Sem balanceamento + FIFO"

    cenario_com_balanceamento = "Balanceamento + FIFO"

    referencia_sem = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_sem_balanceamento
    ]

    referencia_com = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_com_balanceamento
    ]

    if (
        len(referencia_sem) > 0
        and
        len(referencia_com) > 0
    ):

        metricas_ganho_balanceamento = {
            "distancia_total_km": "distancia_total_km_media",
            "equipes_usadas": "equipes_usadas_media",
            "carga_media_horas": "carga_media_horas_media",
            "carga_desvio_padrao_horas": "carga_desvio_padrao_horas_media",
            "coeficiente_variacao_carga": "coeficiente_variacao_carga_media",
            "tempo_ocioso_total_horas": "tempo_ocioso_total_horas_media"
        }

        registros_ganho_balanceamento = []

        for metrica, coluna_media in metricas_ganho_balanceamento.items():

            valor_sem = referencia_sem[coluna_media].iloc[0]

            valor_com = referencia_com[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_sem - valor_com)
                / valor_sem
                * 100
                if valor_sem > 0
                else 0
            )

            registros_ganho_balanceamento.append({
                "metrica": metrica,
                "sem_balanceamento_fifo": valor_sem,
                "com_balanceamento_fifo": valor_com,
                "ganho_percentual": ganho_percentual
            })

        ganho_balanceamento = pd.DataFrame(
            registros_ganho_balanceamento
        )

# ==========================================================
# EXPERIMENTO: GANHO ESPECIFICO DO ACO
# ==========================================================

ganho_aco = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_nearest_neighbor = "Balanceamento + Nearest Neighbor"

    cenario_aco_sem_2opt = "Balanceamento + ACO sem 2OPT"

    referencia_nn = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_nearest_neighbor
    ]

    referencia_aco = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_sem_2opt
    ]

    if (
        len(referencia_nn) > 0
        and
        len(referencia_aco) > 0
    ):

        metricas_ganho_aco = {
            "distancia_total_km": "distancia_total_km_media",
            "tempo_deslocamento_horas": "tempo_deslocamento_horas_media",
            "funcao_objetivo": "funcao_objetivo_media"
        }

        registros_ganho_aco = []

        for metrica, coluna_media in metricas_ganho_aco.items():

            valor_nn = referencia_nn[coluna_media].iloc[0]

            valor_aco = referencia_aco[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_nn - valor_aco)
                / valor_nn
                * 100
                if valor_nn > 0
                else 0
            )

            registros_ganho_aco.append({
                "metrica": metrica,
                "balanceamento_nearest_neighbor": valor_nn,
                "balanceamento_aco_sem_2opt": valor_aco,
                "ganho_percentual": ganho_percentual
            })

        ganho_aco = pd.DataFrame(registros_ganho_aco)

# ==========================================================
# EXPERIMENTO: GANHO ESPECIFICO DO 2OPT
# ==========================================================

ganho_2opt_experimento = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_aco_sem_2opt = "Balanceamento + ACO sem 2OPT"

    cenario_aco_2opt = "Balanceamento + ACO + 2OPT"

    referencia_aco_sem_2opt = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_sem_2opt
    ]

    referencia_aco_2opt = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_2opt
    ]

    if (
        len(referencia_aco_sem_2opt) > 0
        and
        len(referencia_aco_2opt) > 0
    ):

        metricas_ganho_2opt_experimento = {
            "distancia_total_km": "distancia_total_km_media",
            "tempo_deslocamento_horas": "tempo_deslocamento_horas_media",
            "funcao_objetivo": "funcao_objetivo_media"
        }

        registros_ganho_2opt_experimento = []

        for metrica, coluna_media in metricas_ganho_2opt_experimento.items():

            valor_aco_sem_2opt = referencia_aco_sem_2opt[
                coluna_media
            ].iloc[0]

            valor_aco_2opt = referencia_aco_2opt[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_aco_sem_2opt - valor_aco_2opt)
                / valor_aco_sem_2opt
                * 100
                if valor_aco_sem_2opt > 0
                else 0
            )

            registros_ganho_2opt_experimento.append({
                "metrica": metrica,
                "balanceamento_aco_sem_2opt": valor_aco_sem_2opt,
                "balanceamento_aco_2opt": valor_aco_2opt,
                "ganho_percentual": ganho_percentual
            })

        ganho_2opt_experimento = pd.DataFrame(
            registros_ganho_2opt_experimento
        )

# ==========================================================
# TABELA FINAL CONSOLIDADA PARA PUBLICACAO
# ==========================================================

tabela_artigo = pd.DataFrame()

if (
    len(df_resumo_baselines) > 0
    and
    len(comparativo_ganhos) > 0
    and
    "metodo" in df_resumo_baselines.columns
    and
    "metodo" in comparativo_ganhos.columns
):

    ordem_metodos_artigo = [
        "FIFO",
        "Nearest Neighbor",
        "ACO sem 2OPT",
        "ACO + 2OPT"
    ]

    tabela_artigo = df_resumo_baselines.merge(
        comparativo_ganhos,
        on="metodo",
        how="inner"
    )

    tabela_artigo = tabela_artigo[
        tabela_artigo["metodo"].isin(ordem_metodos_artigo)
    ]

    tabela_artigo["metodo"] = pd.Categorical(
        tabela_artigo["metodo"],
        categories=ordem_metodos_artigo,
        ordered=True
    )

    tabela_artigo = tabela_artigo.sort_values(
        "metodo"
    ).reset_index(drop=True)

    tabela_artigo = tabela_artigo.rename(columns={
        "metodo": "Método",
        "distancia_total_km_media": "Distância Média (km)",
        "distancia_total_km_ic95": "IC95 Distância",
        "tempo_deslocamento_horas_media": "Tempo Médio Deslocamento (h)",
        "tempo_deslocamento_horas_ic95": "IC95 Tempo",
        "ganho_distancia_percentual": "Ganho Distância (%)",
        "ganho_tempo_deslocamento_percentual": "Ganho Tempo (%)"
    })

    tabela_artigo = tabela_artigo[
        [
            "Método",
            "Distância Média (km)",
            "IC95 Distância",
            "Tempo Médio Deslocamento (h)",
            "IC95 Tempo",
            "Ganho Distância (%)",
            "Ganho Tempo (%)"
        ]
    ]

    colunas_numericas_artigo = [
        coluna
        for coluna in tabela_artigo.columns
        if coluna != "Método"
    ]

    tabela_artigo[colunas_numericas_artigo] = tabela_artigo[
        colunas_numericas_artigo
    ].round(2)

if len(df_resumo_ablation) > 0:

    referencia_ablation = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == "Sem balanceamento + FIFO"
    ]

    if len(referencia_ablation) > 0:

        distancia_base = referencia_ablation[
            "distancia_total_km_media"
        ].iloc[0]

        tempo_base = referencia_ablation[
            "tempo_deslocamento_horas_media"
        ].iloc[0]

        equipes_base = referencia_ablation[
            "equipes_usadas_media"
        ].iloc[0]

        carga_base = referencia_ablation[
            "carga_media_horas_media"
        ].iloc[0]

        df_resumo_ablation[
            "ganho_distancia_percentual"
        ] = (
            (distancia_base - df_resumo_ablation[
                "distancia_total_km_media"
            ])
            /
            distancia_base
            *
            100
            if distancia_base > 0
            else 0
        )

        df_resumo_ablation[
            "ganho_tempo_deslocamento_percentual"
        ] = (
            (tempo_base - df_resumo_ablation[
                "tempo_deslocamento_horas_media"
            ])
            /
            tempo_base
            *
            100
            if tempo_base > 0
            else 0
        )

        df_resumo_ablation[
            "reducao_uso_equipes_percentual"
        ] = (
            (equipes_base - df_resumo_ablation[
                "equipes_usadas_media"
            ])
            /
            equipes_base
            *
            100
            if equipes_base > 0
            else 0
        )

        df_resumo_ablation[
            "ganho_carga_media_percentual"
        ] = (
            (df_resumo_ablation[
                "carga_media_horas_media"
            ] - carga_base)
            /
            carga_base
            *
            100
            if carga_base > 0
            else 0
        )

df_ganhos_ablation = pd.DataFrame()

if len(df_ablation) > 0:

    cenarios_ganho = {
        "ganho_balanceamento": (
            "Sem balanceamento + FIFO",
            "Balanceamento + FIFO"
        ),
        "ganho_aco": (
            "Balanceamento + FIFO",
            "Balanceamento + ACO sem 2OPT"
        ),
        "ganho_2opt": (
            "Balanceamento + ACO sem 2OPT",
            "Balanceamento + ACO + 2OPT"
        )
    }

    registros_ganhos = []

    for dia_ganho, grupo_dia in df_ablation.groupby("dia"):

        grupo_idx = grupo_dia.set_index("cenario")

        for componente, (antes, depois) in cenarios_ganho.items():

            if (
                antes not in grupo_idx.index
                or
                depois not in grupo_idx.index
            ):
                continue

            linha_antes = grupo_idx.loc[antes]
            linha_depois = grupo_idx.loc[depois]

            registros_ganhos.append({
                "dia": dia_ganho,
                "componente": componente,
                "cenario_antes": antes,
                "cenario_depois": depois,
                "ganho_distancia_percentual": (
                    (
                        linha_antes["distancia_total_km"]
                        -
                        linha_depois["distancia_total_km"]
                    )
                    /
                    linha_antes["distancia_total_km"]
                    *
                    100
                    if linha_antes["distancia_total_km"] > 0
                    else 0
                ),
                "ganho_tempo_deslocamento_percentual": (
                    (
                        linha_antes["tempo_deslocamento_horas"]
                        -
                        linha_depois["tempo_deslocamento_horas"]
                    )
                    /
                    linha_antes["tempo_deslocamento_horas"]
                    *
                    100
                    if linha_antes["tempo_deslocamento_horas"] > 0
                    else 0
                ),
                "ganho_funcao_objetivo_percentual": (
                    (
                        linha_antes["funcao_objetivo"]
                        -
                        linha_depois["funcao_objetivo"]
                    )
                    /
                    linha_antes["funcao_objetivo"]
                    *
                    100
                    if linha_antes["funcao_objetivo"] > 0
                    else 0
                ),
                "variacao_equipes_percentual": (
                    (
                        linha_antes["equipes_usadas"]
                        -
                        linha_depois["equipes_usadas"]
                    )
                    /
                    linha_antes["equipes_usadas"]
                    *
                    100
                    if linha_antes["equipes_usadas"] > 0
                    else 0
                ),
                "variacao_carga_media_percentual": (
                    (
                        linha_depois["carga_media_horas"]
                        -
                        linha_antes["carga_media_horas"]
                    )
                    /
                    linha_antes["carga_media_horas"]
                    *
                    100
                    if linha_antes["carga_media_horas"] > 0
                    else 0
                )
            })

    df_ganhos_ablation = pd.DataFrame(registros_ganhos)

df_resumo_ganhos_ablation = resumir_por_metodo(
    df_ganhos_ablation.rename(
        columns={
            "ganho_distancia_percentual": "distancia_total_km",
            "ganho_tempo_deslocamento_percentual": (
                "tempo_deslocamento_horas"
            ),
            "ganho_funcao_objetivo_percentual": "funcao_objetivo",
            "variacao_equipes_percentual": "equipes_usadas",
            "variacao_carga_media_percentual": "carga_media_horas"
        }
    ),
    "componente"
) if len(df_ganhos_ablation) > 0 else pd.DataFrame()

if len(resumo_baselines) > 0:

    print("\nRESUMO BASELINES:")

    print(
        resumo_baselines
        .sort_values("distancia_total_km_media")
        .to_string(index=False)
    )

df_metricas.to_excel(
    "metricas_operacionais.xlsx",
    index=False
)

pd.DataFrame(sequencias).to_excel(
    "sequencia_atendimentos.xlsx",
    index=False
)

df_baselines.to_excel(
    "metricas_baselines_rotas.xlsx",
    index=False
)

df_balanceamento.to_excel(
    "metricas_balanceamento.xlsx",
    index=False
)

df_ablation.to_excel(
    "ablation_study_diario.xlsx",
    index=False
)

resumo_baselines.to_excel(
    "resumo_baselines.xlsx",
    index=False
)

df_resumo_baselines.to_excel(
    "resumo_baselines_ic95.xlsx",
    index=False
)

df_resumo_ablation.to_excel(
    "resumo_ablation_ic95.xlsx",
    index=False
)

df_ganhos_ablation.to_excel(
    "ganhos_incrementais_ablation.xlsx",
    index=False
)

df_resumo_ganhos_ablation.to_excel(
    "resumo_ganhos_incrementais_ic95.xlsx",
    index=False
)

df_2opt.to_excel(
    "comparativo_antes_depois_2opt.xlsx",
    index=False
)

df_cargas_boxplot.to_excel(
    "cargas_equipes_boxplot.xlsx",
    index=False
)

df_cargas_boxplot_reduzido.to_excel(
    "cargas_equipes_boxplot_reduzido_outliers.xlsx",
    index=False
)

os.makedirs("resultados", exist_ok=True)

comparativo_ganhos.to_excel(
    "resultados/ganhos_percentuais.xlsx",
    index=False
)

resumo_balanceamento.to_excel(
    "resultados/resumo_balanceamento.xlsx",
    index=False
)

teste_wilcoxon.to_excel(
    "resultados/testes_estatisticos.xlsx",
    index=False
)

ablation_study.to_excel(
    "resultados/ablation_study.xlsx",
    index=False
)

ganho_balanceamento.to_excel(
    "resultados/ganho_balanceamento.xlsx",
    index=False
)

ganho_aco.to_excel(
    "resultados/ganho_aco.xlsx",
    index=False
)

ganho_2opt_experimento.to_excel(
    "resultados/ganho_2opt.xlsx",
    index=False
)

tabela_artigo.to_excel(
    "resultados/tabela_artigo.xlsx",
    index=False
)

pd.DataFrame([
    {
        "termo": "distancia_total_km",
        "peso": PESO_DISTANCIA,
        "penalidade": "distancia percorrida"
    },
    {
        "termo": "jornada_excedida_horas",
        "peso": PESO_JORNADA_EXCEDIDA,
        "penalidade": "horas acima da jornada maxima"
    },
    {
        "termo": "tempo_ocioso_total_horas",
        "peso": PESO_OCIOSIDADE,
        "penalidade": "ociosidade agregada das equipes"
    },
    {
        "termo": "diferenca_mais_menos_carregada_horas",
        "peso": PESO_DESBALANCEAMENTO,
        "penalidade": "desbalanceamento entre equipes"
    }
]).to_excel(
    "funcao_objetivo_formalizada.xlsx",
    index=False
)

if plt is not None:

    if (
        len(df_resumo_baselines) > 0
        and
        "metodo" in df_resumo_baselines.columns
    ):

        os.makedirs("graficos", exist_ok=True)

        ordem_metodos_baselines = [
            "FIFO",
            "Nearest Neighbor",
            "ACO sem 2OPT",
            "ACO + 2OPT"
        ]

        df_resumo_baselines_plot = df_resumo_baselines[
            df_resumo_baselines["metodo"].isin(ordem_metodos_baselines)
        ].copy()

        df_resumo_baselines_plot["metodo"] = pd.Categorical(
            df_resumo_baselines_plot["metodo"],
            categories=ordem_metodos_baselines,
            ordered=True
        )

        df_resumo_baselines_plot = df_resumo_baselines_plot.sort_values(
            "metodo"
        )

        plt.figure(figsize=(8, 4.5))
        plt.bar(
            df_resumo_baselines_plot["metodo"].astype(str),
            df_resumo_baselines_plot["distancia_total_km_media"],
            yerr=df_resumo_baselines_plot["distancia_total_km_ic95"],
            capsize=5,
            color="#457b9d"
        )
        plt.ylabel("Distancia total media (km)")
        plt.title("Comparativo de baselines de rota - IC 95%")
        plt.tight_layout()
        plt.savefig(
            "graficos/comparativo_baselines_ic95.png",
            dpi=160
        )
        plt.close()

    if len(df_2opt) > 0:

        df_2opt_plot = df_2opt[
            ["antes_2opt_km", "depois_2opt_km"]
        ].mean()

        plt.figure(figsize=(7, 4))
        df_2opt_plot.plot(kind="bar", color=["#8a8f98", "#2a9d8f"])
        plt.ylabel("Distancia media por equipe (km)")
        plt.title("Antes e depois do 2OPT")
        plt.tight_layout()
        plt.savefig(
            "grafico_antes_depois_2opt.png",
            dpi=160
        )
        plt.close()

    if len(df_cargas_boxplot) > 0:

        plt.figure(figsize=(8, 4))
        df_cargas_boxplot.boxplot(
            column="carga_horas",
            by="cenario",
            grid=False
        )
        plt.suptitle("")
        plt.title("Carga por equipe")
        plt.ylabel("Horas")
        plt.tight_layout()
        plt.savefig(
            "boxplot_carga_por_equipe.png",
            dpi=160
        )
        plt.close()

        if len(df_cargas_boxplot_reduzido) > 0:

            ordem_cenarios = [
                "Com balanceamento",
                "Sem balanceamento"
            ]

            dados_boxplot = [
                df_cargas_boxplot_reduzido.loc[
                    df_cargas_boxplot_reduzido["cenario"] == cenario,
                    "carga_horas_winsorizada"
                ]
                for cenario in ordem_cenarios
                if cenario in df_cargas_boxplot_reduzido["cenario"].unique()
            ]

            labels_boxplot = [
                cenario
                for cenario in ordem_cenarios
                if cenario in df_cargas_boxplot_reduzido["cenario"].unique()
            ]

            medias_reais = [
                df_cargas_boxplot.loc[
                    df_cargas_boxplot["cenario"] == cenario,
                    "carga_horas"
                ].mean()
                for cenario in labels_boxplot
            ]

            plt.figure(figsize=(8, 4.8))
            plt.boxplot(
                dados_boxplot,
                labels=labels_boxplot,
                showfliers=False,
                whis=(5, 95),
                patch_artist=True,
                boxprops={
                    "facecolor": "#9ecae1",
                    "color": "#2b6c8a"
                },
                medianprops={
                    "color": "#0b3d4f",
                    "linewidth": 2
                },
                whiskerprops={
                    "color": "#2b6c8a"
                },
                capprops={
                    "color": "#2b6c8a"
                }
            )
            plt.scatter(
                range(1, len(labels_boxplot) + 1),
                medias_reais,
                color="#e76f51",
                label="Media real",
                zorder=3
            )
            plt.ylabel("Horas")
            plt.title(
                "Carga por equipe - outliers suavizados por percentis 5%-95%"
            )
            plt.ylim(7.85, 8.02)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                "boxplot_carga_por_equipe_outliers_reduzidos.png",
                dpi=160
            )
            plt.close()

            faixas_carga = [
                0,
                2,
                4,
                6,
                7,
                7.5,
                7.75,
                7.9,
                7.95,
                8.01
            ]

            labels_faixas = [
                "0-2",
                "2-4",
                "4-6",
                "6-7",
                "7-7.5",
                "7.5-7.75",
                "7.75-7.9",
                "7.9-7.95",
                "7.95-8"
            ]

            df_histograma_carga = df_cargas_boxplot.copy()

            df_histograma_carga["faixa_carga"] = pd.cut(
                df_histograma_carga["carga_horas"],
                bins=faixas_carga,
                labels=labels_faixas,
                include_lowest=True,
                right=False
            )

            tabela_histograma_carga = pd.crosstab(
                df_histograma_carga["faixa_carga"],
                df_histograma_carga["cenario"]
            ).reindex(labels_faixas).fillna(0)

            ax = tabela_histograma_carga.plot(
                kind="bar",
                figsize=(10, 5),
                color=["#2a9d8f", "#8a8f98"]
            )

            ax.set_title("Distribuicao da carga por equipe")
            ax.set_xlabel("Faixa de carga (horas)")
            ax.set_ylabel("Quantidade de equipes")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(
                "histograma_carga_por_equipe.png",
                dpi=160
            )
            plt.close()

    if len(df_resumo_ablation) > 0:

        df_plot = df_resumo_ablation.sort_values(
            "funcao_objetivo_media"
        )

        plt.figure(figsize=(10, 4))
        plt.bar(
            df_plot["cenario"],
            df_plot["funcao_objetivo_media"],
            color="#457b9d"
        )
        plt.ylabel("Funcao objetivo media")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            "grafico_ablation_funcao_objetivo.png",
            dpi=160
        )
        plt.close()
else:

    log_etapa(
        "matplotlib nao encontrado; graficos PNG nao foram gerados"
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

print("\nPROCESSAMENTO FINALIZADO")
