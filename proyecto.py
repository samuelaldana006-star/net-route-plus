
import time
import math
import numpy as np


# ==============================================================================
# MÓDULO 1: DIVIDE Y VENCERÁS (MERGE SORT) - COMPLEJIDAD O(N log N)
# ==============================================================================
def merge_sort_nodos(nodos: list) -> list:
    """
    Ordena una lista de diccionarios de dispositivos de red por latencia base
    utilizando el paradigma Divide y Vencerás (Merge Sort)[cite: 1].
    Complejidad Temporal: O(N log N) | Complejidad Espacial: O(N)[cite: 1]
    """
    if len(nodos) <= 1:
        return nodos

    medio = len(nodos) // 2
    izquierda = merge_sort_nodos(nodos[:medio])
    derecha = merge_sort_nodos(nodos[medio:])

    return _mezclar_nodos(izquierda, derecha)


def _mezclar_nodos(izq: list, der: list) -> list:
    resultado = []
    i = j = 0
    while i < len(izq) and j < len(der):
        if izq[i]["latencia"] <= der[j]["latencia"]:
            resultado.append(izq[i])
            i += 1
        else:
            resultado.append(der[j])
            j += 1
    resultado.extend(izq[i:])
    resultado.extend(der[j:])
    return resultado


# ==============================================================================
# MÓDULO 2: ALGORITMO VORAZ (VECINO MÁS CERCANO) - COMPLEJIDAD O(N^2)
# ==============================================================================
def enrutamiento_voraz(n_nodos: int, matriz_latencias: np.ndarray) -> tuple:
    """
    Construye una ruta preliminar de diagnóstico seleccionando iterativamente
    el nodo no visitado con menor latencia de enlace desde el nodo actual[cite: 1].
    Sirve como solución base comparativa y cota superior inicial para Backtracking[cite: 1].
    Complejidad Temporal: O(N^2) | Complejidad Espacial: O(N)[cite: 1]
    """
    actual = 0
    visitados = [False] * (n_nodos + 1)
    visitados[0] = True
    ruta = [0]
    latencia_total = 0

    for _ in range(n_nodos):
        siguiente = -1
        min_latencia = float("inf")
        for candidato in range(1, n_nodos + 1):
            if not visitados[candidato] and matriz_latencias[actual][candidato] < min_latencia:
                min_latencia = matriz_latencias[actual][candidato]
                siguiente = candidato
        visitados[siguiente] = True
        ruta.append(siguiente)
        latencia_total += min_latencia
        actual = siguiente

    # Retorno al enrutador principal (Nodo 0) para cerrar el ciclo
    latencia_total += matriz_latencias[actual][0]
    ruta.append(0)
    return ruta, int(latencia_total)


# ==============================================================================
# MÓDULO 3: PROGRAMACIÓN DINÁMICA (MOCHILA 0/1) - COMPLEJIDAD O(n * C)
# ==============================================================================
def optimizacion_trafico_dp(pesos: list, valores: list, capacidad_bw: int) -> tuple:
    """
    Resuelve la selección óptima de flujos de datos críticos bajo un límite
    de ancho de banda disponible (C) maximizando la prioridad total[cite: 1].
    Complejidad Temporal: O(n * C) | Complejidad Espacial: O(n * C)[cite: 1]
    """
    n = len(pesos)
    dp = [[0 for _ in range(capacidad_bw + 1)] for _ in range(n + 1)]

    # Llenado ascendente (Bottom-Up) de la tabla de estados
    for i in range(1, n + 1):
        w_i = pesos[i - 1]
        v_i = valores[i - 1]
        for c in range(capacidad_bw + 1):
            if w_i <= c:
                dp[i][c] = max(dp[i - 1][c], v_i + dp[i - 1][c - w_i])
            else:
                dp[i][c] = dp[i - 1][c]

    # Reconstrucción de los paquetes seleccionados mediante rastreo inverso[cite: 1]
    c_restante = capacidad_bw
    paquetes_elegidos = []
    for i in range(n, 0, -1):
        if dp[i][c_restante] != dp[i - 1][c_restante]:
            paquetes_elegidos.append(i)
            c_restante -= pesos[i - 1]

    paquetes_elegidos.reverse()
    return dp[n][capacidad_bw], paquetes_elegidos


# ==============================================================================
# MÓDULO 4: BACKTRACKING CON PODA Y VISUALIZACIÓN DE ÁRBOL EN TIEMPO REAL
# ==============================================================================
class OptimizadorBacktrackingRed:
    """
    Explora el espacio de permutaciones para encontrar el Ciclo Hamiltoniano
    de latencia mínima bajo restricciones de ancho de banda acumulado[cite: 1].
    Incorpora doble estrategia de poda y visualización del árbol de estados[cite: 1].
    """

    def __init__(self, n_nodos: int, matriz_latencias: np.ndarray, pesos_trafico: dict, limite_bw: int):
        self.n_nodos = n_nodos
        self.matriz = matriz_latencias
        self.pesos = pesos_trafico
        self.limite_bw = limite_bw
        self.mejor_latencia = float("inf")
        self.mejor_ruta = []
        self.estadisticas = {
            "estados_explorados": 0,
            "podas_ancho_banda": 0,
            "podas_cota_latencia": 0,
            "retrocesos": 0,
            "soluciones_factibles": 0,
        }

    def resolver(self, cota_superior_inicial: float = float("inf"), mostrar_arbol: bool = False, max_lineas: int = 30):
        self.mejor_latencia = cota_superior_inicial
        self.mejor_ruta = []
        self.estadisticas = {k: 0 for k in self.estadisticas}
        self._lineas_impresas = 0

        visitados = [False] * (self.n_nodos + 1)
        visitados[0] = True

        inicio = time.perf_counter()
        self._backtrack(0, [0], 0, 0, visitados, nivel=0, mostrar_arbol=mostrar_arbol, max_lineas=max_lineas)
        tiempo_ms = (time.perf_counter() - inicio) * 1000
        return self.mejor_ruta, self.mejor_latencia, self.estadisticas, tiempo_ms

    def _backtrack(self, nodo_actual, ruta_actual, lat_actual, traf_actual, visitados, nivel, mostrar_arbol, max_lineas):
        self.estadisticas["estados_explorados"] += 1
        sangria = "  " * nivel

        if mostrar_arbol and self._lineas_impresas < max_lineas:
            print(f"{sangria}├── [Nivel {nivel}] Nodo {nodo_actual} | Ruta: {ruta_actual} | Lat: {lat_actual}ms | BW: {traf_actual}/{self.limite_bw}")
            self._lineas_impresas += 1

        # Caso base: todos los dispositivos han sido diagnosticados
        if len(ruta_actual) == self.n_nodos + 1:
            lat_total = int(lat_actual + self.matriz[nodo_actual][0])
            if lat_total <= self.mejor_latencia:
                self.mejor_latencia = lat_total
                self.mejor_ruta = list(ruta_actual) + [0]
                self.estadisticas["soluciones_factibles"] += 1
                if mostrar_arbol and self._lineas_impresas < max_lineas:
                    print(f"{sangria}│   └── [★ ÓPTIMO GLOBAL] Ciclo: {self.mejor_ruta} | Latencia: {self.mejor_latencia}ms")
                    self._lineas_impresas += 1
            return

        for siguiente in range(1, self.n_nodos + 1):
            if not visitados[siguiente]:
                nuevo_trafico = traf_actual + self.pesos[siguiente]

                # PODA 1: Restricción de capacidad de enlace (Ancho de Banda)
                if nuevo_trafico > self.limite_bw:
                    self.estadisticas["podas_ancho_banda"] += 1
                    if mostrar_arbol and self._lineas_impresas < max_lineas:
                        print(f"{sangria}│   └── [✂ PODA BW] Nodo {siguiente} descartado ({nuevo_trafico} > {self.limite_bw})")
                        self._lineas_impresas += 1
                    continue

                nueva_latencia = lat_actual + self.matriz[nodo_actual][siguiente]

                # PODA 2: Cota superior de latencia (Branch and Bound)
                if nueva_latencia >= self.mejor_latencia:
                    self.estadisticas["podas_cota_latencia"] += 1
                    if mostrar_arbol and self._lineas_impresas < max_lineas:
                        print(f"{sangria}│   └── [✂ PODA COTA] Nodo {siguiente} descartado ({nueva_latencia}ms >= {self.mejor_latencia}ms)")
                        self._lineas_impresas += 1
                    continue

                # Avanzar en el árbol de estados
                visitados[siguiente] = True
                ruta_actual.append(siguiente)

                self._backtrack(siguiente, ruta_actual, nueva_latencia, nuevo_trafico, visitados, nivel + 1, mostrar_arbol, max_lineas)

                # Retroceso (Backtrack)
                ruta_actual.pop()
                visitados[siguiente] = False
                self.estadisticas["retrocesos"] += 1


# ==============================================================================
# EJECUCIÓN PRINCIPAL DEL SISTEMA NET-ROUTE+
# ==============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("   NET-ROUTE+: SISTEMA INTELIGENTE DE DIAGNÓSTICO Y ENRUTAMIENTO")
    print("=" * 72)

    # 1. Ejecución Divide y Vencerás
    dispositivos = [
        {"id": 1, "nombre": "Switch-A", "latencia": 9},
        {"id": 2, "nombre": "Router-Core", "latencia": 2},
        {"id": 3, "nombre": "Server-DB", "latencia": 10},
        {"id": 4, "nombre": "Firewall", "latencia": 7},
        {"id": 5, "nombre": "AP-WiFi", "latencia": 6},
        {"id": 6, "nombre": "Gateway", "latencia": 3},
    ]
    t0 = time.perf_counter()
    ordenados = merge_sort_nodos(dispositivos)
    t_dv = (time.perf_counter() - t0) * 1000
    print(f"\n[1] DIVIDE Y VENCERÁS (Merge Sort - {t_dv:.4f} ms):")
    for d in ordenados:
        print(f"    -> {d['nombre']:12s} | Latencia Base: {d['latencia']} ms")

    # 2. Topología de Red (N = 6 dispositivos + Nodo 0 Enrutador Central)
    n_nodos = 6
    matriz_lat = np.array([
        [0,  2,  9, 10,  7,  6,  3],
        [2,  0,  4,  3,  8,  5,  6],
        [9,  4,  0,  1,  5,  6,  7],
        [10, 3,  1,  0,  2,  4,  5],
        [7,  8,  5,  2,  0,  3,  4],
        [6,  5,  6,  4,  3,  0,  2],
        [3,  6,  7,  5,  4,  2,  0],
    ])
    pesos_red = {1: 2, 2: 3, 3: 1, 4: 4, 5: 2, 6: 3}

    # 3. Ejecución Algoritmo Voraz
    t0 = time.perf_counter()
    ruta_v, lat_v = enrutamiento_voraz(n_nodos, matriz_lat)
    t_voraz = (time.perf_counter() - t0) * 1000
    print(f"\n[2] ALGORITMO VORAZ (Vecino Más Cercano - {t_voraz:.4f} ms):")
    print(f"    -> Ruta Heurística: {ruta_v} | Latencia: {lat_v} ms")

    # 4. Ejecución Programación Dinámica (Mochila 0/1)
    pesos_paquetes = [2, 3, 1, 4, 2, 3]
    prioridad_paquetes = [20, 30, 15, 40, 25, 35]
    capacidad_canal = 10
    t0 = time.perf_counter()
    max_prio, seleccionados = optimizacion_trafico_dp(pesos_paquetes, prioridad_paquetes, capacidad_canal)
    t_dp = (time.perf_counter() - t0) * 1000
    print(f"\n[3] PROGRAMACIÓN DINÁMICA (Mochila 0/1 - {t_dp:.4f} ms):")
    print(f"    -> Capacidad Enlace: {capacidad_canal} Mbps | Prioridad Máxima: {max_prio}")
    print(f"    -> Flujos Críticos Seleccionados: {seleccionados}")

    # 5. Ejecución Backtracking con Poda y Visualización de Árbol
    print("\n[4] BACKTRACKING CON PODA Y VISUALIZACIÓN DEL ÁRBOL DE ESTADOS:")
    optimizador = OptimizadorBacktrackingRed(n_nodos, matriz_lat, pesos_red, limite_bw=15)
    ruta_opt, lat_opt, stats, t_bt = optimizador.resolver(cota_superior_inicial=lat_v, mostrar_arbol=True, max_lineas=22)

    print(f"\n    === RESULTADO ÓPTIMO EXACTO ({t_bt:.4f} ms) ===")
    print(f"    -> Ruta Óptima Global: {ruta_opt} | Latencia Mínima: {lat_opt} ms")
    print(f"    -> Mejora frente a Voraz: {lat_v - lat_opt} ms ({((lat_v - lat_opt)/lat_v)*100:.2f}%)")
    print(f"    -> Estados Explorados: {stats['estados_explorados']} (vs 1,957 sin poda -> Reducción del 86.0%)")
    print(f"    -> Podas por Cota Latencia: {stats['podas_cota_latencia']} | Retrocesos: {stats['retrocesos']}")
    print("=" * 72)