import pandas as pd
from typing import Any
import os
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import folium

class Graph:

    def __init__(self):
        self.vertices = {}  
        self.pesos = {}  # Pesos
        self.info_aeropuertos = {}  # Información de los aeropuertos
        self.directed = False

    def agregar_vertice(self, airport, info=None):
        if airport not in self.vertices:
            self.vertices[airport] = []
            self.pesos[airport] = []
            if info:
                self.info_aeropuertos[airport] = info  # Guardar la información del aeropuerto

    def agregar_arista(self, origen, destino, peso):
        if origen not in self.vertices:
            return
        if destino not in self.vertices:
            return
        self.vertices[origen].append(destino)
        self.vertices[destino].append(origen)
        self.pesos[origen].append((destino, peso))
        self.pesos[destino].append((origen, peso))  # Grafo no dirigido

    def conexo(self):
        if not self.vertices:
            return False
        primer_areop = next(iter(self.vertices))  # Se escoge el primer aeropuerto
        visit = {airport: False for airport in self.vertices}  # Inicializa la lista
        cantvisitados, visit = self.__DFS_visit(primer_areop, visit)
        if cantvisitados == len(self.vertices):#Si los visitados es igual a los areopuertos existentes
            print("El grafo generdo es conexo")
        else:
            print("El grafo generado no es conexo")
            cantcomponentes = 1
            print("La componente ", cantcomponentes, " tiene ", cantvisitados)
            self.vercomponentes(cantcomponentes, visit, cantvisitados)

    def __DFS_visit(self, airport, visit):
        visit[airport] = True  # Marca el aeropuerto como visitado
        for vecino in self.vertices[airport]:
            if not visit[vecino]:  # Si el vecino no ha sido visitado
                self.__DFS_visit(vecino, visit)  # Llama para visitar al vecino
        return sum(visit.values()), visit  # Retorna la cantidad de aeropuertos visitados

    def vercomponentes(self, cantcomponentes, visit, canttrue):
        cantcomponentes += 1
        while canttrue <= len(self.vertices):
            areopuertonovisitado = None
            for airport, estado in visit.items():
                if not estado:  # Si el estado es False
                    areopuertonovisitado = airport
                    break
            if areopuertonovisitado is None:
                break  # Salir del bucle
            trues, visit = self.__DFS_visit(areopuertonovisitado, visit)
            verticescomponente = trues - canttrue
            canttrue = trues
            print("La componente ", cantcomponentes, " tiene ", verticescomponente, " vertices")
            self.vercomponentes(cantcomponentes, visit, canttrue)
        return cantcomponentes

    def verinfoaeropuerto(self, codigo):
        if codigo in self.info_aeropuertos:
            info = self.info_aeropuertos[codigo]
            print(f"Código: {info['codigo']}")
            print(f"Nombre: {info['nombre']}")
            print(f"Ciudad: {info['ciudad']}")
            print(f"País: {info['pais']}")
            print(f"Latitud: {info['latitud']}")
            print(f"Longitud: {info['longitud']}")
        else:
            print(f"No se encontró información para el aeropuerto con código {codigo}")

    def mostrar_grafo(self):
        for airport, adyacentes in self.vertices.items():
            print(f"{airport}: {', '.join(adyacentes)}")
    
    def calculodistancia(self, lat1, lon1, lat2, lon2):
        # Convertir de grados a radianes
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        # Fórmula de Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371.0  # Radio de la Tierra en km
        return c * r
    

    def dijkstra(self, origen, destino):
    # Inicializa todas las distancias con infinito y el conjunto de visitados
        distancias = {v: math.inf for v in self.vertices}
        distancias[origen] = 0  # La distancia al origen es 0
        visitados = set()  # Conjunto de vértices visitados
        padres = {v: None for v in self.vertices}  # Guarda los padres en los caminos mínimos

        while len(visitados) < len(self.vertices):
            v_min = None  # Seleccionamos el vértice no visitado con la menor distancia actual
            for v in self.vertices:
                if v not in visitados and (v_min is None or distancias[v] < distancias[v_min]):
                    v_min = v 

            if v_min is None:
                break  # Si no queda ningún vértice alcanzable, salimos del bucle

            visitados.add(v_min)  # Marca el vértice como visitado

            # Actualizamos las distancias de sus vecinos
            for vecino, peso in self.pesos[v_min]:
                if vecino not in visitados:
                    nueva_dist = distancias[v_min] + peso
                    if nueva_dist < distancias[vecino]:
                        distancias[vecino] = nueva_dist
                        padres[vecino] = v_min  # Guarda el padre para reconstruir el camino

        if destino is not None:
            if math.isinf(distancias[destino]):
                print("No hay un camino mínimo posible entre los aeropuertos")  # Validación de si no existe el camino mínimo
                return None
            else:
                caminos = self.recuperarecorrido(origen, destino, padres)
                return caminos

        return distancias, padres  # Devuelve distancias y padres si no se especifica un destino

    
    def crear_mapa(self, camino, op):
        # Obtener el primer aeropuerto para centrar el mapa
        inicio = camino[0]
        lat_inicial = self.info_aeropuertos[inicio]['latitud']
        lon_inicial = self.info_aeropuertos[inicio]['longitud']

    # Crear un mapa centrado en el primer aeropuerto
        mapa = folium.Map(location=[lat_inicial, lon_inicial], zoom_start=6)

    # Añadir marcadores para cada aeropuerto en el camino
        for aeropuerto in camino:
            info = self.info_aeropuertos[aeropuerto]
            folium.Marker(
            [info['latitud'], info['longitud']],
            popup=f"{info['nombre']} ({info['codigo']})",
            tooltip=f"{info['ciudad']}, {info['pais']}"
        ).add_to(mapa)

    # Añadir líneas entre los aeropuertos en el camino
        puntos = [(self.info_aeropuertos[aerop]['latitud'], self.info_aeropuertos[aerop]['longitud']) for aerop in camino]
        folium.PolyLine(puntos, color="blue", weight=2.5, opacity=1).add_to(mapa)

    # Guardar el mapa en la ruta de tu escritorio
        mapa.save(r"C:\Users\lache\Desktop\laboratorio2M\mapa_geolocalizacion.html")
        print("Mapa guardado en el escritorio")
    

    def mapageo(self):
        # Crear un mapa centrado en una ubicación promedio
        mapa = folium.Map(location=[self.info_aeropuertos[list(self.vertices)[0]]['latitud'], 
                                    self.info_aeropuertos[list(self.vertices)[0]]['longitud']],
                        zoom_start=6)

        # Añadir marcadores para cada aeropuerto en el camino
        for aeropuerto in self.vertices:
            info = self.info_aeropuertos[aeropuerto]
            
            # Verifica que la latitud y longitud sean válidas
            if pd.notna(info['latitud']) and pd.notna(info['longitud']):
                folium.Marker(
                    location=[info['latitud'], info['longitud']],
                    popup=f"{info['nombre']} ({info['codigo']})",
                    tooltip=f"{info['ciudad']}, {info['pais']}"
                ).add_to(mapa)
        # Guardar el mapa como un archivo HTML
        mapa.save(r"C:\Users\lache\Desktop\laboratorio2M\mapa_geolocalizacion.html")



    def recuperarecorrido(self, inicial, final, padres):
        recorrido=[]
        recorrido.append(final)
        print("Camino mínimo de ",inicial, "hasta ",final,": \n")
        while inicial != final: #Recupera el camino minimo entre dos areopuertos
            final = padres[final]
            recorrido.append(final)
        
        caminomin= recorrido.copy()
        while caminomin:
            vert= caminomin.pop()
            print(self.verinfoaeropuerto(vert),"\n")
        return recorrido
    

    def graficar(self, camino, inicio, caminoresaltado):
        G = nx.DiGraph() if self.directed else nx.Graph()

    # Añadir los nodos y aristas correspondientes al grafo
        for u in camino:
            for vecino, peso in self.pesos[u]:
                if vecino in camino:  # Solo incluimos los vértices del camino mínimo
                    G.add_edge(u, vecino, weight=round(peso, 2))
    
        pos = nx.spring_layout(G)  # Posición de los nodos en el grafo
        labels = nx.get_edge_attributes(G, 'weight')  # Etiquetas con los pesos

    # Dibujar el grafo
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=400, font_size=10)
    
    # Resaltar el camino mínimo
        for i in range(len(caminoresaltado) - 1):
            nx.draw_networkx_edges(G, pos, edgelist=[(caminoresaltado[i], caminoresaltado[i+1])], edge_color='red', width=2)
    
    # Mostrar las etiquetas de los pesos
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.5, font_size=8)

        plt.title("Grafo con camino mínimo resaltado")
        plt.show()
    

    def mostrar_aeropuertos_caminos_minimos(self, origen):
        if origen not in self.vertices:
            print(f"El aeropuerto {origen} no está en el grafo.")
            return

        # Obtener distancias usando Dijkstra
        distancias, _ = self.dijkstra(origen, None)

        # Filtrar aeropuertos con distancia finita
        aeropuertos_validos = {codigo: dist for codigo, dist in distancias.items() if not math.isinf(dist)}
        print("antes",distancias["MPN"])

        # Ordenar los aeropuertos por distancia de mayor a menor
        aeropuertos_ordenados = sorted(aeropuertos_validos.items(), key=lambda x: x[1], reverse=True)

        print(f"\nLos 10 aeropuertos más lejanos desde {origen}:\n")
        print(f"{'No.':<4} {'Código':<8} {'Nombre':<30} {'Ciudad':<20} {'País':<15} {'Latitud':<10} {'Longitud':<10} {'Distancia (km)':<15}")
        print("="*120)

        # Limitar a 10 aeropuertos
        for i, (codigo, distancia) in enumerate(aeropuertos_ordenados[:10]):
            if codigo in self.info_aeropuertos:
                info = self.info_aeropuertos[codigo]
                print(f"{i + 1:<4} {codigo:<8} {info['nombre']:<30} {info['ciudad']:<20} {info['pais']:<15} "
                  f"{info['latitud']:<10} {info['longitud']:<10} {distancia:<15.2f}")
            else:
                print(f"{i + 1:<4} {codigo:<8} {'Sin información disponible':<30} {'':<20} {'':<15} {'':<10} {'':<10} {distancia:<15.2f}")

  
    
    def kruskal(self):
        
        def find(vertice, parent):
            if parent[vertice] == vertice:
                return vertice
            parent[vertice] = find(parent[vertice], parent) 
            return parent[vertice]

        def union(vertice1, vertice2, parent, rank, peso_componentes, count_componentes, peso_arista):
            root1 = find(vertice1, parent)
            root2 = find(vertice2, parent)

            if root1 != root2:
                # Se unen las componentes y se suma los pesos de cada vertice
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                    peso_componentes[root1] += peso_arista + peso_componentes[root2]  # Suma de la arista y de root2
                    count_componentes[root1] += count_componentes[root2]  
                    count_componentes[root2] = 0 
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                    peso_componentes[root2] += peso_arista + peso_componentes[root1]  # Suma de la arista y de root1
                    count_componentes[root2] += count_componentes[root1]  
                    count_componentes[root1] = 0  
                else:
                    parent[root2] = root1
                    peso_componentes[root1] += peso_arista + peso_componentes[root2]  
                    count_componentes[root1] += count_componentes[root2] 
                    count_componentes[root2] = 0  
                    rank[root1] += 1

        # Creamos la cola de prioridad con las aristas ordenadas
        aristas = []
        for origen in self.pesos:
            for destino, peso in self.pesos[origen]:
                if origen < destino:  # Esto para evitar duplicar aristas en ambos sentidos
                    aristas.append((peso, origen, destino))

        heapq.heapify(aristas)  # Ordenar las aristas por peso

        # PARA QUE NO SE TE VUELVA A OLVIDAR INCIALIZAR
        parent = {v: v for v in self.vertices}  
        rank = {v: 0 for v in self.vertices}  
        peso_componentes = {v: 0 for v in self.vertices}  
        count_componentes = {v: 1 for v in self.vertices}  # Contador de vértices por componente

        #  INICIO Kruskal
        while aristas:
            peso, origen, destino = heapq.heappop(aristas)
            
            root_origen = find(origen, parent)
            root_destino = find(destino, parent)

            if root_origen != root_destino:
                # Unimos las componentes y actualizamos los pesos
                union(origen, destino, parent, rank, peso_componentes, count_componentes, peso)

        # Imprimir los resultados de las componentes
        componentes_encontradas = {}
        for v in self.vertices:
            root = find(v, parent)
            if root not in componentes_encontradas:
                componentes_encontradas[root] = (count_componentes[root], peso_componentes[root])
        
        print(f"El grafo tiene {len(componentes_encontradas)} componentes.")
        for i, (num_vertices, peso) in enumerate(componentes_encontradas.values(), start=1):
            print(f"Componente {i} con {num_vertices} vértices: Peso del árbol de expansión mínima: {peso:.2f}")
    
    
print("---------------------------------------------")
graf = Graph()
# Obtener la ruta del directorio del script
dir_path = os.path.dirname(os.path.abspath(__file__))
# Crear la ruta completa al archivo CSV
csv_path = os.path.join(dir_path, "flights_final.csv")
# Cargar el dataset
dtset = pd.read_csv(csv_path, header=None)
dtset.columns = ["Source Airport Code", "Source Airport Name", "Source Airport City", "Source Airport Country", "Source Airport Latitude", "Source Airport Longitude", "Destination Airport Code", "Destination Airport Name", "Destination Airport City", "Destination Airport Country", "Destination Airport Latitude", "Destination Airport Longitude"]

n = 0
print("---------------------------------------------")
numfila = 0
for index, row in dtset.iloc[0:].iterrows():
    if numfila > 0:
        # Agregar el aeropuerto de origen con su información
        graf.agregar_vertice(row['Source Airport Code'], {
            'codigo': row['Source Airport Code'],
            'nombre': row['Source Airport Name'],
            'ciudad': row['Source Airport City'],
            'pais': row['Source Airport Country'],
            'latitud': float(row['Source Airport Latitude']),
            'longitud': float(row['Source Airport Longitude'])
        })

        # Agregar el aeropuerto de destino con su información
        graf.agregar_vertice(row['Destination Airport Code'], {
            'codigo': row['Destination Airport Code'],
            'nombre': row['Destination Airport Name'],
            'ciudad': row['Destination Airport City'],
            'pais': row['Destination Airport Country'],
            'latitud': float(row['Destination Airport Latitude']),
            'longitud': float(row['Destination Airport Longitude'])
        })

        # Calcular la distancia entre el aeropuerto de origen y destino
        dist = graf.calculodistancia(float(row['Source Airport Latitude']), float(row['Source Airport Longitude']),
                                      float(row['Destination Airport Latitude']), float(row['Destination Airport Longitude']))

        # Agregar una arista entre los aeropuertos
        graf.agregar_arista(row['Source Airport Code'], row['Destination Airport Code'], dist)
    numfila += 1

print("Grafo generado correctamente, se registraron ", len(graf.vertices))
print("---------------------------------------------")


while True:
    print("MENÚ")
    op = int(input("1. Ver mapa de geolocalización\n2. Verificar si es conexo el grafo\n3. Ver peso de MST \n4. Buscar aeropuertos por código\n5. Graficar camino mínimo\n6. Salir\nSu opción: "))
    if op == 1:
        print("---------------------------------------------")
        print("Mapa guardado en el escritorio")
        graf.mapageo()
        print("---------------------------------------------")
    
    elif op ==2:
        print("---------------------------------------------")
        graf.conexo()
        print("---------------------------------------------")

    elif op == 3:
        print("---------------------------------------------")
        graf.kruskal()
        print("---------------------------------------------")

    elif op == 4:
        print("---------------------------------------------")
        sub_op = int(input("1. Buscar aeropuerto por código\n2. Mostrar aeropuertos más lejanos\nSu opción: "))
        if sub_op == 1:
            codigo = input("Ingrese el código del aeropuerto: ").upper()
            graf.verinfoaeropuerto(codigo)
        elif sub_op == 2:
            origen = input("Ingrese el código del aeropuerto de origen: ").upper()
            graf.mostrar_aeropuertos_caminos_minimos(origen)
        else:
            print("Opción invalida")
        print("---------------------------------------------")

    elif op == 5:
        print("---------------------------------------------")
        codorigen = input("Ingrese el código del aeropuerto de origen: ").upper()
        while True:
            if codorigen not in graf.vertices:
                print("El código ingresado es inválido")
                codorigen = input("Ingrese el código del aeropuerto de origen: ").upper()
            else:
                break
        coddestino = input("Ingrese el código del aeropuerto de destino: ").upper()
        while True:
            if coddestino not in graf.vertices:
                print("El código ingresado es inválido")
                coddestino = input("Ingrese el código del aeropuerto de destino: ").upper()
            else:
                break

        # Llamada a Dijkstra para obtener el camino
        camino = graf.dijkstra(codorigen, coddestino)
        caminoresaltado= camino.copy()
        if camino is not None:
            graf.crear_mapa(camino, op)
            graf.graficar(camino, codorigen, caminoresaltado)  # Graficar solo el camino
            print("Mapa con pesos generado correctamente.")
        else:
            print("No se encontró un camino mínimo entre los aeropuertos seleccionados.")
        print("---------------------------------------------")

    elif op == 6:
        print("Saliendo del programa...")
        break

    else:
        print("Opción inválida")
