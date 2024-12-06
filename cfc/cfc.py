from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        """
        Cette classe permet de trouver les composante fortement connexe

        :param vertices: le nombre de sommets
        """
        self.V = vertices
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        """
        Cette function permet d'ajouter les arrête du graphe
        :param u: sommet principal
        :param v: sommet secondaire
        :return:
        """
        self.graph[u].append(v)


    def dfs(self, v, visited, stack):
        """
        Algorihtme du profondeur d'abord
        :param v: Nombre de sommet
        :param visited: deja visité ou pas
        :param stack: pour empiler
        :return:
        """
        visited[v] = True
        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                self.dfs(neighbor, visited, stack)
        stack.append(v)


    def transpose(self):
        """

        :return:
        """
        g_t = Graph(self.V)
        for u in self.graph:
            for v in self.graph[u]:
                g_t.add_edge(v, u)
        return g_t

    def dfs_util(self, v, visited):
        visited[v] = True
        print(v, end=' ')
        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                self.dfs_util(neighbor, visited)


    def find_sccs(self):
        """
        Trouver et imprimer toutes les composante fortement connecté
        :return:
        """
        stack = []
        visited = [False] * self.V

        for i in range(self.V):
            if not visited[i]:
                self.dfs(i, visited, stack)

        g_t = self.transpose()

        visited = [False] * self.V

        while stack:
            v = stack.pop()
            if not visited[v]:
                g_t.dfs_util(v, visited)
                print()


if __name__=="__main__":

    graph = Graph(11)
    graph.add_edge(0,)