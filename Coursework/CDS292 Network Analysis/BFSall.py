def BFS(G,ori,dest=None):
	d={}
	l=0
	w=list(G.nodes())
	d[ori]=l
	w.remove(ori)
	ActvShell=[ori]
	while len(ActvShell)>0:
		l=l+1
		NewActvShell=[]
		for node in ActvShell:
			for neigh in G.neighbors(node):
				if neigh==dest:
					d[neigh]=l
					return(l)
				if neigh in w:
					NewActvShell.append(neigh)
					w.remove(neigh)
					d[neigh]=l
		ActvShell=NewActvShell
	if dest!=None:
		return(-1)
	return(d)

def BFSall(G):
    d = {}
    for origin in G.nodes():
        d[origin] = {}
        s = BFS(G, origin)
        for j in s.keys():
            d[origin][j] = s[j]
    return d
