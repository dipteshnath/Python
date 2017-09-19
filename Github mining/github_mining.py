
# coding: utf-8

# IDENTIFY POPULAR USER AND REPOSITORY

# Get OAuth token for the user 

# In[1]:


# Obtain API token
import requests
from getpass import getpass
import json
username='dipteshnath@gmail.com'
password='Google@5654'
url = 'https://api.github.com/authorizations'
note = 'Mining GitHub'
post_data={'scopes':['repo'],'note':note}

response=requests.post(url,auth=(username,password),data=(json.dumps(post_data)),)
print("API response", response.text)
print()
print("Your OAuth token is", response.json()['token'])


# At first I will seed an interest graph from a popular GitHub repository and create connections between it and its stargazers. Listing the stargazers for a repository is possible with the List Stargazers API. I will use authenticated request using the given ACCESS TOKEN. 

# In[4]:


from github import Github
ACCESS_TOKEN = 'a7fcdb93177c4bdb70429ae9971f2e0f7c875b2b'#'26d46b82678151d41e5ec2a68a8e26e2c1a348fd'

USER = 'Raynos' # User Name
REPO = 'mercury'# Repository name

client = Github(ACCESS_TOKEN, per_page=100)
user = client.get_user(USER)
repo = user.get_repo(REPO)

# Star gazers are number of people whoo have bookmarked the repo of a particular 

stargazers = [ s for s in repo.get_stargazers() ]
print("Number of star gazers", len(stargazers))


# In[ ]:


Here I have initialized the a small graph and printed the intial values


# In[2]:


import networkx as nx
# Create a directed graph
g = nx.DiGraph()
# Add an edge to the directed graph from X to Y
g.add_edge('X', 'Y')
# Print some statistics about the graph
print(nx.info(g))
print()
# Get the nodes and edges from the graph
print("Nodes:", g.nodes())
print("Edges:", g.edges())
print()
# Get node properties
print("X props:", g.node['X'])
print("Y props:", g.node['Y'])
# Get edge properties
print("X=>Y props:", g['X']['Y'])
print()
# Update a node property
g.node['X'].update({'prop1' : 'value1'})
print("X props:", g.node['X'])
print()
# Update an edge property
g['X']['Y'].update({'label' : 'label1'})
print("X=>Y props:", g['X']['Y'])


# Expanded the above graph by adding the other users and their repository

# In[3]:


import networkx as nx
g = nx.DiGraph()
g.add_node(repo.name + '(repo)', type='repo', lang=repo.language, owner=user.login)

for sg in stargazers:
    g.add_node(sg.login + '(user)', type='user')
    g.add_edge(sg.login + '(user)', repo.name + '(repo)', type='gazes')


# Here we are showing some graph statistics 

# In[29]:


print(nx.info(g))
print('------------------------')
print(g.node['mercury(repo)'])
print(g.node['Raynos(user)'])
print('---------------------------')
print(g['Raynos(user)']['mercury(repo)'])
print(g['Raynos(user)'])
print(g['mercury(repo)'])
print('-----------In edges of the user-------------')
print(g.in_edges(['Raynos(user)']))
print('-----------Out edges of the user-------------')
print(g.out_edges(['Raynos(user)']))
print('-------------In edges of the repository-----------')
print(g.in_edges(['mercury(repo)']))
print('---------------Out edges of the repository---------')
print(g.out_edges(['repo)']))


# Create relationship between star gazers nodes if any exists

# In[31]:


import sys
for i, sg in enumerate(stargazers):
    try:
        for follower in sg.get_followers():
            if follower.login + '(user)' in g:
                g.add_edge(follower.login + '(user)', sg.login + '(user)', 
                           type='follows')
    except Exception as e: #ssl.SSLError
        print >> sys.stderr, "Encountered an error fetching followers for",                              sg.login, "Skipping."
        print >> sys.stderr, e

    print("Processed", i+1, " stargazers. Num nodes/edges in graph",           g.number_of_nodes(), "/", g.number_of_edges())
    print("Rate limit remaining", client.rate_limiting)


# In[34]:


from operator import itemgetter
from collections import Counter

# Check social edges we added since last time.
print(nx.info(g))
print('---------------------------------')

# The number of "follows" edges is the difference
print(len([e for e in g.edges_iter(data=True) if e[2]['type'] == 'follows']))
print('----------------------------------')

# The repository owner is possibly one of the more popular users in this graph.
print(len([e 
           for e in g.edges_iter(data=True) 
               if e[2]['type'] == 'follows' and e[1] == 'Raynos(user)']))
print('-----------------------------------')

# Let's examine the number of adjacent edges to each node
print(sorted([n for n in g.degree_iter()], key=itemgetter(1), reverse=True)[:10])
print('------------------------------------')
# Our central user who is followed by many

print(len(g.out_edges('Raynos(user)')))
print(len(g.in_edges('Raynos(user)')))
print('----------------------------------------')
# The number of popular users and top 10 popular users
c = Counter([e[1] for e in g.edges_iter(data=True) if e[2]['type'] == 'follows'])
popular_users = [ (u, f) for (u, f) in c.most_common() if f > 1 ]
print("Number of popular users", len(popular_users))
print("Top 10 popular users:", popular_users[:10])


# Degree centrality of a node in the graph is a measure of the number of incident edges upon it.
# Betweenness centrality of a node is a measure of how often it connects any other nodes in the graph in the sense of being in between other nodes.
# Closeness centrality of a node is a measure of how highly connected (“close”) it is to all other nodes in the graph.
# TO not to get a biased value of this parameters I have removed the central repository.

# In[35]:


from operator import itemgetter
h = g.copy()
h.remove_node('mercury(repo)')
dc = sorted(nx.degree_centrality(h).items(), 
            key=itemgetter(1), reverse=True)

print("Degree Centrality")
print(dc[:10])
print('-------------------------------------------')

bc = sorted(nx.betweenness_centrality(h).items(), 
            key=itemgetter(1), reverse=True)

print("Betweenness Centrality")
print(bc[:10])
print('--------------------------------------------')

print("Closeness Centrality")
cc = sorted(nx.closeness_centrality(h).items(), 
            key=itemgetter(1), reverse=True)
print(cc[:10])

