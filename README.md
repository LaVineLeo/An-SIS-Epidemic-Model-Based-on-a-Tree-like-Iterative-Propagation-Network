# An-SIS-Epidemic-Model-Based-on-a-Tree-like-Iterative-Propagation-Network
# 基于树状迭代传播网络的SIS传染病模型
传染病模型是预测和控制传染病传播的重要工具之一，而树状迭代传播网络可以更准确地模拟现实世界中的传染病传播过程。

模型的构建步骤如下：
1. 构建树状迭代传播网络：开发一个算法或方法来构建树状迭代传播网络，其中节点表示个体，边表示传播关系。
2. 实现SIS传染病模型：实现SIS传染病模型的基本方程和参数。
3. 模拟传染病传播过程：根据构建的树状迭代传播网络和SIS传染病模型，模拟传染病在网络中的传播过程。
4. 分析模拟结果：提供分析工具和可视化界面，用于分析模拟结果和探索传染病传播的特点。

#1 SIS网络的计算原理

令N为总人口数，S(t)表示在时间t未感染的人数，I(t)表示在时间t感染的人数。则S(t) + I(t) = N，即总人口数不变。

SIS模型的动力学方程可以表示为：
dS(t)/dt = - beta * S(t) * I(t) + mu * I(t)
dI(t)/dt = beta * S(t) * I(t) - mu * I(t)

# # 1.1 代码复现
```
import networkx as nx
import numpy as np

def sis_model(graph, beta, mu, initial_infected):
    """
    SIS传染病模型的计算公式
    :param graph: 输入的网络图，使用networkx库表示
    :param beta: 传染率
    :param mu: 恢复率
    :param initial_infected: 初始感染节点列表
    :return: 每个时间步的感染节点数列表
    """
    num_nodes = len(graph.nodes)
    infected = np.zeros(num_nodes)  # 记录每个节点的感染状态，0表示未感染，1表示感染
    infected[initial_infected] = 1  # 设置初始感染节点

    infected_counts = []  # 记录每个时间步的感染节点数

    for t in range(100):  # 迭代100个时间步
        new_infected = np.zeros(num_nodes)  # 记录每个时间步新增感染的节点数

        for node in graph.nodes:
            if infected[node] == 1:  # 如果节点已感染
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    if infected[neighbor] == 0:  # 如果邻居节点未感染
                        if np.random.rand() < beta:  # 根据传染率决定是否感染邻居节点
                            new_infected[neighbor] += 1

            elif infected[node] == 0:  # 如果节点未感染
                if np.random.rand() < mu:  # 根据恢复率决定是否从感染状态恢复
                    infected[node] = 1

        infected += new_infected  # 更新感染节点数
        infected_counts.append(np.sum(infected))  # 记录当前时间步的感染节点数

    return infected_counts
```
# 2 SIS网络的可视化设置
# # 2.1 SIS网络的图绘制有几个需要注意的点：
1.网络布局：选择适合的网络布局可以更好地展示网络结构。常见的网络布局包括随机布局、环形布局、力导向布局等。根据网络规模和结构特点，选择合适的布局方式可以更好地展示节点之间的关系。
2.节点大小和颜色：可以根据节点的属性来决定节点的大小和颜色。例如，可以根据节点的度中心性或其他重要性指标来设置节点的大小，以突出重要节点。另外，可以根据节点的感染状态来设置节点的颜色，以区分感染节点和未感染节点。
3.边的粗细和颜色：可以根据边的权重或连接强度来设置边的粗细和颜色。例如，可以根据边的权重来设置边的粗细，以展示节点之间的连接强度。另外，可以根据边的方向或类型来设置边的颜色，以区分不同类型的边或表示不同的传播方向。
4.标签显示：可以选择性地显示节点或边的标签，以展示节点或边的相关信息。标签可以包括节点的名称、属性值、度中心性等信息。在网络较大时，可以考虑只显示部分节点或边的标签，以避免视觉混乱。
5.动态效果：如果希望展示SIS模型的动态过程，可以考虑添加动态效果，例如逐步更新节点颜色或显示感染节点的扩散过程。这样可以更直观地展示疾病的传播过程。
6.图例和标题：为了帮助理解图表，可以添加图例和标题来解释节点和边的含义，以及整个可视化的目的和主题。
```
def show_iteration(Connections,Amount,Beta):        #传染病迭代输出模型        #Connections为网络关系矩阵，Amount为初始（0时期）感染者数量
    C = copy.deepcopy(Connections)
    Nodes = len(C)
    InfecterStatus = catastrophe(Nodes,Amount)        #根据设定的初始感染数，在随机位置生成感染者
    g = nx.Graph()                                  #新建画布
    for n in range(Nodes):                          #在画布上设置节点
        g.add_node(n)
    for ed in range(Nodes):                         #在画布上设置边（连结关系）
        for lin in range(ed+1,Nodes):
            if C[ed][lin] == 1:
                g.add_edge(ed,lin)
    pos  =nx.kamada_kawai_layout(g)                 #kamada-kawai路径长度成本函数计算
    Status = {}
    times = 0
    while sum(InfecterStatus) <= Nodes:             #当感染数大于等于节点数时停止迭代
        # plt.imshow(InfecterStatus)
        # plt.pause(3)#帧数
        for s in range(len(InfecterStatus)):        #把感染状态写入字典
            SI = InfecterStatus[s]
            Status[s] = SI
        colors = []
        for c in g:                                 #分配各节点颜色表示感染状态
            sta = Status[c]
            if sta == 1:
                clr = 'r'
            if sta == 0 :
                clr = 'g'
            colors.append(clr)
        nodesize = []
        for ns in g:
            de = ((sum(C[ns])*10)+50)                 #节点大小(节点度数越大，节点越大)alse
            nodesize.append(de)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(g , pos=pos , label=True , node_color=colors , node_size=nodesize , alpha=0.6)
        nx.draw_networkx_edges(g , pos=pos , label=True , width=0.3 , alpha=0.3)
        print(f'迭代第 {times} 次 ---- 感染者数量：{sum(InfecterStatus)} ---- 占比：{(sum(InfecterStatus)/Nodes)}')
        plt.show()
        if sum(InfecterStatus) == Nodes:
            Nodes = Nodes - 1
        InfecterStatus = infect(C,InfecterStatus,Beta)     #传染模型
        times += 1
    print('---------- 迭代完成 ----------')

```
# # 2.2 可视化效果（包括网络分布与迭代过程）

![index](https://github.com/LaVineLeo/An-SIS-Epidemic-Model-Based-on-a-Tree-like-Iterative-Propagation-Network/blob/main/tree_network.png)


后话：其实整个SIS传染病模型的实现并不难，还能根据各个垂直领域的大数据特征来仿真风险的传染路径，这样其实比传统的链路预测做得会更科学一点的，特别是对于某些行业的限制，如建筑物料供应运输网络，它们自身有着很大的运输半径的限制。
PS：欢迎在Issue中留言讨论更多的研究可能性

作者：LaVine

