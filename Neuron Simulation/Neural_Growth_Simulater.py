import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------- 超参数定义 ----------------------------

# 定义神经元空间的最大范围
MaxWidth = 100
MaxLength = 100
MaxHeight = 100

LinkThreshold = 5  # 连接阈值距离

node = []
epoch = 10  # 增加模拟的次数以观察更复杂的连接
num_of_node = 2  # 增加神经元数量以展示更多连接
connections = []

# 定义每个神经元的树突和轴突数量
dendrites_per_node = 3
axons_per_node = 3

# 定义树突和轴突的最大生长步数以限制其长度
max_dendrite_steps = 10
max_axon_steps = 20

# 兴奋值定义（范围0到10）
excitation_dendrite = 0  # 树突的兴奋值
excitation_axon = 0.2  # 轴突的兴奋值 0~1

# 统计信息保存路径
log_file_path = 'neural_connections.txt'

# ---------------------------- 神经元类定义 ----------------------------

class NeuralNode():
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.dendrites = [self.initialize_dendrite() for _ in range(dendrites_per_node)]
        self.axons = [self.initialize_axon() for _ in range(axons_per_node)]
        self.color_dendrite = (random.random(), random.random(), random.random())  # 树突颜色
        self.color_axon = (random.random(), random.random(), random.random())      # 轴突颜色

    def initialize_dendrite(self):
        # 初始化树突，限制其生长
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        direction = [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ]
        return {
            'path': [(self.x, self.y, self.z)],
            'direction': direction,
            'stopped': False,  # 标记树突是否停止生长
            'steps': 0,         # 树突生长步数
            'excitation': excitation_dendrite,  # 树突的兴奋值
            'growth_counter': 0  # 控制生长速度
        }

    def initialize_axon(self):
        # 初始化轴突，具有目标导向的生长
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        direction = [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ]
        return {
            'path': [(self.x, self.y, self.z)],
            'direction': direction,
            'stopped': False,  # 标记轴突是否停止生长
            'steps': 0,         # 轴突生长步数
            'excitation': excitation_axon   # 轴突的兴奋值
        }

    def grow_dendrite(self, dendrite, step=1):
        """
        生长树突，限制其生长步数，并根据兴奋值决定是否生长
        """
        if dendrite['stopped']:
            return

        if dendrite['steps'] >= max_dendrite_steps:
            dendrite['stopped'] = True
            return

        # 控制树突的生长速度：例如每两次迭代生长一次
        dendrite['growth_counter'] += 1
        if dendrite['growth_counter'] < 2:
            return
        dendrite['growth_counter'] = 0  # 重置计数器

        last_x, last_y, last_z = dendrite['path'][-1]
        dx, dy, dz = dendrite['direction']

        # 根据兴奋值决定生长方向
        probability_directed = dendrite['excitation'] / 10.0  # 将兴奋值映射为概率
        if random.random() < probability_directed:
            # 朝向目标导向的生长
            target = self.select_target()
            if target:
                target_x, target_y, target_z = target.x, target.y, target.z
                vec_to_target = [
                    target_x - last_x,
                    target_y - last_y,
                    target_z - last_z
                ]
                distance = math.sqrt(vec_to_target[0]**2 + vec_to_target[1]**2 + vec_to_target[2]**2)
                if distance != 0:
                    direction = [vec_to_target[0]/distance, vec_to_target[1]/distance, vec_to_target[2]/distance]
                    self.set_direction(dendrite, direction)
        else:
            # 随机生长
            angle_change = random.uniform(-math.pi / 36, math.pi / 36) * (1 - dendrite['excitation'] / 10.0)
            self.randomly_change_direction(dendrite, angle_change)

        # 计算新位置
        new_x = last_x + dendrite['direction'][0] * step
        new_y = last_y + dendrite['direction'][1] * step
        new_z = last_z + dendrite['direction'][2] * step

        # 检查是否超出边界
        if 0 <= new_x <= MaxWidth and 0 <= new_y <= MaxLength and 0 <= new_z <= MaxHeight:
            dendrite['path'].append((new_x, new_y, new_z))
            dendrite['steps'] += 1
        else:
            dendrite['stopped'] = True

    def grow_axon(self, axon, step=2):
        """
        生长轴突，朝向目标神经元，并根据兴奋值调整生长方向
        """
        if axon['stopped']:
            return

        if axon['steps'] >= max_axon_steps:
            axon['stopped'] = True
            return

        last_x, last_y, last_z = axon['path'][-1]
        dx, dy, dz = axon['direction']

        # 根据兴奋值决定生长方向
        probability_directed = axon['excitation'] / 10.0  # 将兴奋值映射为概率
        if random.random() < probability_directed:
            # 朝向目标导向的生长
            target = self.select_target()
            if target:
                target_x, target_y, target_z = target.x, target.y, target.z
                vec_to_target = [
                    target_x - last_x,
                    target_y - last_y,
                    target_z - last_z
                ]
                distance = math.sqrt(vec_to_target[0]**2 + vec_to_target[1]**2 + vec_to_target[2]**2)
                if distance != 0:
                    direction = [vec_to_target[0]/distance, vec_to_target[1]/distance, vec_to_target[2]/distance]
                    self.set_direction(axon, direction)
        else:
            # 随机生长
            angle_change = random.uniform(-math.pi / 18, math.pi / 18) * (1 - axon['excitation'] / 10.0)
            self.randomly_change_direction(axon, angle_change)

        # 计算新位置
        new_x = last_x + axon['direction'][0] * step
        new_y = last_y + axon['direction'][1] * step
        new_z = last_z + axon['direction'][2] * step

        # 检查是否超出边界
        if 0 <= new_x <= MaxWidth and 0 <= new_y <= MaxLength and 0 <= new_z <= MaxHeight:
            axon['path'].append((new_x, new_y, new_z))
            axon['steps'] += 1
        else:
            axon['stopped'] = True

    def set_direction(self, structure, new_direction):
        """
        设置生长方向
        """
        norm = math.sqrt(new_direction[0]**2 + new_direction[1]**2 + new_direction[2]**2)
        if norm == 0:
            return
        structure['direction'] = [new_direction[0]/norm, new_direction[1]/norm, new_direction[2]/norm]

    def randomly_change_direction(self, structure, angle_change):
        """
        随机微调生长方向
        """
        dx, dy, dz = structure['direction']
        axis = random.choice(['x', 'y', 'z'])
        if axis == 'x':
            new_dy = dy * math.cos(angle_change) - dz * math.sin(angle_change)
            new_dz = dy * math.sin(angle_change) + dz * math.cos(angle_change)
            new_dx = dx
        elif axis == 'y':
            new_dx = dx * math.cos(angle_change) + dz * math.sin(angle_change)
            new_dz = -dx * math.sin(angle_change) + dz * math.cos(angle_change)
            new_dy = dy
        else:
            new_dx = dx * math.cos(angle_change) - dy * math.sin(angle_change)
            new_dy = dx * math.sin(angle_change) + dy * math.cos(angle_change)
            new_dz = dz

        # 归一化方向向量
        norm = math.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
        if norm == 0:
            return
        structure['direction'] = [new_dx / norm, new_dy / norm, new_dz / norm]

    def select_target(self):
        """
        随机选择一个目标神经元，排除自身
        """
        possible_targets = [n for n in node if n.id != self.id]
        if not possible_targets:
            return None
        return random.choice(possible_targets)

    def Distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def connect_axon_to_dendrite(self):
        """检查轴突是否连接到其他神经元的树突，若连接则停止该树突的生长"""
        global connections
        for axon in self.axons:
            if axon['stopped']:
                continue
            last_axon = axon['path'][-1]
            for other_node in node:
                if other_node != self:
                    for dendrite in other_node.dendrites:
                        if dendrite['stopped']:
                            continue
                        last_dendrite = dendrite['path'][-1]
                        distance = math.sqrt(
                            (last_axon[0] - last_dendrite[0]) ** 2 +
                            (last_axon[1] - last_dendrite[1]) ** 2 +
                            (last_axon[2] - last_dendrite[2]) ** 2
                        )
                        if distance < LinkThreshold:
                            connections.append((self.id, axon['path'][-1], other_node.id, dendrite['path'][-1]))
                            dendrite['stopped'] = True  # 仅停止该树突的生长
                            # 根据兴奋值决定是否停止轴突的生长
                            if axon['excitation'] > 5:
                                axon['stopped'] = True  # 高兴奋值的轴突停止生长
                            # 不停止轴突的生长，以允许其连接更多树突
                            # 如果希望每个轴突只连接一个树突，可以取消注释以下行
                            # break

    def connect_dendrite_to_axon(self):
        """检查树突是否连接到其他神经元的轴突，基于兴奋值决定连接"""
        global connections
        for dendrite in self.dendrites:
            if dendrite['stopped']:
                continue
            last_dendrite = dendrite['path'][-1]
            for other_node in node:
                if other_node != self:
                    for axon in other_node.axons:
                        if axon['stopped']:
                            continue
                        last_axon = axon['path'][-1]
                        distance = math.sqrt(
                            (last_dendrite[0] - last_axon[0]) ** 2 +
                            (last_dendrite[1] - last_axon[1]) ** 2 +
                            (last_dendrite[2] - last_axon[2]) ** 2
                        )
                        if distance < LinkThreshold:
                            # 根据兴奋值决定是否连接
                            excitation_product = dendrite['excitation'] * axon['excitation']
                            if excitation_product > 25:  # 25对应 (5 * 5)，阈值可调节
                                connections.append((other_node.id, axon['path'][-1], self.id, dendrite['path'][-1]))
                                dendrite['stopped'] = True  # 停止该树突的生长
                                axon['stopped'] = True      # 停止该轴突的生长
                                # 如果希望每个树突只连接一个轴突，可以取消注释以下行
                                # break

# ---------------------------- 初始化神经元 ----------------------------

for i in range(num_of_node):
    node.append(NeuralNode(
        i,
        random.uniform(20, MaxWidth - 20),
        random.uniform(20, MaxLength - 20),
        random.uniform(20, MaxHeight - 20)
    ))

# ---------------------------- 创建绘图窗口 ----------------------------

fig = plt.figure()
ax_plot = fig.add_subplot(111, projection='3d')

# 设置图像范围和标签
ax_plot.set_xlim([0, MaxWidth])
ax_plot.set_ylim([0, MaxLength])
ax_plot.set_zlim([0, MaxHeight])
ax_plot.set_xlabel('X')
ax_plot.set_ylabel('Y')
ax_plot.set_zlabel('Z')

# ---------------------------- 模拟循环 ----------------------------

for i in range(epoch):
    ax_plot.cla()  # 清除之前的绘图
    ax_plot.set_xlim([0, MaxWidth])
    ax_plot.set_ylim([0, MaxLength])
    ax_plot.set_zlim([0, MaxHeight])
    ax_plot.set_xlabel('X')
    ax_plot.set_ylabel('Y')
    ax_plot.set_zlabel('Z')

    # 更新树突和轴突的生长
    for j in range(len(node)):
        current_node = node[j]

        # 生长树突
        for dendrite in current_node.dendrites:
            current_node.grow_dendrite(dendrite)
            path = dendrite['path']
            if len(path) > 1:
                xs, ys, zs = zip(*path)
                ax_plot.plot(xs, ys, zs, color=current_node.color_dendrite, linestyle='dashed', alpha=0.5)

        # 生长轴突
        for axon in current_node.axons:
            current_node.grow_axon(axon)
            path = axon['path']
            if len(path) > 1:
                xs, ys, zs = zip(*path)
                ax_plot.plot(xs, ys, zs, color=current_node.color_axon, linestyle='solid', alpha=0.5)

        # 检查轴突是否连接到其他神经元的树突
        current_node.connect_axon_to_dendrite()

        # 检查树突是否连接到其他神经元的轴突
        current_node.connect_dendrite_to_axon()

    # 绘制神经元位置
    for n in node:
        ax_plot.scatter(n.x, n.y, n.z, color='black', s=50)  # 使用统一颜色表示神经元位置

    # 绘制已建立的连接（高亮显示）
    for conn in connections:
        _, axon_point, _, dendrite_point = conn
        # 绘制连接线
        ax_plot.plot(
            [axon_point[0], dendrite_point[0]],
            [axon_point[1], dendrite_point[1]],
            [axon_point[2], dendrite_point[2]],
            color='red', linewidth=2, alpha=0.8  # 高亮用红色粗线
        )
        # 用红色圈出连接点
        ax_plot.scatter(axon_point[0], axon_point[1], axon_point[2], color='red', s=50, marker='o')
        ax_plot.scatter(dendrite_point[0], dendrite_point[1], dendrite_point[2], color='red', s=50, marker='o')

    ax_plot.set_title(f"神经生长模拟 第 {i+1} 次迭代")

    # 实时更新图像
    plt.draw()
    plt.pause(0.05)  # 根据需要调整暂停时间

# ---------------------------- 保存结果 ----------------------------

# 保存最终图像
plt.savefig('neural_growth_final.png')

# 记录连接信息到文本文件
with open(log_file_path, 'w') as f:
    for conn in connections:
        f.write(f"神经元 {conn[0]} 的轴突在位置 {conn[1]} 连接到神经元 {conn[2]} 的树突在位置 {conn[3]}\n")

# 保持图像窗口不关闭
plt.show()

print("实时生长可视化已显示，连接信息已保存到文件。")
