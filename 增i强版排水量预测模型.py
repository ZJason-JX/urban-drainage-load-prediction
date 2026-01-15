"""''"增强排水量预测模型
预测公式：排水量 = w1 * 居民用水量 + w2 * 降水量  + w3 * 工业排水+b
根据真实情况，加入周期型，不再使用简单线性模型"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#可视化图表部分
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_results(model, x1_train, x2_train, x3_train, y_train,
                 x1_test, x2_test, x3_test, y_test,
                 train_predictions, test_predictions, losses=None):
    """
    绘制四张可视化图表
    """
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('排水量预测模型可视化分析', fontsize=16, fontweight='bold')

    # 1. Loss曲线图（如果记录了losses）
    ax1 = axes[0, 0]
    if losses is not None:
        ax1.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Loss训练曲线')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 使用对数坐标更清晰
    else:
        ax1.text(0.5, 0.5, '未记录Loss数据', ha='center', va='center', fontsize=12)
        ax1.set_title('Loss训练曲线')

    # 2. 风险情况统计图（预测误差分布）
    ax2 = axes[0, 1]
    # 计算训练集和测试集的误差
    train_errors = train_predictions - y_train
    test_errors = test_predictions - y_test

    # 合并所有误差
    all_errors = np.concatenate([train_errors, test_errors])

    ax2.hist(all_errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax2.axvline(x=all_errors.mean(), color='green', linestyle='-', linewidth=2, label=f'均值: {all_errors.mean():.1f}')

    # 标注统计信息
    stats_text = f'误差统计:\n均值: {all_errors.mean():.1f}\n标准差: {all_errors.std():.1f}\n最大值: {all_errors.max():.1f}\n最小值: {all_errors.min():.1f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('预测误差')
    ax2.set_ylabel('频次')
    ax2.set_title('预测误差分布（风险统计）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 真实值 vs 预测值对比图
    ax3 = axes[1, 0]
    days = range(len(y_test))
    ax3.plot(days, y_test, 'b-o', linewidth=2, markersize=6, label='真实值')
    ax3.plot(days, test_predictions, 'r-s', linewidth=2, markersize=6, label='预测值')

    # 填充误差区域
    ax3.fill_between(days, y_test, test_predictions,
                     where=(test_predictions > y_test),
                     color='red', alpha=0.2, label='高估')
    ax3.fill_between(days, y_test, test_predictions,
                     where=(test_predictions <= y_test),
                     color='blue', alpha=0.2, label='低估')

    ax3.set_xlabel('测试集天数')
    ax3.set_ylabel('排水量 (吨)')
    ax3.set_title('测试集：真实值 vs 预测值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 添加预测误差标注
    for i, (true, pred) in enumerate(zip(y_test, test_predictions)):
        error = pred - true
        ax3.annotate(f'{error:+.0f}',
                     xy=(i, max(true, pred) + 10),
                     ha='center', fontsize=8, color='green')

    # 4. 特征重要性分析图
    ax4 = axes[1, 1]

    # 计算各特征的平均值
    avg_x1 = np.mean(x1_train)
    avg_x2 = np.mean(x2_train)
    avg_x3 = np.mean(x3_train)

    # 计算各特征对预测的贡献度
    contributions = [
        model.w1 * avg_x1,
        model.w2 * avg_x2,
        model.w3 * avg_x3,
        model.b
    ]

    labels = ['居民用水贡献', '降雨量贡献', '工业排水贡献', '基础偏移量']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax4.bar(labels, contributions, color=colors, alpha=0.8)
    ax4.set_ylabel('贡献度 (吨)')
    ax4.set_title('各特征对排水量的贡献度')

    # 在每个柱子上添加数值
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # 添加比例标注
    total = sum(contributions)
    for i, (label, contrib) in enumerate(zip(labels, contributions)):
        percentage = (contrib / total) * 100
        ax4.text(i, -total * 0.05, f'{percentage:.1f}%',
                 ha='center', va='top', fontsize=9, fontweight='bold')

    ax4.grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout()
    plt.show()

    # 额外创建一个时间序列分析图（可选，单独显示）
    plt.figure(figsize=(12, 6))

    # 绘制完整35天的排水量
    full_days = range(35)
    full_residential = np.concatenate([x1_train, x1_test])
    full_rainfall = np.concatenate([x2_train, x2_test])
    full_industrial = np.concatenate([x3_train, x3_test])
    full_actual = np.concatenate([y_train, y_test])

    # 创建预测值（训练集用训练预测，测试集用测试预测）
    full_predictions = np.concatenate([train_predictions, test_predictions])

    plt.plot(full_days, full_actual, 'b-', linewidth=2, alpha=0.7, label='真实排水量')
    plt.plot(full_days, full_predictions, 'r--', linewidth=2, alpha=0.7, label='预测排水量')

    # 标记训练集和测试集分界线
    plt.axvline(x=29.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    plt.text(15, plt.ylim()[1] * 0.95, '训练集', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.text(32.5, plt.ylim()[1] * 0.95, '测试集', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.fill_between(full_days, full_actual, full_predictions,
                     where=(full_predictions >= full_actual),
                     color='red', alpha=0.2, label='高估区域')
    plt.fill_between(full_days, full_actual, full_predictions,
                     where=(full_predictions < full_actual),
                     color='blue', alpha=0.2, label='低估区域')

    plt.xlabel('天数')
    plt.ylabel('排水量 (吨)')
    plt.title('完整时间序列：真实排水量 vs 预测排水量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
#数据准备
#假设我有30天训练数据 + 7天未来真实值
#居民用水量（吨/天）
residential_water = np.array([
    # 工作日较高，周末较低
    130, 128, 132, 135, 128, 122, 118,  # 第一周
    116, 120, 125, 130, 133, 128, 122,  # 第二周
    120, 118, 115, 112, 110, 108, 125,  # 第三周
    128, 130, 132, 128, 125, 122, 120,  # 第四周
    118, 125, 130, 135, 132, 128, 125  # 训练数据最后几天 + 测试数据开始
])
# 降水量（毫米/天）
rainfall = np.array([
    0, 5, 2, 0, 10, 8, 3,  # 第一周
    1, 0, 2, 5, 12, 15, 20,  # 第二周（雨季）
    8, 3, 0, 0, 1, 4, 6,  # 第三周
    10, 5, 2, 0, 0, 0, 0,  # 第四周
    2, 8, 15, 5, 0, 3, 10  # 训练数据最后几天 + 测试数据开始
])
#工业排水 - 新增特征
#工作日较高，周末较低
industrial_drainage = np.array([
    80, 85, 88, 90, 85, 40, 35,  # 周末工业排水减少
    38, 82, 86, 89, 92, 87, 42,  # 第二周
    36, 80, 84, 88, 90, 86, 38,  # 第三周
    34, 83, 87, 89, 91, 45, 37,  # 第四周
    39, 84, 88, 90, 86, 43, 41  # 训练数据最后几天 + 测试数据开始
])
# 实际排水量（吨/天） - (使用周期性权重)
actual_drainage = np.zeros(35)
weekly_weights = {
    0: {'w1': 5.2, 'w2': 14.5, 'w3': 0.9, 'b': 25},  # 周一
    1: {'w1': 5.0, 'w2': 15.0, 'w3': 1.0, 'b': 30},  # 周二
    2: {'w1': 4.8, 'w2': 14.8, 'w3': 1.1, 'b': 28},  # 周三
    3: {'w1': 5.1, 'w2': 15.2, 'w3': 0.95, 'b': 26},  # 周四
    4: {'w1': 4.9, 'w2': 14.9, 'w3': 1.05, 'b': 32},  # 周五
    5: {'w1': 4.5, 'w2': 16.0, 'w3': 0.5, 'b': 20},  # 周六（工业排水权重降低）
    6: {'w1': 4.6, 'w2': 15.5, 'w3': 0.4, 'b': 18}  # 周日
}
for i in range(35):
    day_of_week = i % 7
    weights = weekly_weights[day_of_week]
    #基础排水量 = w1*居民用水 + w2 * 降雨量 + w3工业排水 + b
    base_drainage = (
        weights['w1'] * residential_water[i] +
        weights['w2'] * rainfall[i] +
        weights['w3'] * industrial_drainage[i] +
        weights['b']
    )
#添加随机噪声（保留5%左右的波动）
    noise = np.random.normal(0,base_drainage * 0.05)
    actual_drainage[i] = base_drainage + noise
    train_size = 30
    x1_train = residential_water[:train_size]  # 居民用水量
    x2_train = rainfall[:train_size]  # 降水量
    x3_train = industrial_drainage[:train_size]  # 工业排水
    y_train = actual_drainage[:train_size]  # 实际排水量

    x1_test = residential_water[train_size:]  # 测试集居民用水量
    x2_test = rainfall[train_size:]  # 测试集降水量
    x3_test = industrial_drainage[train_size:]  # 测试集工业排水
    y_test = actual_drainage[train_size:]  # 测试集实际排水量
print("数据生成完成！")
print(f"训练集: {len(x1_train)} 天")
print(f"测试集: {len(x1_test)} 天")
print(f"居民用水量范围: {residential_water.min():.1f} - {residential_water.max():.1f} 吨")
print(f"降水量范围: {rainfall.min():.1f} - {rainfall.max():.1f} 毫米")
print(f"工业排水范围: {industrial_drainage.min():.1f} - {industrial_drainage.max():.1f} 吨")
print(f"排水量范围: {actual_drainage.min():.1f} - {actual_drainage.max():.1f} 吨\n")

class EnhanceDrainageModel :
    #增强版模型：三变量，周期性变化
    def __init__(self):
        self.w1 = 5 #初始居民用水量权重
        self.w2 = 15 #初始降雨量权重
        self.w3 = 1.0 #初始工业排水权重
        self.b = 0 #初始排水偏移

    def predict(self,x1,x2,x3):
        """根据x1和x2预测排水量
        公式 y_pred = w1 * x + w2 * x2 + w3 * x3+ b
        """
        y_pred = self.w1 * x1 + self.w2 * x2 +self.w3 * x3 +self.b
        return y_pred
    def  compute_loss(self,y_true,y_pred):
        loss = np.mean((y_pred - y_true) **2)
        return loss
    def compute_gradients(self,x1,x2,x3,y_pred,y_true):
        n = len(y_true)
        dw1 = (2/n) * np.sum((y_pred - y_true) * x1)
        dw2 = (2/n) * np.sum((y_pred - y_true) * x2)
        dw3 = (2 / n) * np.sum((y_pred - y_true) * x3)
        db = (2/n) * np.sum((y_pred - y_true))
        return dw1,dw2,dw3,db
    def update_parameters(self,dw1,dw2,dw3,db,learning_rate):
        self.w1= self.w1 - dw1 * learning_rate
        self.w2= self.w2 - dw2 * learning_rate
        self.w3= self.w3 - dw3 * learning_rate
        self.b = self.b - db * learning_rate
    def train(self,x1,x2,x3,y_true,learning_rate ,epochs ):

        print('开始训练模型')
        losses = []
        for epoch in range(epochs):
            y_pred = self.predict(x1,x2,x3)
            loss = self.compute_loss(y_true,y_pred)
            losses.append(loss)#记录loss
            dw1,dw2,dw3,db = self.compute_gradients(x1,x2,x3,y_pred,y_true)
            self.update_parameters(dw1,dw2,dw3,db,learning_rate)
            if (epoch+1) % 100 == 0:
                print(f'第{epoch + 1:4d}轮 | loss:{loss:.2f} | ' f'参数： w1 = {self.w1:.4f},w2 = {self.w2:.4f},b = {self.b:.4f}')
        return losses
def main():
    #1.创建模型
    print('1.创建模型实例')
    model = EnhanceDrainageModel()
    print(f' 初始参数：w1 = {model.w1} , w2 = {model.w2} ,w3 = {model.w3} , b = {model.b}')
    #2.训练模型
    print("\n2. 开始训练...")
    print(f"   学习率: 0.00001")
    print(f"   训练轮数: 5000\n")
    losses = model.train(x1_train,x2_train,x3_train,y_train,learning_rate = 0.00001,epochs =5000)
    #3.训练集预测结果
    print('\n3.预测训练集结果')
    train_predictions = model.predict(x1_train,x2_train,x3_train)
    train_loss = model.compute_loss(y_train,train_predictions)
    print(f'训练集损失（MSE ): {train_loss:.2f}')
    #4.训练集测试（未来五天）
    print('\n4.测试集预测结果(5天)')
    test_predictions = model.predict(x1_test,x2_test,x3_test)
    test_loss = model.compute_loss(y_test,test_predictions)
    print(f'测试集损失（MSE）: {test_loss:.2f}')
    #显示每天的预测误差
    print('\n每日预测结果对比：')
    print('天数 | 真实值 | 预测值 | 误差 | 误差%')
    print('-' * 50)
    for i in range(len(y_test)):
        true_value = y_test[i]
        pred_value = test_predictions[i]
        error = pred_value - true_value
        error_percent = (error / true_value) * 100
        print(f'{i + 1:2d} | {true_value:6.1f} | {pred_value:6.1f} | {error:6.1f} | {error_percent:5.1f}%')
    # 5. 模型分析
    print('\n 5.模型分析...')
    print(f"学习到的公式：")
    print(f"排水量 = {model.w1:.2f}×居民用水量 + {model.w2:.2f}×降水量 + {model.w3:.2f}×工业排水 + {model.b:.2f}")
    # 与平均权重比较（用于评估模型是否学到周期性）
    avg_w1 = sum([weekly_weights[d]['w1'] for d in range(7)]) / 7
    avg_w2 = sum([weekly_weights[d]['w2'] for d in range(7)]) / 7
    avg_w3 = sum([weekly_weights[d]['w3'] for d in range(7)]) / 7
    avg_b = sum([weekly_weights[d]['b'] for d in range(7)]) / 7
    print(f"\n真实平均权重（一周平均）:")
    print(f"w1={avg_w1:.2f}, w2={avg_w2:.2f}, w3={avg_w3:.2f}, b={avg_b:.2f}")

    # 6. 预测新数据
    print("\n5. 预测新数据...")
    new_residential = 125  # 新的居民用水量
    new_rainfall = 5  # 新的降水量
    new_industrial = 85  # 新的工业排水
    prediction = model.predict(new_residential, new_rainfall,new_industrial)# 新的工业排水)
    print(f"   输入：居民用水量={new_residential}吨, 降水量={new_rainfall}毫米,工业排水={new_industrial}吨")
    print(f"   预测排水量：{prediction:.2f}吨")
    # 7. 可视化分析
    print("\n7. 生成可视化图表...")

    # 注意：为了绘制loss曲线，需要修改train函数以记录loss
    # 这里假设你修改了train函数，让它返回losses列表
    # 如果还没有修改，可以先传入None

    # 调用可视化函数
    plot_results(model, x1_train, x2_train, x3_train, y_train,
                 x1_test, x2_test, x3_test, y_test,
                 train_predictions, test_predictions,
                 losses=losses)  # 如果记录了losses就传入，否则None


# 运行程序
if __name__ == "__main__":
    main()