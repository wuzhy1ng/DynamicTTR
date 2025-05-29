import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

tx_counts = [
    17893, 334, 237, 195, 672, 7439, 46,
    421, 24, 4741, 510, 1306, 111, 2806,
    3371, 758, 59550, 2401, 51,
    1805890, 48, 2923, 251
]
time_cost = [36074.19420322458, 35526.01972966804, 78148.58867924528, 20921.629959327758, 15549.885733200927,
             35159.39034603371, 15333.226098704601, 9666.887384008978, 10506.554221897506, 19011.175482373368,
             4877.987412204689, 14884.45168320114, 12802.280811747236, 49932.82601261768, 11944.738395929377,
             30971.762885894925, 35140.96977621815, 30743.495674154216, 10534.300403821531, 53091.009572662064,
             23981.726265634305, 27642.38499278499, 22783.78392884195]
# time_cost = [
#     1.547260893767986, 8.19416056968146, 58.32145389824381, 143.21384914808618, 13.570329915652632, 15.16296657809591,
#     166.22296199421393, 12.266987991272762, 316.8621288811664, 4.454736824217398, 12.704660764102393,
#     10.076139729640778, 125.01651266766021, 4.239968460317621, 4.78587117879764, 29.006452326779577, 2.0229511628232255,
#     50.935745946930595, 165.8876760006328, 0.03260828462192183, 63.2443794387732, 12.541987494804452, 62.25866496459796
# ]


# 转换为NumPy数组
tx_counts = np.array(tx_counts)
time_cost = np.array(time_cost)

# 按 tx_counts 排序
sort_idx = np.argsort(tx_counts)
x = tx_counts[sort_idx]
y = time_cost[sort_idx]

# 关键改进 1：对数变换处理量级差异
log_x = np.log(x)  # 对自变量取对数

# 关键改进 2：使用更合理的非线性函数
def rational_func(x, a, b, c, d):
    """ (a + b*x) / (1 + c*x + d*x**2) """
    return (a + b * x) / (1 + c * x + d * x ** 2)


# 关键改进 3：加权拟合（抑制超大值影响）
weights = 1 / np.sqrt(x)  # 给中小值更高权重

params, _ = curve_fit(
    rational_func, log_x, y,
    p0=[1, 1, 0.1, 0.1],
    maxfev=5000,
    sigma=weights
)

# 生成平滑曲线
x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
log_x_new = np.log(x_new)
y_new = rational_func(log_x_new, *params)

# 绘图
plt.figure(figsize=(12, 7))
plt.scatter(x, y, color='darkorange', label='原始数据', zorder=3)
plt.plot(x_new, y_new, 'steelblue', linewidth=2.5, label='拟合曲线')
plt.xscale('log')  # 对数坐标显示
plt.xlabel("tx_counts (对数坐标)", fontsize=12)
plt.ylabel("time_cost", fontsize=12)
plt.yscale('log')
plt.title("tx_counts 与 time_cost 的非线性关系（改进拟合）", pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# 输出函数关系
print(f"拟合函数：")
print(f"time_cost = ({params[0]:.2f} + {params[1]:.2f}*ln(x)) / (1 + {params[2]:.2f}*ln(x) + {params[3]:.2f}*ln(x)^2)")