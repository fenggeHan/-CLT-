import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform, skew
import platform

# --- 1. 环境配置与中文适配 ---
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="高级 CLT 仿真平台", layout="wide")

st.title("📊 中心极限定理 (CLT) 深度交互仿真平台")
st.markdown("本系统通过自定义母体分布的**核心数学参数**，演示均值序列如何向正态分布收敛。")

# --- 2. 侧边栏：参数输入模块 ---
st.sidebar.header("🔧 配置模拟参数")

dist_list = [
    "均匀分布 (Uniform)", 
    "泊松分布 (Poisson)", 
    "指数分布 (Exponential)", 
    "正态分布 (Normal)", 
    "0-1 分布 (Bernoulli)", 
    "二项分布 (Binomial)", 
    "几何分布 (Geometric)", 
    "卡方分布 (Chi-Square)", 
    "t 分布", 
    "F 分布"
]

dist_type = st.sidebar.selectbox("选择母体分布类型", dist_list)

# --- 3. 动态分布参数设置 (专利亮点：精细化参数控制) ---
st.sidebar.subheader("母体分布自定义参数")

params = {} # 用于存储用户输入的特定参数

# 针对你要求的四个分布进行详细参数设置
if dist_type == "均匀分布 (Uniform)":
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        params['a'] = st.number_input("区间左端点 a", value=0.0)
    with col_b:
        params['b'] = st.number_input("区间右端点 b", value=10.0)
    if params['a'] >= params['b']:
        st.error("错误：下限 a 必须小于上限 b")

elif dist_type == "泊松分布 (Poisson)":
    # 泊松分布参数 λ (在专利中可表述为强度参数 theta)
    params['theta'] = st.sidebar.number_input("参数 θ (单位时间内事件发生率)", min_value=0.1, value=3.0, step=0.5)

elif dist_type == "指数分布 (Exponential)":
    # 指数分布参数 λ
    params['lambda'] = st.sidebar.number_input("参数 λ (Rate parameter)", min_value=0.01, value=1.0, step=0.1)

elif dist_type == "正态分布 (Normal)":
    col_mu, col_std = st.sidebar.columns(2)
    with col_mu:
        params['mu'] = st.number_input("母体均值 μ", value=0.0)
    with col_std:
        params['sigma'] = st.number_input("母体标准差 σ", min_value=0.01, value=1.0)

# 其他辅助分布参数
elif dist_type == "0-1 分布 (Bernoulli)":
    params['p'] = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5)
elif dist_type == "二项分布 (Binomial)":
    params['n_trial'] = st.sidebar.slider("试验次数 n", 1, 100, 10)
    params['p'] = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5)
elif dist_type == "卡方分布 (Chi-Square)":
    params['df'] = st.sidebar.slider("自由度 df", 1, 50, 5)
elif dist_type == "t 分布":
    params['df'] = st.sidebar.slider("自由度 df", 1, 100, 10)
elif dist_type == "F 分布":
    params['dfn'] = st.sidebar.slider("分子自由度 dfn", 1, 100, 10)
    params['dfd'] = st.sidebar.slider("分母自由度 dfd", 1, 100, 20)

# 核心抽样参数调节
st.sidebar.subheader("CLT 抽样规模控制")
n_sample = st.sidebar.slider("单次样本容量 (n)", 1, 5000, 30)
N_sim = st.sidebar.slider("总模拟抽样次数 (N)", 100, 10000, 2000)

# --- 4. 核心计算模块 (数据处理单元) ---
def get_sample_means(dist, p, n, N):
    if dist == "均匀分布 (Uniform)":
        # loc 是起点, scale 是区间长度
        return np.mean(uniform.rvs(loc=p['a'], scale=p['b'] - p['a'], size=(N, n)), axis=1)
    
    elif dist == "泊松分布 (Poisson)":
        return np.mean(poisson.rvs(mu=p['theta'], size=(N, n)), axis=1)
    
    elif dist == "指数分布 (Exponential)":
        # Scipy expon 中 scale = 1/lambda
        return np.mean(expon.rvs(scale=1/p['lambda'], size=(N, n)), axis=1)
    
    elif dist == "正态分布 (Normal)":
        return np.mean(norm.rvs(loc=p['mu'], scale=p['sigma'], size=(N, n)), axis=1)
    
    elif dist == "0-1 分布 (Bernoulli)":
        return np.mean(bernoulli.rvs(p['p'], size=(N, n)), axis=1)
    
    elif dist == "二项分布 (Binomial)":
        return np.mean(binom.rvs(p['n_trial'], p['p'], size=(N, n)), axis=1)
    
    elif dist == "几何分布 (Geometric)":
        return np.mean(geom.rvs(0.5, size=(N, n)), axis=1)
    
    elif dist == "卡方分布 (Chi-Square)":
        return np.mean(chi2.rvs(df=p['df'], size=(N, n)), axis=1)
    
    elif dist == "t 分布":
        return np.mean(t.rvs(df=p['df'], size=(N, n)), axis=1)
    
    elif dist == "F 分布":
        return np.mean(f.rvs(dfn=p['dfn'], dfd=p['dfd'], size=(N, n)), axis=1)
    
    return np.zeros(N)

# 执行仿真计算
try:
    means = get_sample_means(dist_type, params, n_sample, N_sim)

    # --- 5. 可视化渲染模块 (专利：图像叠加显示技术) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    # 绘制样本均值的直方图
    ax.hist(means, bins=60, density=True, alpha=0.6, color='#1f77b4', label='样本均值观测分布')
    
    # 自动计算拟合正态曲线的参数
    mu_fit, std_fit = norm.fit(means)
    x_range = np.linspace(min(means), max(means), 200)
    y_pdf = norm.pdf(x_range, mu_fit, std_fit)
    
    # 在同一坐标系叠加理论曲线
    ax.plot(x_range, y_pdf, 'r--', lw=2, label=f'理论正态拟合曲线\n($\mu={mu_fit:.2f}, \sigma={std_fit:.2f}$)')
    
    ax.set_title(f"{dist_type} 母体在样本量 n={n_sample} 时的均值分布结果", fontsize=14)
    ax.set_xlabel("样本均值 (Sample Mean)")
    ax.set_ylabel("概率密度 (Density)")
    ax.legend()
    st.pyplot(fig)

    # --- 6. 统计面板 (数据分析单元) ---
    st.subheader("📋 仿真数据分析")
    c1, c2, c3 = st.columns(3)
    c1.metric("模拟期望值 (Mean)", f"{mu_fit:.4f}")
    c2.metric("模拟标准误 (Std Error)", f"{std_fit:.4f}")
    c3.metric("分布偏度 (Skewness)", f"{skew(means):.4f}")

except Exception as e:
    st.warning("请确保输入的分布参数合法。")

st.info("💡 专利技术总结：本系统通过『动态参数映射算法』将抽象的分布参数（如λ、θ）实时转化为大规模随机矩阵，并利用向量化均值运算展示了中心极限定理的普遍适用性。")
