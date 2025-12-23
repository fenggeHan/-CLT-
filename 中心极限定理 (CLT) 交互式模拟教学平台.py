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
st.markdown("本系统通过自定义母体分布参数，演示均值序列如何向正态分布收敛。")

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

params = {}  # 存储特定分布参数

# 设置各个分布的参数
if dist_type == "均匀分布 (Uniform)":
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        params['a'] = st.sidebar.number_input("区间下限 a", value=0.0)
    with col_b:
        params['b'] = st.sidebar.number_input("区间上限 b", value=1.0)
    if params['a'] >= params['b']:
        st.error("错误：下限 a 必须小于上限 b")

elif dist_type == "泊松分布 (Poisson)":
    params['theta'] = st.sidebar.slider("参数 θ (Lambda)", 0.1, 20.0, 3.0)

elif dist_type == "指数分布 (Exponential)":
    params['lambda'] = st.sidebar.slider("参数 λ (Rate)", 0.1, 5.0, 1.0)

elif dist_type == "正态分布 (Normal)":
    params['mu'] = st.sidebar.number_input("均值 μ", value=0.0)
    params['sigma'] = st.sidebar.number_input("标准差 σ", value=1.0, min_value=0.01)

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

# 核心抽样参数
st.sidebar.subheader("CLT 抽样参数")
n_sample = st.sidebar.slider("样本容量 (n)", 1, 5000, 30)
N_sim = st.sidebar.slider("模拟次数 (N)", 100, 10000, 2000)

# --- 4. 核心计算模块 ---
def get_sample_means(dist, p, n, N):
    if dist == "均匀分布 (Uniform)":
        data = uniform.rvs(loc=p['a'], scale=p['b'] - p['a'], size=(N, n))
    elif dist == "泊松分布 (Poisson)":
        data = poisson.rvs(mu=p['theta'], size=(N, n))
    elif dist == "指数分布 (Exponential)":
        # Scipy expon 中 scale = 1/lambda
        data = expon.rvs(scale=1/p['lambda'], size=(N, n))
    elif dist == "正态分布 (Normal)":
        data = norm.rvs(loc=p['mu'], scale=p['sigma'], size=(N, n))
    elif dist == "0-1 分布 (Bernoulli)":
        data = bernoulli.rvs(p['p'], size=(N, n))
    elif dist == "二项分布 (Binomial)":
        data = binom.rvs(p['n_trial'], p['p'], size=(N, n))
    elif dist == "几何分布 (Geometric)":
        data = geom.rvs(0.5, size=(N, n))
    elif dist == "卡方分布 (Chi-Square)":
        data = chi2.rvs(df=p['df'], size=(N, n))
    elif dist == "t 分布":
        data = t.rvs(df=p['df'], size=(N, n))
    elif dist == "F 分布":
        data = f.rvs(dfn=p['dfn'], dfd=p['dfd'], size=(N, n))
    return np.mean(data, axis=1)

# 执行计算
try:
    means = get_sample_means(dist_type, params, n_sample, N_sim)

    # --- 5. 可视化渲染 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(means, bins=60, density=True, alpha=0.6, color='#1f77b4', label='样本均值分布直方图')

    # 叠加正态拟合曲线
    mu_fit, std_fit = norm.fit(means)
    x_range = np.linspace(min(means), max(means), 200)
    y_pdf = norm.pdf(x_range, mu_fit, std_fit)
    ax.plot(x_range, y_pdf, 'r--', lw=2, label=f'拟合正态曲线\n($\mu={mu_fit:.2f}, \sigma={std_fit:.2f}$)')
    
    ax.set_title(f"{dist_type} (n={n_sample}) 的均值分布", fontsize=14)
    ax.set_xlabel("均值数值")
    ax.set_ylabel("概率密度")
    ax.legend()
    st.pyplot(fig)

    # --- 6. 统计面板 ---
    st.subheader("📋 统计推断结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("模拟均值 Expectation", f"{mu_fit:.4f}")
    c2.metric("标准误 Standard Error", f"{std_fit:.4f}")
    c3.metric("偏度 Skewness", f"{skew(means):.4f}")

except Exception as e:
    st.warning(f"等待输入有效参数... (Error: {e})")

st.info("💡 专利技术说明：本系统实现了“参数驱动型母体建模”，允许用户通过调整底层分布的矩参数（如指数分布的 λ），观察其对大样本收敛速率的影响。")
